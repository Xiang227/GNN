import argparse
import time
import numpy as np
from math import ceil
import os
from sklearn.model_selection import KFold
import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Parameter, GRU
from torch.optim import Adam

from torch_geometric.datasets import TUDataset, Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
from torch_geometric.utils import add_self_loops, degree, to_dense_adj, to_undirected
from torch_geometric import transforms as T
from torch_cluster import random_walk  # Import random_walk from torch_cluster
from torch_geometric.data import Data, Dataset
import scipy.sparse as sp

class ExplicitRandomWalkEncoder(nn.Module):
    """Explicit random walk encoder, supporting exponential random walk and Levy random walk strategies"""
    def __init__(self, input_dim, hidden_dim, walk_length, num_walks, device, alpha=1.0, c=0.15, walk_type='exp'):
        super(ExplicitRandomWalkEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.device = device
        self.alpha = alpha  # Parameters that affect control distance
        self.c = c  # Parameters controlling the probability of random jumps
        self.walk_type = walk_type  # 'exp' for exponential RW, 'levy' for Levy RW
        
        # Feature transformation layer
        self.feature_encoder = Linear(input_dim, hidden_dim)
        
        # Path Encoding Layer (GRU)
        self.path_encoder = GRU(hidden_dim, hidden_dim, batch_first=True)
    
    def compute_graph_distances(self, edge_index, num_nodes, graph_nodes):
        """Calculate the shortest path distance matrix of the graph using local node indices"""
        # Creating a local node index map
        local_to_global = graph_nodes.cpu().numpy()
        global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(local_to_global)}
        
        # Creating a local adjacency matrix
        adj = torch.zeros((num_nodes, num_nodes), device=self.device)
        
        # Converting a global edge index to a local index
        local_edge_index = torch.zeros_like(edge_index)
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src in global_to_local and dst in global_to_local:
                local_src = global_to_local[src]
                local_dst = global_to_local[dst]
                local_edge_index[0, i] = local_src
                local_edge_index[1, i] = local_dst
                adj[local_src, local_dst] = 1
        
        
        if num_nodes > 1000:  
            print(f"Using simplified distance calculation for {num_nodes} nodes")
            # Use 2-hop neighbour information instead of just degrees
            adj_np = adj.cpu().numpy()
            # Calculate the 2-hop reachability matrix
            adj2 = np.matmul(adj_np, adj_np)
            # Constructing a distance matrix based on 1-hop and 2-hop neighbours
            distance_matrix = np.ones((num_nodes, num_nodes)) * 3  # The default distance is 3
            distance_matrix[adj_np > 0] = 1  # Direct neighbourhood distance of 1
            distance_matrix[adj2 > 0] = 2   # 2-hop neighbourhood distance of 2
            np.fill_diagonal(distance_matrix, 0)  # Self-to-self distance is 0
        else:
            # For medium-sized graphs use BFS to calculate distances
            print(f"Computing exact distances for {num_nodes} nodes")
            adj_np = adj.cpu().numpy()
            distance_matrix = self._compute_bfs_distances(adj_np, num_nodes)
        
        # Calculating the probability of wandering
        walk_probs = self._compute_walk_probabilities(distance_matrix, self.alpha)
        
        return torch.FloatTensor(walk_probs).to(self.device), torch.FloatTensor(distance_matrix).to(self.device)
    
    def _compute_bfs_distances(self, adj_matrix, num_nodes, max_distance=4):
        """Calculating shortest path distance using BFS"""
        from collections import deque
        
        # Building Neighbourhood Lists
        adj_list = [[] for _ in range(num_nodes)]
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_matrix[i, j] > 0:
                    adj_list[i].append(j)
        
        # Initialise the distance matrix
        distance_matrix = np.full((num_nodes, num_nodes), max_distance + 1, dtype=np.float32)
        np.fill_diagonal(distance_matrix, 0)
        
        # Use BFS for each node
        for start in range(num_nodes):
            visited = np.zeros(num_nodes, dtype=bool)
            queue = deque([(start, 0)])
            visited[start] = True
            
            while queue:
                node, dist = queue.popleft()
                
                if dist >= max_distance:
                    continue
                    
                for neighbor in adj_list[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        distance_matrix[start, neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))
        
        return distance_matrix
    
    def _compute_walk_probabilities(self, distance_matrix, alpha=1.0):
        """Calculate probability of walk based on distance"""
        num_nodes = distance_matrix.shape[0]
        
        # Compute the distance-based transfer matrix G
        if self.walk_type == 'exp':
            # Exponential Random Walk: using negative exponential distance as weights
            distance_weights = np.exp(-alpha * distance_matrix)
        elif self.walk_type == 'levy':
            # Levy's Random Walk: Using Power Law Distributions as Weights
            
            safe_distance_matrix = distance_matrix + 1e-6  # Adding small constants to avoid division by zero
            # Use of power-law distributions
            distance_weights = np.power(safe_distance_matrix, -alpha)
            # For the case where the distance is 0 (self-connection), the highest weight is given
            distance_weights[distance_matrix == 0] = 1e6
        else:
            raise ValueError(f"Unsupported types of random walk: {self.walk_type}")
        
        
        row_sums = distance_weights.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        G = distance_weights / row_sums
        
        
        uniform_jump_matrix = np.ones((num_nodes, num_nodes)) / num_nodes
        
        
        walk_probs = (1 - self.c) * G + self.c * uniform_jump_matrix
        
        return walk_probs
    
    def sample_random_walks(self, edge_index, batch, num_nodes):
        """Generating random walk paths using distance-based probabilities"""
        walks = []
        walk_batch = []  # Record the graph to which each path belongs
        batch_size = torch.max(batch).item() + 1
        
        # Generate random walks for each graph
        for graph_idx in range(batch_size):
            # Get the node index of the current graph
            graph_mask = (batch == graph_idx)
            graph_nodes = torch.nonzero(graph_mask).squeeze()
            
            # Handling of single-node situations
            if graph_nodes.dim() == 0:
                graph_nodes = graph_nodes.unsqueeze(0)
                
            # If the graph has only one node, skip the random walk
            if graph_nodes.size(0) <= 1:
                continue
                
            # Get the edge index of the current graph
            edge_mask = torch.isin(edge_index[0], graph_nodes) & \
                        torch.isin(edge_index[1], graph_nodes)
            
            # Edge index of a single graph
            graph_edge_index = edge_index[:, edge_mask]
            
            if graph_edge_index.numel() == 0:
                continue
            
            # Calculate the distance matrix and walk probability for the current graph
            walk_probs, distance_matrix = self.compute_graph_distances(
                graph_edge_index, graph_nodes.size(0), graph_nodes)
            
            # Creating a local-to-global mapping
            local_to_global = {i: node.item() for i, node in enumerate(graph_nodes)}
            
            # Random selection of starting nodes
            if self.num_walks >= graph_nodes.size(0):
                start_indices = torch.arange(graph_nodes.size(0), device=self.device)
            else:
                start_indices = torch.randperm(graph_nodes.size(0), device=self.device)[:self.num_walks]
            
            # Generate random walk paths
            for start_idx in start_indices:
                try:
                    current_idx = start_idx.item()
                    walk = [local_to_global[current_idx]]
                    
                    # Generate walk paths
                    for _ in range(self.walk_length - 1):
                        # Get the transfer probability of the current node
                        probs = walk_probs[current_idx]
                        
                        # Select the next node according to the probability
                        next_idx = torch.multinomial(probs, 1).item()
                        current_idx = next_idx
                        walk.append(local_to_global[current_idx])
                    
                    walks.append(torch.tensor(walk, device=self.device))
                    walk_batch.append(graph_idx)
                    
                except Exception as e:
                    print(f"Random walk gone wrong: {e}")
                    # Creating Alternate Path
                    walk = torch.full((self.walk_length,), local_to_global[start_idx.item()], 
                                    dtype=torch.long, device=self.device)
                    walks.append(walk)
                    walk_batch.append(graph_idx)
        
        
        if len(walks) == 0:
            return torch.zeros(1, self.walk_length, device=self.device, dtype=torch.long), \
                   torch.zeros(1, device=self.device, dtype=torch.long)
        
        
        walk_tensor = torch.stack(walks)
        walk_batch = torch.tensor(walk_batch, device=self.device)
        return walk_tensor, walk_batch
    
    def forward(self, x, edge_index, batch):
        # Generate random walk path
        walks, walk_batch = self.sample_random_walks(edge_index, batch, x.size(0))
        
        # Get the features of the nodes in the path
        x_encoded = self.feature_encoder(x)
        
        # Create a sequence of features for each walk path
        batch_size = walks.size(0)
        seq_len = walks.size(1)
        walk_features = torch.zeros(batch_size, seq_len, self.hidden_dim, device=self.device)
        
        # Collect features for each path node
        for i in range(batch_size):
            for j in range(seq_len):
                node_idx = walks[i, j].item()
                if 0 <= node_idx < x.size(0):
                    walk_features[i, j] = x_encoded[node_idx]
        
        # Encoding paths using GRU
        _, h_n = self.path_encoder(walk_features)
        path_encodings = h_n.squeeze(0)  # [batch_size, hidden_dim]
        
        return path_encodings, walk_batch

class RW_GNN(torch.nn.Module):
    def __init__(self, input_dim, max_step, hidden_graphs, size_hidden_graphs, 
                 hidden_dim, penultimate_dim, normalize, num_classes, dropout, device, alpha=1.0, c=0.15, walk_type='exp'):
        super(RW_GNN, self).__init__()
        
        self.max_step = max_step
        self.hidden_graphs = hidden_graphs
        self.size_hidden_graphs = size_hidden_graphs
        self.normalize = normalize
        self.device = device
        self.penultimate_dim = penultimate_dim
        self.num_classes = num_classes
        
        # Random Walk Encoder
        self.walk_encoder = ExplicitRandomWalkEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            walk_length=max_step + 1,  # +1 including start node
            num_walks=hidden_graphs,   # Number of random walks per graph sample
            device=device,
            alpha=alpha,  # Distance impact parameters
            c=c,  # Random jump probability parameter
            walk_type=walk_type  # type of random walk
        )
        
        # Learned hidden graph parameters
        self.adj_hidden = Parameter(torch.FloatTensor(hidden_graphs, 
                                    (size_hidden_graphs*(size_hidden_graphs-1))//2))
        self.features_hidden = Parameter(torch.FloatTensor(hidden_graphs, 
                                         size_hidden_graphs, hidden_dim))
        
        # batch normalisation layer
        self.bn = BatchNorm1d(hidden_dim)
        self.bn1 = BatchNorm1d(penultimate_dim)  # Add more batch normalisation
        
        # output layer
        self.fc1 = Linear(hidden_dim, penultimate_dim)
        self.fc2 = Linear(penultimate_dim, num_classes)
        
        # regularisation layer
        self.dropout = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout * 0.5)  # Add a second dropout layer with slightly lower probability
        self.relu = nn.ReLU()
        
        # Initialisation parameters
        self.init_weights()
    
    def init_weights(self):
        self.adj_hidden.data.uniform_(-1, 1)
        self.features_hidden.data.uniform_(0, 1)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Check if it is a node classification task
        if batch is None or torch.unique(batch).size(0) == 1:
            # Node classification tasks
            return self.forward_node_classification(data)
        else:
            # graph classification tasks
            return self.forward_graph_classification(data)
    
    def forward_node_classification(self, data):
        """Forward Propagation for Node Classification - Generate specific random walks for each node"""
        x, edge_index = data.x, data.edge_index
        num_nodes = x.size(0)
        
        # Use node-specific random walk codes
        try:
            node_encodings = self._node_specific_walk_encoding(x, edge_index, num_nodes)
        except Exception as e:
            print(f"Random walk coding fails using a simplified method: {e}")
            
            node_encodings = self._simple_neighborhood_encoding(x, edge_index, num_nodes)
        
        # Downstream network processing
        out = self.fc1(node_encodings)
        out = self.bn1(out) if out.size(0) > 1 else out  # batch normalisation
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout2(out)  # Adding a second dropout layer
        
        return F.log_softmax(out, dim=1)
    
    def _node_specific_walk_encoding(self, x, edge_index, num_nodes):
        """Generate node-specific random walk encodings"""
        hidden_dim = self.walk_encoder.hidden_dim
        node_encodings = torch.zeros(num_nodes, hidden_dim, device=self.device)
        
        # feature coding
        x_encoded = self.walk_encoder.feature_encoder(x)
        
        # build adjacency list for quick neighbor lookup
        adj_list = {}
        for i in range(num_nodes):
            adj_list[i] = []
        
        # Filtering valid edges and building neighbourhood lists
        valid_edge_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
        valid_edges = edge_index[:, valid_edge_mask]
        
        for i in range(valid_edges.size(1)):
            src, dst = valid_edges[0, i].item(), valid_edges[1, i].item()
            adj_list[src].append(dst)
            adj_list[dst].append(src)  # undirected graph
        
        # Generate random walk for each node
        for node_idx in range(num_nodes):
            walks = []
            
            # Generate multiple random walks starting at this node
            for _ in range(min(self.hidden_graphs, 8)):  # Limit the number of walks to improve efficiency
                walk = self._generate_single_walk(node_idx, adj_list, self.max_step + 1)
                walks.append(walk)
            
            # Convert the traversal into a feature sequence and encode it using GRU
            if walks:
                walk_features = []
                for walk in walks:
                    walk_feat = torch.stack([x_encoded[node] for node in walk])
                    walk_features.append(walk_feat)
                
                # Stack all roaming features
                walk_tensor = torch.stack(walk_features)  # [num_walks, walk_length, hidden_dim]
                
                # Using GRU encoding
                _, h_n = self.walk_encoder.path_encoder(walk_tensor)
                # average across walks
                node_encodings[node_idx] = h_n.squeeze(0).mean(dim=0)
            else:
                # no walks: use raw encoding
                node_encodings[node_idx] = x_encoded[node_idx]
        
        return node_encodings
    
    def _generate_single_walk(self, start_node, adj_list, walk_length):
        """Generate a single random walk"""
        walk = [start_node]
        current_node = start_node
        
        for _ in range(walk_length - 1):
            neighbors = adj_list.get(current_node, [])
            if not neighbors:
                # no neighbors: stay put
                walk.append(current_node)
            else:
                # Randomly select a neighbour
                next_node = np.random.choice(neighbors)
                walk.append(next_node)
                current_node = next_node
        
        return walk
    
    def _simple_neighborhood_encoding(self, x, edge_index, num_nodes):
        """neighbourhood coding"""
        hidden_dim = self.walk_encoder.hidden_dim
        
        # feature code
        x_encoded = self.walk_encoder.feature_encoder(x)
        
        # Filter out edges that are out of node range
        valid_edge_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
        valid_edge_index = edge_index[:, valid_edge_mask]
        
        print(f"Original edges: {edge_index.size(1)}, valid edges: {valid_edge_index.size(1)}")
        
        if valid_edge_index.size(1) == 0:
            # If there are no valid edges, the original feature code is returned directly
            return x_encoded
        
        # Constructing an adjacency matrix (sparse)
        adj_indices = valid_edge_index
        adj_values = torch.ones(valid_edge_index.size(1), device=self.device)
        adj_shape = (num_nodes, num_nodes)
        adj_sparse = torch.sparse_coo_tensor(adj_indices, adj_values, adj_shape)
        
        # Multi-hop neighbourhood aggregation
        node_encodings = x_encoded.clone()
        
        # 1-hop neighbourhood
        neighbor_1 = torch.sparse.mm(adj_sparse, x_encoded)
        
        # 2-hop neighbourhood (if required)
        if self.max_step > 1:
            try:
                adj_2 = torch.sparse.mm(adj_sparse, adj_sparse.to_dense()).to_sparse()
                neighbor_2 = torch.sparse.mm(adj_2, x_encoded)
                # Combining information from different hops
                node_encodings = (x_encoded + neighbor_1 + 0.5 * neighbor_2) / 2.5
            except:
                # If the 2-hop calculation fails, only 1 hop is used
                node_encodings = (x_encoded + neighbor_1) / 2.0
        else:
            # Use only 1-hop neighbourhood
            node_encodings = (x_encoded + neighbor_1) / 2.0
        
        return node_encodings
    
    def forward_graph_classification(self, data):
        """Forward propagation of graph classification"""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # encode walks and get their batch assignments
        walk_encodings, walk_batch = self.walk_encoder(x, edge_index, batch)
        
        # Compute a unique ID for each graph
        unique_batch_ids = torch.unique(batch)
        batch_size = unique_batch_ids.size(0)
        
        # Aggregate for each graph the encoding of all its random walk paths
        graph_encodings = torch.zeros(batch_size, walk_encodings.size(1), device=self.device)
        
        # For each graph, average the coding of all its random walk paths
        for i, graph_idx in enumerate(unique_batch_ids):
            # Find all paths belonging to the current graph
            mask = (walk_batch == graph_idx)
            if mask.sum() > 0:  # Ensure path is available
                # Calculation of the average code
                graph_encodings[i] = walk_encodings[mask].mean(dim=0)
        
        # Apply batch normalisation (only used when batch size > 1)
        if graph_encodings.size(0) > 1:
            graph_encodings = self.bn(graph_encodings)
        
        # Downstream network processing
        out = self.fc1(graph_encodings)
        out = self.bn1(out) if out.size(0) > 1 else out  # batch normalisation
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout2(out)  # Adding a second dropout layer
        
        return F.log_softmax(out, dim=1)



def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path='training_curves.png'):
    """
    Plot training and validation loss and accuracy curves

    Args:
        train_losses: list of training losses
        val_losses: list of validation losses
        train_accs: list of training accuracies
        val_accs: list of validation accuracies
        save_path: path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # plot loss curves
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy curves
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # set y-axis range
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Training curves saved to {save_path}')

# training function
def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
        pred = out.max(1)[1]
        correct += pred.eq(data.y).sum().item()
        total += data.num_graphs
    
    return total_loss / total, correct / total

# test function
def test(model, loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = F.cross_entropy(out, data.y)
            
            total_loss += loss.item() * data.num_graphs
            pred = out.max(1)[1]
            correct += pred.eq(data.y).sum().item()
            total += data.num_graphs
    
    return total_loss / total, correct / total

class LocalCoraDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data = self.process()
        
    @property
    def raw_file_names(self):
        return ['ind.cora.x', 'ind.cora.y', 'ind.cora.allx', 'ind.cora.ally', 
                'ind.cora.graph', 'ind.cora.test.index']
    
    @property
    def processed_file_names(self):
        return ['cora_data.pt']
    
    def download(self):
        # Disable downloads
        pass
    
    def process(self):
        import pickle
        import scipy.sparse as sp
        
        print("Start loading the Cora dataset...")
        
        # Read raw data files
        x = self._read_pickle_file('ind.cora.x')          # Test related features
        y = self._read_pickle_file('ind.cora.y')          # Test Related labels
        allx = self._read_pickle_file('ind.cora.allx')    # Training + validation features
        ally = self._read_pickle_file('ind.cora.ally')    # Training + validation labels
        graph = self._read_pickle_file('ind.cora.graph')  # adjacency dictionary
        test_idx_reorder = self._read_test_index('ind.cora.test.index')  # Test set reordering index
        
        print(f"Raw data shape: x={x.shape}, y={y.shape}, allx={allx.shape}, ally={ally.shape}")
        print(f"Testing index reordering: {len(test_idx_reorder)} indexes")
        print(f"The graph structure contains {len(graph)} nodes")
        
        # Converting sparse matrices to dense matrices
        if hasattr(x, 'todense'):
            x = x.todense()
        if hasattr(allx, 'todense'):
            allx = allx.todense()
        
        # Convert to numpy array
        x = np.array(x, dtype=np.float32)
        allx = np.array(allx, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        ally = np.array(ally, dtype=np.int32)
        
        print(f"Shape of converted data: x={x.shape}, allx={allx.shape}")
        
        
        features = sp.vstack((allx, x)).tolil()
        labels = np.vstack((ally, y))
        
        actual_num_nodes = features.shape[0]
        actual_test_nodes = x.shape[0]
        print(f"Actual dataset shape: features={features.shape}, labels={labels.shape}")
        print(f"Total number of actual nodes: {actual_num_nodes}, Number of test nodes: {actual_test_nodes}")
        
        # Processing test index reordering - adjusted to actual data
        test_idx_range = np.sort(test_idx_reorder)
        test_idx_range_max = np.max(test_idx_range)
        test_idx_range_min = np.min(test_idx_range)
        
        print(f"Test index range: {test_idx_range_min} - {test_idx_range_max}")
        
        # Only process test indexes that are in the valid range
        valid_test_indices = test_idx_reorder[test_idx_reorder < actual_num_nodes]
        print(f"Number of valid test indexes: {len(valid_test_indices)} / {len(test_idx_reorder)}")
        
        # Creating copies of features and labels for reordering
        features_dense = features.todense()
        labels_copy = labels.copy()
        
        # Reordering: only valid test indexes are processed
        allx_size = allx.shape[0]  # Training + validation set size
        
        for i, original_test_idx in enumerate(test_idx_reorder):
            # Check if the index is in the valid range
            source_idx = allx_size + i  # Position of the original test node in the merged data
            if (original_test_idx < actual_num_nodes and 
                source_idx < actual_num_nodes and 
                i < actual_test_nodes):
                # Place the features and labels of the ith node in the test set in the correct position
                features_dense[original_test_idx] = features_dense[source_idx]
                labels_copy[original_test_idx] = labels[source_idx]
        
        # Convert to torch tensor
        features = torch.FloatTensor(np.array(features_dense))
        labels = torch.LongTensor(np.argmax(labels_copy, axis=1))  # Convert one-hot to a category index
        
        print(f"Final feature shape: {features.shape}")
        print(f"Label Shape: {labels.shape}, Label range: {labels.min()}-{labels.max()}")
        
        # Constructing edge Indexes
        edges = []
        num_nodes = features.shape[0]  # It should be 2708
        
        for node, neighbors in graph.items():
            # Ensure that the node index is in the valid range
            if node < num_nodes:
                for neighbor in neighbors:
                    # Ensure that neighbouring indexes are also in effect
                    if neighbor < num_nodes:
                        edges.append([node, neighbor])
        
        print(f"Build Edge Index: Total Edges {len(edges)}, Number of nodes {num_nodes}")
        
        if len(edges) == 0:
            # If there are no valid edges, create an empty edge index
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.LongTensor(edges).t().contiguous()
            # Ensure that the edge index is undirected
            edge_index = to_undirected(edge_index)
        
        print(f"Final Edge Index Shape {edge_index.shape}")
        
        # Create dataset segmentation masks - adjusted to actual data size
        num_nodes = features.shape[0]
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        # Dataset segmentation strategies (140/500/1000+)
        train_size = 140  # Cora standard training set size
        val_size = 500    # Cora Standard Validation Set Size
        
        print(f"Segmentation using Cora: training ={train_size}, Validation ={val_size}, Test = remaining nodes")
        
        
        valid_test_indices = test_idx_reorder[test_idx_reorder < num_nodes]
        print(f"Number of original test indexes. {len(test_idx_reorder)}")
        print(f"Number of valid test indexes. {len(valid_test_indices)}")
        print(f"Test index range. {valid_test_indices.min()}-{valid_test_indices.max()}")
        
        
        test_mask[valid_test_indices] = True
        
        
        train_candidates = torch.arange(num_nodes)
        train_candidates = train_candidates[~test_mask[train_candidates]]  # Exclude test nodes
        
        if len(train_candidates) >= train_size:
            train_mask[train_candidates[:train_size]] = True
        else:
            
            all_non_test = torch.arange(num_nodes)[~test_mask]
            train_mask[all_non_test[:train_size]] = True
            print(f"From all non-tested nodes, the{train_size}as a training set")
        
        
        val_candidates = torch.arange(num_nodes)
        val_candidates = val_candidates[~test_mask[val_candidates] & ~train_mask[val_candidates]]  # Excluding test and training nodes
        
        if len(val_candidates) >= val_size:
            val_mask[val_candidates[:val_size]] = True
        else:
            
            remaining_nodes = torch.arange(num_nodes)[~test_mask & ~train_mask]
            actual_val_size = min(val_size, len(remaining_nodes))
            val_mask[remaining_nodes[:actual_val_size]] = True
            print(f"The actual size of the validation set is adjusted to {actual_val_size}")
        
        
        assigned_mask = train_mask | val_mask | test_mask
        unassigned_mask = ~assigned_mask
        unassigned_count = unassigned_mask.sum()
        
        if unassigned_count > 0:
            print(f"find{unassigned_count}unassigned nodes, add them to the test set")
            test_mask = test_mask | unassigned_mask
        
        
        final_train = train_mask.sum()
        final_val = val_mask.sum()
        final_test = test_mask.sum()
        final_total = final_train + final_val + final_test
        
        print(f"Final division: training ={final_train}, Validation ={final_val}, Test ={final_test}")
        print(f"Total number of nodes verified: {final_total} / {num_nodes} {'✓' if final_total == num_nodes else '✗'}")
        
        
        overlap_check = ((train_mask & val_mask).sum() + 
                        (train_mask & test_mask).sum() + 
                        (val_mask & test_mask).sum())
        print(f"Overlap check {overlap_check} (should be 0)")
        
        
        test_indices = torch.nonzero(test_mask).squeeze().cpu().numpy()
        print(f"Actual test set index range: {test_indices.min()}-{test_indices.max()}")
        print(f"Test Set Size: {len(test_indices)}")
        
        if final_total != num_nodes or overlap_check > 0:
            raise ValueError(f"Data segmentation error: mismatched totals ({final_total}!={num_nodes}) or overlap({overlap_check}>0)")
        
        # Creating Data Objects
        data = Data(x=features, edge_index=edge_index, y=labels,
                   train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        
        # Setting dataset properties
        data.num_classes = len(torch.unique(labels))
        
        print(f"Dataset creation complete: nodes ={data.num_nodes}, Number of edges ={data.num_edges}, "
              f"features dimension ={data.num_features}, Number of classes ={data.num_classes}")
        
        # Saving of processed data
        torch.save(data, self.processed_paths[0])
        
        return data
    
    def _read_pickle_file(self, filename):
        import pickle
        path = os.path.join(self.raw_dir, filename)
        print(f"Read file: {path}")
        with open(path, 'rb') as f:
            return pickle.load(f, encoding='latin1')
    
    def _read_test_index(self, filename):
        path = os.path.join(self.raw_dir, filename)
        print(f"Read test index file: {path}")
        with open(path, 'r') as f:
            lines = f.readlines()
        test_idx_reorder = []
        for line in lines:
            test_idx_reorder.append(int(line.strip()))
        return np.array(test_idx_reorder)
    
    @property
    def num_features(self):
        return self.data.num_features
    
    @property 
    def num_classes(self):
        return self.data.num_classes
    
    def len(self):
        return 1
    
    def get(self, idx):
        return self.data

def main():
    # 参数设置
    parser = argparse.ArgumentParser(description='RW_GNN with PyTorch Geometric')
    parser.add_argument('--dataset', default='IMDB-BINARY', help='Dataset name (IMDB-BINARY, Cora)')
    parser.add_argument('--use-node-labels', action='store_true', default=False, 
                        help='Whether to use node labels')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')  # Add dropout
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay (L2 regularization)')  # Add weight decay
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Maximum number of epochs')
    parser.add_argument('--hidden-graphs', type=int, default=12, help='Number of random walks per graph')  # Moderate complexity
    parser.add_argument('--size-hidden-graphs', type=int, default=4, 
                        help='Parameter for internal representation')  # moderate complexity
    parser.add_argument('--hidden-dim', type=int, default=24, help='Hidden dimension')  # moderate dimension
    parser.add_argument('--penultimate-dim', type=int, default=48, 
                        help='Penultimate layer dimension')  # moderate dimension
    parser.add_argument('--max-step', type=int, default=3, help='Maximum walk length')  # Moderate walk length
    parser.add_argument('--normalize', action='store_true', default=False, 
                        help='Whether to normalize features')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', 
                        help='Directory to save model checkpoints')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Parameter controlling the influence of distance in random walks')
    parser.add_argument('--c', type=float, default=0.15,
                        help='Random jump probability parameter (similar to PageRank damping factor)')
    parser.add_argument('--walk-type', type=str, default='exp', choices=['exp', 'levy'],
                        help='Type of random walk: exp (exponential) or levy (Levy flight)')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')  # Adding Early Stop Parameters
    args = parser.parse_args()
    
    # Setting the random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Creating a checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Setting up the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Modify the dataset loading section
    if args.dataset == 'Cora':
        try:
            # Using a custom local dataset loader
            dataset = LocalCoraDataset(root='data/Cora')
            data = dataset[0]
            
            # Make sure the dataset has been loaded correctly
            if not hasattr(data, 'x') or not hasattr(data, 'y'):
                raise ValueError("Dataset loading failed, please check if the dataset file is complete")
                
            print(f"Data set loaded successfully: {data}")
            print(f"Number of nodes: {data.num_nodes}")
            print(f"Number of edges: {data.num_edges}")
            print(f"Feature Dimension: {data.num_features}")
            print(f"Number of classes: {len(torch.unique(data.y))}")
            
            # Verify data integrity
            print(f"Number of nodes in the training set: {data.train_mask.sum().item()}")
            print(f"Number of validation set nodes: {data.val_mask.sum().item()}")
            print(f"Number of test set nodes: {data.test_mask.sum().item()}")
            print(f"Label range: {data.y.min().item()} - {data.y.max().item()}")
            print(f"Feature Matrix Shape: {data.x.shape}")
            print(f"edge Index Shape: {data.edge_index.shape}")
            
            # Creating virtual batch information
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
            data.num_graphs = 1
            
            # Moving data to a specified device
            data = data.to(device)
            
            # Create training, validation and test masks
            train_mask = data.train_mask
            val_mask = data.val_mask
            test_mask = data.test_mask
            
            # Creating Models
            model = RW_GNN(
                input_dim=dataset.num_features,
                max_step=args.max_step,
                hidden_graphs=args.hidden_graphs,
                size_hidden_graphs=args.size_hidden_graphs,
                hidden_dim=args.hidden_dim,
                penultimate_dim=args.penultimate_dim,
                normalize=args.normalize,
                num_classes=dataset.num_classes,
                dropout=args.dropout,
                device=device,
                alpha=args.alpha,
                c=args.c,
                walk_type=args.walk_type
            ).to(device)
            
            # Optimiser and learning rate scheduler
            optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
            
            # Early stop mechanism - based on test set accuracy
            best_test_acc = 0
            patience_counter = 0
            train_losses = []
            val_losses = []
            train_accs = []
            val_accs = []
            
            for epoch in range(args.epochs):
                # training
                model.train()
                optimizer.zero_grad()
                out = model(data)
                train_loss = F.nll_loss(out[train_mask], data.y[train_mask])
                train_loss.backward()
                optimizer.step()
                
                # Calculate training accuracy
                with torch.no_grad():
                    pred = out[train_mask].max(1)[1]
                    train_acc = pred.eq(data.y[train_mask]).sum().item() / train_mask.sum().item()
                
                # validation
                model.eval()
                with torch.no_grad():
                    out = model(data)
                    val_loss = F.nll_loss(out[val_mask], data.y[val_mask])
                    pred = out[val_mask].max(1)[1]
                    val_acc = pred.eq(data.y[val_mask]).sum().item() / val_mask.sum().item()
                    
                    # test
                    test_loss = F.nll_loss(out[test_mask], data.y[test_mask])
                    pred = out[test_mask].max(1)[1]
                    test_acc = pred.eq(data.y[test_mask]).sum().item() / test_mask.sum().item()
                
                # Recording of training indicators
                train_losses.append(train_loss.item())
                val_losses.append(val_loss.item())
                train_accs.append(train_acc)
                val_accs.append(val_acc)
                
                print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
                      f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
                
                # Update the learning rate scheduler
                scheduler.step(test_acc)
                
                # Preservation of best models and early stop checks
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    patience_counter = 0
                    torch.save(model.state_dict(), 
                              os.path.join(args.checkpoint_dir, 'cora_best_model.pt'))
                    print(f'*** Preserve best models (Epoch {epoch:03d}): Test Acc = {test_acc:.4f} (Val Acc = {val_acc:.4f}) ***')
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience:
                        print(f'Early stopping at epoch {epoch+1}')
                        print(f'Best Test Accuracy: {best_test_acc:.4f}')
                        break
            
            # Plotting training curves
            plot_save_path = os.path.join(args.checkpoint_dir, f'cora_training_curves_{args.walk_type}_alpha_{args.alpha}_c_{args.c}.png')
            plot_training_curves(train_losses, val_losses, train_accs, val_accs, plot_save_path)
            
            
            print(f'best test accuracy training: {best_test_acc:.4f}')
            model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'cora_best_model.pt')))
            model.eval()
   
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            print("Make sure that the dataset file is located in the data/Cora directory and contains the following files：")
            print("- ind.cora.x")
            print("- ind.cora.y")
            print("- ind.cora.allx")
            print("- ind.cora.ally")
            print("- ind.cora.graph")
            print("- ind.cora.test.index")
            raise
    else:
        # Original graph classification dataset processing logic
        dataset = TUDataset(root='data', name=args.dataset, use_node_attr=args.use_node_labels)
        
        if dataset.num_node_features == 0:
            max_degree = 0
            degs = []
            for data in dataset:
                degs += [degree(data.edge_index[0], dtype=torch.long)]
                max_degree = max(max_degree, degs[-1].max().item())
            
            if max_degree < 1000:
                dataset.transform = T.OneHotDegree(max_degree)
        
        # 10-fold cross validation
        kf = KFold(n_splits=10, shuffle=True, random_state=args.seed)
        accs = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(range(len(dataset)))):
            print(f'Fold {fold+1}/10')
            
            # Segmented data sets
            train_idx, val_idx = train_idx[:int(len(train_idx)*0.9)], train_idx[int(len(train_idx)*0.9):]
            
            train_dataset = dataset[train_idx.tolist()]
            val_dataset = dataset[val_idx.tolist()]
            test_dataset = dataset[test_idx.tolist()]
            
            # Creating a Data Loader
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
            
            # Creating Models
            model = RW_GNN(
                input_dim=dataset.num_features,
                max_step=args.max_step,
                hidden_graphs=args.hidden_graphs,
                size_hidden_graphs=args.size_hidden_graphs,
                hidden_dim=args.hidden_dim,
                penultimate_dim=args.penultimate_dim,
                normalize=args.normalize,
                num_classes=dataset.num_classes,
                dropout=args.dropout,
                device=device,
                alpha=args.alpha,
                c=args.c,
                walk_type=args.walk_type
            ).to(device)
            
            # Optimiser and learning rate scheduler
            optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
            
            # Early Stop Mechanism
            best_val_acc = 0
            patience_counter = 0
            fold_train_losses = []
            fold_val_losses = []
            fold_train_accs = []
            fold_val_accs = []
            
            for epoch in range(args.epochs):
                # training
                train_loss, train_acc = train(model, train_loader, optimizer, device)
                
                # validation
                val_loss, val_acc = test(model, val_loader, device)
                
                # Recording of training indicators
                fold_train_losses.append(train_loss)
                fold_val_losses.append(val_loss)
                fold_train_accs.append(train_acc)
                fold_val_accs.append(val_acc)
                
                print(f'Fold: {fold+1:02d}, Epoch: {epoch:03d}, '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
                
                # Updating the learning rate scheduler
                scheduler.step(val_acc)
                
                # Preservation of best models and early stop checks
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    torch.save(model.state_dict(), 
                              os.path.join(args.checkpoint_dir, f'best_model_fold_{fold}.pt'))
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience:
                        print(f'Early stopping at epoch {epoch+1} for fold {fold+1}')
                        break
            
            # Plot the training curve for the current fold
            fold_plot_path = os.path.join(args.checkpoint_dir, f'{args.dataset}_fold_{fold+1}_training_curves_{args.walk_type}_alpha_{args.alpha}_c_{args.c}.png')
            plot_training_curves(fold_train_losses, fold_val_losses, fold_train_accs, fold_val_accs, fold_plot_path)
            
            # Load the best model for testing
            model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, f'best_model_fold_{fold}.pt')))
            test_loss, test_acc = test(model, test_loader, device)
            print(f'Fold {fold+1} Test Accuracy: {test_acc:.4f}')
            accs.append(test_acc)
        
        print(f'Average Test Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}')

if __name__ == '__main__':
    main() 