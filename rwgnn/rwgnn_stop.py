import argparse
import time
import numpy as np
from math import ceil
import os
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Parameter, GRU
from torch.optim import Adam

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
from torch_geometric.utils import add_self_loops, degree, to_dense_adj, to_undirected
from torch_geometric import transforms as T
from torch_cluster import random_walk  # Import random_walk from torch_cluster

class ExplicitRandomWalkEncoder(nn.Module):
    """Explicit random walk encoder that generates and encodes random walk paths"""
    def __init__(self, input_dim, hidden_dim, walk_length, num_walks, device):
        super(ExplicitRandomWalkEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.device = device
        
        # feature transformation layer
        self.feature_encoder = Linear(input_dim, hidden_dim)
        
        # Path Coding Layer (GRU)
        self.path_encoder = GRU(hidden_dim, hidden_dim, batch_first=True)
    
    def sample_random_walks(self, edge_index, batch, num_nodes):
        """Generate multiple randomized wandering paths for each graph"""
        walks = []
        walk_batch = []  # Record the graph to which each path belongs
        batch_size = torch.max(batch).item() + 1
        
        # Generate random walks for each graph
        for graph_idx in range(batch_size):
            # Get the node index of the current graph
            graph_mask = (batch == graph_idx)
            graph_nodes = torch.nonzero(graph_mask).squeeze()
            
            # Handling single-node situations
            if graph_nodes.dim() == 0:
                graph_nodes = graph_nodes.unsqueeze(0)
                
            # If the graph has only one node, skip the random wandering
            if graph_nodes.size(0) <= 1:
                continue
                
            # Get the edge index of the current graph
            edge_mask = torch.isin(edge_index[0], graph_nodes) & \
                        torch.isin(edge_index[1], graph_nodes)
            
            # Edge index of a single graph
            graph_edge_index = edge_index[:, edge_mask]
            
            if graph_edge_index.numel() == 0:
                continue
                
            # Ensure that directed graphs are converted to undirected graphs
            graph_edge_index = to_undirected(graph_edge_index)
            
            # Preparing inputs for random wandering
            row, col = graph_edge_index
            
            # Random selection of starting nodes
            if self.num_walks >= graph_nodes.size(0):
                # If the number of required wanderings is greater than or equal to the number of nodes,
                # use all nodes as the starting point
                start_nodes = graph_nodes
            else:
                # Otherwise randomly selected nodes
                indices = torch.randperm(graph_nodes.size(0))[:self.num_walks]
                start_nodes = graph_nodes[indices]
            
            # Generate randomized wandering paths
            for start_node in start_nodes:
                try:
                    # Convert the start node to a tensor
                    # and make sure it is a one-dimensional tensor of length 1
                    start = start_node.view(-1)
                    
                    # Generate paths using torch_cluster.random_walk
                    # The return value is the tensor of [batch_size, walk_length].
                    walk = random_walk(row, col, start, walk_length=self.walk_length-1)[0]
                    
                    # Make sure the path length is correct
                    if walk.size(0) < self.walk_length:
                        # If the path is too short, fill it with the last node
                        last_node = walk[-1] if walk.size(0) > 0 else start_node
                        padding = torch.full((self.walk_length - walk.size(0),), 
                                           last_node.item(), 
                                           dtype=torch.long, device=self.device)
                        walk = torch.cat([walk, padding])
                    elif walk.size(0) > self.walk_length:
                        # If the path is too long, truncate
                        walk = walk[:self.walk_length]
                    
                    walks.append(walk)
                    walk_batch.append(graph_idx)  # Record which graph the path belongs to
                except Exception as e:
                    print(f"Random walk gone wrong: {e}")
                    # Creates an alternate path - contains only the start node,
                    # the rest is populated with the same values
                    walk = torch.full((self.walk_length,), start_node.item(), 
                                     dtype=torch.long, device=self.device)
                    walks.append(walk)
                    walk_batch.append(graph_idx)  # Record which graph the path belongs to
        
        # If there are no valid random walks, create an empty placeholder
        if len(walks) == 0:
            return torch.zeros(1, self.walk_length, device=self.device, dtype=torch.long), torch.zeros(1, device=self.device, dtype=torch.long)
            
        # Stack all wandering paths into a single batch
        walk_tensor = torch.stack(walks)
        walk_batch = torch.tensor(walk_batch, device=self.device)
        return walk_tensor, walk_batch
    
    def forward(self, x, edge_index, batch):
        # Generate randomized wandering paths
        walks, walk_batch = self.sample_random_walks(edge_index, batch, x.size(0))
        
        # Get the features of the nodes in the path
        x_encoded = self.feature_encoder(x)
        
        # Create a sequence of features for each wandering path
        batch_size = walks.size(0)
        seq_len = walks.size(1)
        walk_features = torch.zeros(batch_size, seq_len, self.hidden_dim, device=self.device)
        
        # Collect features for each path node
        for i in range(batch_size):
            for j in range(seq_len):
                node_idx = walks[i, j].item()  # Get integer index
                if 0 <= node_idx < x.size(0):  # Make sure the index is valid
                    walk_features[i, j] = x_encoded[node_idx]
        
        # Using GRU encoded paths
        _, h_n = self.path_encoder(walk_features)
        path_encodings = h_n.squeeze(0)  # [batch_size, hidden_dim]
        
        # Returns path code and batch information for subsequent aggregation
        return path_encodings, walk_batch

class RW_GNN(torch.nn.Module):
    def __init__(self, input_dim, max_step, hidden_graphs, size_hidden_graphs, 
                 hidden_dim, penultimate_dim, normalize, num_classes, dropout, device):
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
            device=device
        )
        
        # Learned hidden graph parameters - for interacting with randomized wandering paths
        self.adj_hidden = Parameter(torch.FloatTensor(hidden_graphs, 
                                    (size_hidden_graphs*(size_hidden_graphs-1))//2))
        self.features_hidden = Parameter(torch.FloatTensor(hidden_graphs, 
                                         size_hidden_graphs, hidden_dim))
        
        # batch normalization layer
        self.bn = BatchNorm1d(hidden_dim)
        
        # output layer
        self.fc1 = Linear(hidden_dim, penultimate_dim)
        self.fc2 = Linear(penultimate_dim, num_classes)
        
        # regularization layer
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        
        # Initialization parameters
        self.init_weights()
    
    def init_weights(self):
        self.adj_hidden.data.uniform_(-1, 1)
        self.features_hidden.data.uniform_(0, 1)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Getting path encoding
        # and batch information using a randomized wandering encoder
        walk_encodings, walk_batch = self.walk_encoder(x, edge_index, batch)
        
        # Compute a unique ID for each graph
        unique_batch_ids = torch.unique(batch)
        batch_size = unique_batch_ids.size(0)
        
        # Aggregate for each graph the encoding of all its random wandering paths
        graph_encodings = torch.zeros(batch_size, walk_encodings.size(1), device=self.device)
        
        # For each graph, average the coding of all its random wandering paths
        for i, graph_idx in enumerate(unique_batch_ids):
            # Find all paths belonging to the current diagram
            mask = (walk_batch == graph_idx)
            if mask.sum() > 0:  # Ensure pathways are available
                # Calculation of the average code
                graph_encodings[i] = walk_encodings[mask].mean(dim=0)
        
        # Apply batch normalization
        graph_encodings = self.bn(graph_encodings)
        
        # Downstream network processing
        out = self.relu(self.fc1(graph_encodings))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return F.log_softmax(out, dim=1)

# Early Stop Class Implementation
class EarlyStopping:
    """Early stop mechanism to prevent overfitting"""
    def __init__(self, patience=10, verbose=True, delta=0, path='checkpoint.pt'):
        """
        Parameters.
            patience (int): the number of rounds to wait after the validation set is no longer performing well
            verbose (bool): Whether to print an early stop message.
            delta (float): the minimum change threshold for determining performance improvement
            path (str): the path to save the best model
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
    
    def __call__(self, val_loss, model):
        """
        Determine if an early stop is required

        Parameters.
            val_loss (float): validation set loss
            model (torch.nn.Module): model
        """
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'Early Stop Counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        """Save Model"""
        if self.verbose:
            print(f'Verification Loss Reduction ({self.val_loss_min:.6f} --> {val_loss:.6f})，Save Model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

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

def main():
    # parameterization
    parser = argparse.ArgumentParser(description='RW_GNN with PyTorch Geometric')
    parser.add_argument('--dataset', default='IMDB-BINARY', help='Dataset name')
    parser.add_argument('--use-node-labels', action='store_true', default=False, 
                        help='Whether to use node labels')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Maximum number of epochs')
    parser.add_argument('--hidden-graphs', type=int, default=16, help='Number of random walks per graph')
    parser.add_argument('--size-hidden-graphs', type=int, default=5, 
                        help='Parameter for internal representation')
    parser.add_argument('--hidden-dim', type=int, default=32, help='Hidden dimension')
    parser.add_argument('--penultimate-dim', type=int, default=64, 
                        help='Penultimate layer dimension')
    parser.add_argument('--max-step', type=int, default=4, help='Maximum walk length')
    parser.add_argument('--normalize', action='store_true', default=False, 
                        help='Whether to normalize features')
    parser.add_argument('--patience', type=int, default=20, 
                        help='Patience for early stopping')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', 
                        help='Directory to save model checkpoints')
    args = parser.parse_args()
    
    # Setting the random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Creating a Checkpoint Catalog
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Setting up the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Dataset
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
            device=device
        ).to(device)
        
        # Optimizer and learning rate scheduler
        optimizer = Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5
        )
        
        # Initialize Early Stop
        checkpoint_path = os.path.join(args.checkpoint_dir, f'model_fold_{fold+1}.pt')
        early_stopping = EarlyStopping(
            patience=args.patience, 
            verbose=True, 
            path=checkpoint_path
        )
        
        # training model
        print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Train Acc':^10} | {'Val Loss':^10} | {'Val Acc':^9} | {'Time':^7}")
        print("-" * 65)
        
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            train_loss, train_acc = train(model, train_loader, optimizer, device)
            val_loss, val_acc = test(model, val_loader, device)
            
            # Updated learning rate
            scheduler.step(val_loss)
            
            # Print training information
            print(f"{epoch:^7} | {train_loss:^12.4f} | {train_acc:^10.4f} | {val_loss:^10.4f} | {val_acc:^9.4f} | {time.time() - start:^7.2f}")
            
            # Early stop check
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print(f"Stop early! Stop training in the {epoch} round")
                break
        
        # Load the best model
        model.load_state_dict(torch.load(checkpoint_path))
        
        # Testing the best models
        test_loss, test_acc = test(model, test_loader, device)
        print(f"Test Results - Loss {test_loss:.4f}, Accuracy {test_acc:.4f}")
        accs.append(test_acc)
    
    # Output the final result
    print(f'Average test accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}')
    print(f'Accuracy of all folds: {accs}')

if __name__ == '__main__':
    main() 