import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree
import numpy as np
from scipy.spatial.distance import cdist
from collections import deque
import math


class Walker():
    def __init__(self, edge_index, num_nodes=None, enable_levy=False, alpha=1.5):
        self.edge_index = edge_index
        if num_nodes is None:
            num_nodes = edge_index.max().item() + 1
        else:
            assert num_nodes > edge_index.max().item()
            self.num_nodes = num_nodes

        self.src_degrees = degree(edge_index[0], num_nodes=num_nodes)
        self.dst_degrees = degree(edge_index[1], num_nodes=num_nodes)
        self.has_neighbour_mask = self.src_degrees > 0

        # sort is necessary for quick neighbourhood lookup
        perm = torch.argsort(edge_index[0])
        self.sorted_destinations = edge_index[1, perm]

        # offsets[i] is the starting index in sorted_destinations for neighbours of node i
        self.offsets = torch.zeros(num_nodes+1, dtype=torch.long, device=edge_index.device)
        self.offsets[1:] = torch.cumsum(self.src_degrees, dim=0) 

        # Lévy random walk支持
        self.enable_levy = enable_levy
        self.alpha = alpha
        
        if enable_levy:
            self._precompute_levy_data()

    def _precompute_levy_data(self):
        """Data structures required for precomputing Lévy's walks"""
        print(f"Precomputing data for Lévy random walk with α={self.alpha}...")
        
        # Constructing an adjacency list representation
        self.adj_list = [[] for _ in range(self.num_nodes)]
        for src, dst in self.edge_index.t():
            self.adj_list[src.item()].append(dst.item())
            
        # Pre-calculate multi-hop neighbours for each node
        max_precompute_hops = min(10, self.num_nodes // 100 + 2)  
        self.multi_hop_neighbors = {}
        
        for node in range(self.num_nodes):
            self.multi_hop_neighbors[node] = self._compute_multi_hop_neighbors(node, max_precompute_hops)
            
        print(f"Lévy walk precomputation completed. Max precomputed hops: {max_precompute_hops}")

    def _compute_multi_hop_neighbors(self, start_node, max_hops):
        """Calculate the multi-hop neighbours of a node"""
        neighbors_by_hop = {0: [start_node]}
        visited = {start_node}
        
        for hop in range(1, max_hops + 1):
            current_hop_neighbors = []
            prev_hop_nodes = neighbors_by_hop.get(hop - 1, [])
            
            for node in prev_hop_nodes:
                for neighbor in self.adj_list[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        current_hop_neighbors.append(neighbor)
                        
            if not current_hop_neighbors:
                break
                
            neighbors_by_hop[hop] = current_hop_neighbors
            
        return neighbors_by_hop

    def _sample_levy_jump_length(self, size, alpha, device):
        """Sampling the jump length of the Lévy distribution"""
        # Sampling the Lévy distribution using the inverse cumulative distribution function approach
        # For a stable distribution, when β = 1, μ = 0: P(r) ∝ r^(-α-1)
        
        # Generate uniform random numbers
        u = torch.rand(size, device=device)
        
        
        u = torch.clamp(u, min=1e-8, max=1-1e-8)
        
        if alpha == 1.0:
            
            jump_lengths = torch.tan(math.pi * (u - 0.5))
            jump_lengths = torch.abs(jump_lengths)
        else:
            
            # P(r) ∝ r^(-α)，CDF^(-1)(u) ≈ (1-u)^(-1/α)
            power = -1.0 / alpha
            jump_lengths = torch.pow(1 - u, power)
            
        
        max_jump = min(20, self.num_nodes // 10 + 1)
        jump_lengths = torch.clamp(jump_lengths, min=1.0, max=float(max_jump))
        return jump_lengths.long()

    def _levy_jump(self, current_nodes, jump_lengths):
        """Implementation of Lévy jumps"""
        batch_size = current_nodes.size(0)
        target_nodes = torch.zeros_like(current_nodes)
        
        for i in range(batch_size):
            current_node = current_nodes[i].item()
            jump_length = jump_lengths[i].item()
            
            # Get multi-hop neighbours of the current node
            neighbors_by_hop = self.multi_hop_neighbors.get(current_node, {})
            
            # Attempts to find neighbours with the specified number of hops
            target_candidates = []
            for hop in range(jump_length, 0, -1):  # Starting from the target number of jumps and gradually decreasing
                if hop in neighbors_by_hop and neighbors_by_hop[hop]:
                    target_candidates = neighbors_by_hop[hop]
                    break
                    
            if target_candidates:
                
                idx = torch.randint(0, len(target_candidates), (1,)).item()
                target_nodes[i] = target_candidates[idx]
            else:
                
                target_nodes[i] = self._local_random_step(current_node)
                
        return target_nodes

    def _local_random_step(self, node):
        """Single-step localised random walks"""
        if self.has_neighbour_mask[node]:
            start_idx = self.offsets[node]
            end_idx = self.offsets[node + 1]
            neighbors = self.sorted_destinations[start_idx:end_idx]
            
            if len(neighbors) > 0:
                idx = torch.randint(0, len(neighbors), (1,)).item()
                return neighbors[idx].item()
                
        return node  # 

    def levy_random_walk(self, start_nodes, walk_length, num_walks_per_node=1):
        """Lévy random walk"""
        if not self.enable_levy:
            raise ValueError("Lévy random walk not enabled")
            
        start_nodes = torch.unique(start_nodes)
        start_nodes = start_nodes.repeat(num_walks_per_node)
        num_walks = start_nodes.size(0)

        assert start_nodes.ndim == 1
        
        # shape [num_walks, walk_length+1]
        paths = torch.full((num_walks, walk_length+1), -1, dtype=torch.long, device=self.edge_index.device)
        
        current_positions = start_nodes.clone()
        paths[:, 0] = current_positions

        if walk_length == 0:
            return paths

        for step in range(1, walk_length + 1):
            # Sampling the jump length of the Lévy distribution
            jump_lengths = self._sample_levy_jump_length(
                size=num_walks, 
                alpha=self.alpha, 
                device=self.edge_index.device
            )
            
            # Implementation of Lévy jumps
            current_positions = self._levy_jump(current_positions, jump_lengths)
            paths[:, step] = current_positions
        
        paths = paths.view(num_walks_per_node, -1, walk_length+1)
        
        return paths

    def adaptive_levy_random_walk(self, start_nodes, walk_length, num_walks_per_node=1, alpha_schedule=None):
        """Adaptive alpha lévy random walk"""
        if not self.enable_levy:
            raise ValueError("Lévy random walk not enabled")
            
        if alpha_schedule is None:
            
            alpha_schedule = torch.linspace(self.alpha + 0.5, self.alpha - 0.5, walk_length)
            alpha_schedule = torch.clamp(alpha_schedule, min=1.1, max=2.9)
            
        start_nodes = torch.unique(start_nodes)
        start_nodes = start_nodes.repeat(num_walks_per_node)
        num_walks = start_nodes.size(0)

        assert start_nodes.ndim == 1
        
        paths = torch.full((num_walks, walk_length+1), -1, dtype=torch.long, device=self.edge_index.device)
        current_positions = start_nodes.clone()
        paths[:, 0] = current_positions

        if walk_length == 0:
            return paths

        for step in range(1, walk_length + 1):
            
            current_alpha = alpha_schedule[step - 1].item()
            
            
            jump_lengths = self._sample_levy_jump_length(
                size=num_walks, 
                alpha=current_alpha, 
                device=self.edge_index.device
            )
            
            
            current_positions = self._levy_jump(current_positions, jump_lengths)
            paths[:, step] = current_positions
        
        paths = paths.view(num_walks_per_node, -1, walk_length+1)
        return paths

    def mixed_random_walk(self, start_nodes, walk_length, num_walks_per_node=1, levy_ratio=0.5):
        """Mixed random walk"""
        start_nodes = torch.unique(start_nodes)
        start_nodes = start_nodes.repeat(num_walks_per_node)
        num_walks = start_nodes.size(0)
        
        
        use_levy = torch.rand(num_walks) < levy_ratio
        
        paths = torch.full((num_walks, walk_length+1), -1, dtype=torch.long, device=self.edge_index.device)
        paths[:, 0] = start_nodes
        
        if walk_length == 0:
            return paths.view(num_walks_per_node, -1, walk_length+1)
        
        
        if use_levy.any():
            levy_indices = use_levy.nonzero(as_tuple=True)[0]
            levy_starts = start_nodes[levy_indices]
            if len(levy_starts) > 0:
                levy_paths = self.levy_random_walk(levy_starts, walk_length, num_walks_per_node=1)
                paths[levy_indices] = levy_paths.squeeze(0)
        
        if (~use_levy).any():
            uniform_indices = (~use_levy).nonzero(as_tuple=True)[0]
            uniform_starts = start_nodes[uniform_indices]
            if len(uniform_starts) > 0:
                uniform_paths = self.uniform_random_walk(uniform_starts, walk_length, num_walks_per_node=1)
                paths[uniform_indices] = uniform_paths.squeeze(0)
        
        return paths.view(num_walks_per_node, -1, walk_length+1)
        
    def uniform_random_walk(self, start_nodes, walk_length, num_walks_per_node=1):
        start_nodes = torch.unique(start_nodes)
        start_nodes = start_nodes.repeat(num_walks_per_node)
        num_walks = start_nodes.size(0)

        assert start_nodes.ndim == 1
        
        # shape [num_walks, walk_length+1]
        paths = torch.full((num_walks, walk_length+1), -1, dtype=torch.long, device=self.edge_index.device)
        
        current_positions = start_nodes.clone()
        paths[:, 0] = current_positions

        if walk_length == 0:
            return paths

        for step in range(1, walk_length + 1):
            current_src_degrees = self.src_degrees[current_positions]
            current_offsets = self.offsets[current_positions]
            has_neighbour_mask = self.has_neighbour_mask[current_positions]

            # sample random neighbours
            random_neighbour_indices = (torch.rand(num_walks, device=self.edge_index.device) * current_src_degrees).long()

            # this is possible because destination is sorted
            absolute_neighbour_indices = current_offsets + random_neighbour_indices

            # Initialize next_nodes with current_positions (for nodes that are stuck or have no neighbours)
            next_nodes = current_positions.clone()
            if has_neighbour_mask.any():
                selected_absolute_indices = absolute_neighbour_indices[has_neighbour_mask]
                next_nodes[has_neighbour_mask] = self.sorted_destinations[selected_absolute_indices]
            
            current_positions = next_nodes
            paths[:, step] = current_positions
        
        paths = paths.view(num_walks_per_node, -1, walk_length+1)
        
        return paths
    
def anonymise_walks(paths): # only anonymise the last dim
    eq_mat = paths.unsqueeze(-1) == paths.unsqueeze(-2)
    first_idx = eq_mat.type(torch.long).argmax(dim=-1)

    # mask first occurrence positions
    mask_first = first_idx == torch.arange(paths.shape[-1], device=paths.device).unsqueeze(-2)

    fi_j = first_idx.unsqueeze(-1)
    fi_k = first_idx.unsqueeze(-2)
    mk_k = mask_first.unsqueeze(-2)
    
    less_mask = (fi_k < fi_j) & mk_k
    # number of distinct unique (mk_k) elements to the left
    # that is less than the current element 
    labels = less_mask.sum(dim=-1)
    
    return labels


if __name__ == '__main__':
    edge_index_example = torch.tensor([
        [0, 1, 1, 2, 2, 3],  # sources
        [1, 0, 2, 1, 3, 2]   # destinations
    ], dtype=torch.long)
    num_nodes_example = 5  # Nodes 0, 1, 2, 3, 4 (node 4 is isolated)

    walker = Walker(edge_index_example, num_nodes_example)
    
    start_nodes_example = torch.tensor([0, 1, 4], dtype=torch.long)
    walk_length_example = 5

    print(f'Graph: {num_nodes_example} nodes')
    print(f'Edge Index:\n{edge_index_example}')
    print(f'Start Nodes: {start_nodes_example}')
    print(f'Walk Length: {walk_length_example}\n')

    walks = walker.uniform_random_walk(start_nodes_example, walk_length_example)
    
    print(walks)

    data_example = Data(edge_index=edge_index_example, num_nodes=num_nodes_example)
    
