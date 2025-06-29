import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree


class Walker():
    def __init__(self, edge_index, num_nodes=None):
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
    
