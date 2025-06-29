'''
Code adapted from https://github.com/twitter-research/graph-neural-pde/blob/main/src/data.py
'''

from torch_geometric.datasets import Planetoid, Amazon, Coauthor, WebKB, ZINC, HeterophilousGraphDataset, WikipediaNetwork, Twitch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
import torch

from ogb.nodeproppred import PygNodePropPredDataset
import numpy as np

# TODO: ZINC
def get_dataset(dataset_name, data_dir='./data/', use_lcc=False, seed=12345, split=0) -> InMemoryDataset:
    if split < 0:
        split = np.random.randint(0, 10)
    if dataset_name in ['Cora', 'Citeseer', 'Pubmed', 'Computers', 'Photo', 'CoauthorCS', 'ogbn-arxiv']:
        return get_homo_dataset(dataset_name, data_dir, use_lcc, seed)
    elif dataset_name in ['Cornell', 'Texas', 'Wisconsin']:
        return get_webkb_dataset(dataset_name, data_dir, split=split)
    elif dataset_name in ['Roman-empire', 'Amazon-ratings', 'Minesweeper', 'Tolokers', 'Questions', 'Chameleon', 'Crocodile', 'Squirrel']:
        return get_other_dataset(dataset_name, data_dir, split=split)
    elif dataset_name in ['DE', 'EN', 'ES', 'FR', 'PT', 'RU']:
        return get_twitch_dataset(dataset_name, data_dir)
    else:
        raise Exception('Unknown dataset.')


def get_homo_dataset(dataset_name, data_dir='./data/', use_lcc=False, seed=12345) -> InMemoryDataset:
    if dataset_name in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(data_dir, dataset_name)
    elif dataset_name in ['Computers', 'Photo']:
        dataset = Amazon(data_dir, dataset_name)
    elif dataset_name == 'CoauthorCS':
        dataset = Coauthor(data_dir, 'CS')
    elif dataset_name == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(name=dataset_name, root=data_dir,
                                         transform=T.ToSparseTensor())
        use_lcc = False  # never need to calculate the lcc with ogb datasets
    else:
        raise Exception('Unknown dataset.')

    if use_lcc:
        lcc = get_largest_connected_component(dataset)

        x_new = dataset._data.x[lcc]
        y_new = dataset._data.y[lcc]

        row, col = dataset._data.edge_index.numpy()
        edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]
        edges = remap_edges(edges, get_node_mapper(lcc))

        data = Data(
            x=x_new,
            edge_index=torch.LongTensor(edges),
            y=y_new,
            train_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
            test_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
            val_mask=torch.zeros(y_new.size()[0], dtype=torch.bool)
        )
        dataset._data = data

    train_mask_exists = True
    try:
        dataset._data.train_mask
    except AttributeError:
        train_mask_exists = False

    if dataset_name == 'ogbn-arxiv':
        split_idx = dataset.get_idx_split()
        ei = to_undirected(dataset._data.edge_index)
        data = Data(
            x=dataset._data.x,
            edge_index=ei,
            y=dataset._data.y,
            train_mask=split_idx['train'],
            test_mask=split_idx['test'],
            val_mask=split_idx['valid'])
        dataset._data = data
        train_mask_exists = True

    if use_lcc or not train_mask_exists:
        dataset._data = set_train_val_test_split(
            seed,
            dataset._data,
            num_development=5000 if dataset_name == 'CoauthorCS' else 1500)

    return dataset

def get_webkb_dataset(dataset_name, data_dir='./data/', split=0):
    
    assert dataset_name in ['Cornell', 'Texas', 'Wisconsin']
    
    dataset = WebKB(data_dir,name=dataset_name)
    
    splits_file = np.load(f'{data_dir}/{dataset_name}/raw/{dataset_name}_split_0.6_0.2_{split}.npz')
    train_mask = splits_file['train_mask']
    val_mask = splits_file['val_mask']
    test_mask = splits_file['test_mask']

    dataset._data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
    dataset._data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
    dataset._data.test_mask = torch.tensor(test_mask, dtype=torch.bool)

    return dataset

def get_other_dataset(dataset_name, data_dir='./data/', split=0):
    if dataset_name == 'ZINC':
        train_set = ZINC(data_dir, subset=True, split='train')
        val_set = ZINC(data_dir, subset=True, split='train')
        test_set = ZINC(data_dir, subset=True, split='train')
        return train_set, val_set, test_set
    elif dataset_name in ['Roman-empire', 'Amazon-ratings', 'Minesweeper', 'Tolokers', 'Questions']:
        dataset = HeterophilousGraphDataset(data_dir, name=dataset_name)
        dataset._data.train_mask = dataset._data.train_mask[:, split_num]
        dataset._data.val_mask = dataset._data.val_mask[:, split_num]
        dataset._data.test_mask = dataset._data.test_mask[:, split_num]
    elif dataset_name in ['Chameleon', 'Crocodile', 'Squirrel']:
        dataset = WikipediaNetwork(data_dir, name=dataset_name)
        dataset._data.train_mask = dataset._data.train_mask[:, split_num]
        dataset._data.val_mask = dataset._data.val_mask[:, split_num]
        dataset._data.test_mask = dataset._data.test_mask[:, split_num]
    else:
        raise Exception('Unknown dataset.')
    
    return dataset

def get_twitch_dataset(dataset_name, data_dir='./data/'): 
    assert dataset_name in ['DE', 'EN', 'ES', 'FR', 'PT', 'RU']
    
    dataset = Twitch(data_dir, name=dataset_name)
    num_nodes = dataset._data.num_nodes
    perm = torch.randperm(num_nodes)

    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)

    train_idx = perm[:train_size]
    val_idx = perm[train_size:train_size + val_size]
    test_idx = perm[train_size + val_size:]

    dataset._data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    dataset._data.train_mask[train_idx] = True

    dataset._data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    dataset._data.val_mask[val_idx] = True

    dataset._data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    dataset._data.test_mask[test_idx] = True
    return dataset

def get_node_mapper(lcc: np.ndarray) -> dict:
    mapper = {}
    counter = 0
    for node in lcc:
        mapper[node] = counter
        counter += 1
    return mapper


def remap_edges(edges: list, mapper: dict) -> list:
    row = [e[0] for e in edges]
    col = [e[1] for e in edges]
    row = list(map(lambda x: mapper[x], row))
    col = list(map(lambda x: mapper[x], col))
    return [row, col]


def get_component(dataset: InMemoryDataset, start: int = 0) -> set:
    visited_nodes = set()
    queued_nodes = set([start])
    row, col = dataset._data.edge_index.numpy()
    while queued_nodes:
        current_node = queued_nodes.pop()
        visited_nodes.update([current_node])
        neighbors = col[np.where(row == current_node)[0]]
        neighbors = [n for n in neighbors if n not in visited_nodes and n not in queued_nodes]
        queued_nodes.update(neighbors)
    return visited_nodes


def get_largest_connected_component(dataset: InMemoryDataset) -> np.ndarray:
    remaining_nodes = set(range(dataset._data.x.shape[0]))
    comps = []
    while remaining_nodes:
        start = min(remaining_nodes)
        comp = get_component(dataset, start)
        comps.append(comp)
        remaining_nodes = remaining_nodes.difference(comp)
    return np.array(list(comps[np.argmax(list(map(len, comps)))]))


def set_train_val_test_split(
        seed: int,
        data: Data,
        num_development: int = 1500,
        num_per_class: int = 20) -> Data:
    rnd_state = np.random.RandomState(seed)
    num_nodes = data.y.shape[0]
    development_idx = rnd_state.choice(num_nodes, num_development, replace=False)
    test_idx = [i for i in np.arange(num_nodes) if i not in development_idx]

    train_idx = []
    rnd_state = np.random.RandomState(seed)
    for c in range(data.y.max() + 1):
        class_idx = development_idx[np.where(data.y[development_idx].cpu() == c)[0]]
        train_idx.extend(rnd_state.choice(class_idx, num_per_class, replace=False))

    val_idx = [i for i in development_idx if i not in train_idx]

    def get_mask(idx):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[idx] = 1
        return mask

    data.train_mask = get_mask(train_idx)
    data.val_mask = get_mask(val_idx)
    data.test_mask = get_mask(test_idx)

    return data