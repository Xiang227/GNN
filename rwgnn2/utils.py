import numpy as np

import torch
import torch.utils.data as utils

from math import ceil
from scipy.sparse import csr_matrix,lil_matrix
from sklearn.preprocessing import OneHotEncoder


def load_data(ds_name, use_node_labels):    
    graph_indicator = np.loadtxt("datasets/%s/%s_graph_indicator.txt"%(ds_name,ds_name), dtype=np.int64)
    _,graph_size = np.unique(graph_indicator, return_counts=True)
    
    edges = np.loadtxt("datasets/%s/%s_A.txt"%(ds_name,ds_name), dtype=np.int64, delimiter=",")
    edges -= 1
    A = csr_matrix((np.ones(edges.shape[0]), (edges[:,0], edges[:,1])), shape=(graph_indicator.size, graph_indicator.size))
    
    if use_node_labels:
        x = np.loadtxt("datasets/%s/%s_node_labels.txt"%(ds_name,ds_name), dtype=np.int64).reshape(-1,1)
        enc = OneHotEncoder(sparse=False)
        x = enc.fit_transform(x)
    else:
        x = A.sum(axis=1)
        
    adj = []
    features = []
    idx = 0
    for i in range(graph_size.size):
        adj.append(A[idx:idx+graph_size[i],idx:idx+graph_size[i]])
        features.append(x[idx:idx+graph_size[i],:])
        idx += graph_size[i]

    class_labels = np.loadtxt("datasets/%s/%s_graph_labels.txt"%(ds_name,ds_name), dtype=np.int64)
    return adj, features, class_labels


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def generate_batches(adj, features, y, batch_size, device, shuffle=False):
    N = len(y)
    if shuffle:
        index = np.random.permutation(N)
    else:
        index = np.array(range(N), dtype=np.int32)

    n_batches = ceil(N/batch_size)

    adj_lst = list()
    features_lst = list()
    graph_indicator_lst = list()
    y_lst = list()

    for i in range(0, N, batch_size):
        n_graphs = min(i+batch_size, N) - i
        n_nodes = sum([adj[index[j]].shape[0] for j in range(i, min(i+batch_size, N))])

        adj_batch = lil_matrix((n_nodes, n_nodes))
        features_batch = np.zeros((n_nodes, features[0].shape[1]))
        graph_indicator_batch = np.zeros(n_nodes)
        y_batch = np.zeros(n_graphs)

        idx = 0
        for j in range(i, min(i+batch_size, N)):
            n = adj[index[j]].shape[0]
            adj_batch[idx:idx+n, idx:idx+n] = adj[index[j]]    
            features_batch[idx:idx+n,:] = features[index[j]]
            graph_indicator_batch[idx:idx+n] = j-i
            y_batch[j-i] = y[index[j]]

            idx += n
                  
        adj_lst.append(sparse_mx_to_torch_sparse_tensor(adj_batch).to(device))
        features_lst.append(torch.FloatTensor(features_batch).to(device))
        graph_indicator_lst.append(torch.LongTensor(graph_indicator_batch).to(device))
        y_lst.append(torch.LongTensor(y_batch).to(device))

    return adj_lst, features_lst, graph_indicator_lst, y_lst


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def compute_shortest_path_distances(adj_matrix, n_nodes, max_distance=5):
    """
    使用BFS算法计算所有节点对之间的最短路径距离（限制最大距离以提高效率）
    
    参数:
        adj_matrix: 邻接矩阵 (scipy.sparse.csr_matrix 或 numpy.ndarray)
        n_nodes: 节点数量
        max_distance: 最大计算距离，超过此距离的设为max_distance+1
    
    返回:
        distance_matrix: 距离矩阵 (numpy.ndarray)
    """
    from collections import deque
    
    # 转换为密集矩阵
    if hasattr(adj_matrix, 'toarray'):
        adj_dense = adj_matrix.toarray()
    else:
        adj_dense = adj_matrix
    
    # 构建邻接列表以提高效率
    adj_list = [[] for _ in range(n_nodes)]
    for i in range(n_nodes):
        for j in range(n_nodes):
            if adj_dense[i, j] > 0:
                adj_list[i].append(j)
    
    # 初始化距离矩阵
    distance_matrix = np.full((n_nodes, n_nodes), max_distance + 1, dtype=np.float32)
    np.fill_diagonal(distance_matrix, 0)
    
    # 对每个节点使用BFS计算到其他节点的距离
    for start in range(n_nodes):
        if start % 100 == 0:  # 进度提示
            print(f"Computing distances from node {start}/{n_nodes}")
            
        visited = np.zeros(n_nodes, dtype=bool)
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

def compute_walk_probabilities(distance_matrix, alpha=1.0):
    """
    基于距离计算游走概率
    
    参数:
        distance_matrix: 距离矩阵 (numpy.ndarray)
        alpha: 控制距离影响的参数
    
    返回:
        walk_probs: 游走概率矩阵 (numpy.ndarray)
    """
    # 计算基于距离的转移概率
    # P(v|u) = exp(-alpha * d(u,v)) / Σ exp(-alpha * d(u,w))
    exp_distances = np.exp(-alpha * distance_matrix)
    # 避免除以零
    row_sums = exp_distances.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    walk_probs = exp_distances / row_sums
    
    return walk_probs

def add_distance_features(features, distance_matrix):
    """
    将距离特征添加到节点特征中
    
    参数:
        features: 原始节点特征 (numpy.ndarray)
        distance_matrix: 距离矩阵 (numpy.ndarray)
    
    返回:
        enhanced_features: 增强后的特征 (numpy.ndarray)
    """
    # 对距离矩阵进行归一化
    max_dist = np.max(distance_matrix)
    if max_dist > 0:
        normalized_distances = distance_matrix / max_dist
    else:
        normalized_distances = distance_matrix
    
    # 将距离特征与原始特征拼接
    enhanced_features = np.hstack([features, normalized_distances])
    
    return enhanced_features