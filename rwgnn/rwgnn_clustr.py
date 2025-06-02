import argparse
import time
import numpy as np
from math import ceil
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
from torch_cluster import random_walk  # 从torch_cluster导入random_walk

class ExplicitRandomWalkEncoder(nn.Module):
    """显式随机游走编码器，生成随机游走路径并编码"""
    def __init__(self, input_dim, hidden_dim, walk_length, num_walks, device):
        super(ExplicitRandomWalkEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.device = device
        
        # 特征转换层
        self.feature_encoder = Linear(input_dim, hidden_dim)
        
        # 路径编码层(GRU)
        self.path_encoder = GRU(hidden_dim, hidden_dim, batch_first=True)
    
    def sample_random_walks(self, edge_index, batch, num_nodes):
        """为每个图生成多条随机游走路径"""
        walks = []
        walk_batch = []  # 记录每条路径所属的图
        batch_size = torch.max(batch).item() + 1
        
        # 为每个图生成随机游走
        for graph_idx in range(batch_size):
            # 获取当前图的节点索引
            graph_mask = (batch == graph_idx)
            graph_nodes = torch.nonzero(graph_mask).squeeze()
            
            # 处理单节点情况
            if graph_nodes.dim() == 0:
                graph_nodes = graph_nodes.unsqueeze(0)
                
            # 如果图只有一个节点，则跳过随机游走
            if graph_nodes.size(0) <= 1:
                continue
                
            # 获取当前图的边索引
            edge_mask = torch.isin(edge_index[0], graph_nodes) & \
                        torch.isin(edge_index[1], graph_nodes)
            
            # 单个图的边索引
            graph_edge_index = edge_index[:, edge_mask]
            
            if graph_edge_index.numel() == 0:
                continue
                
            # 确保有向图转为无向图
            graph_edge_index = to_undirected(graph_edge_index)
            
            # 准备随机游走的输入
            row, col = graph_edge_index
            
            # 随机选择起始节点
            if self.num_walks >= graph_nodes.size(0):
                # 如果要求的游走数量大于等于节点数，使用所有节点作为起点
                start_nodes = graph_nodes
            else:
                # 否则随机选择节点
                indices = torch.randperm(graph_nodes.size(0))[:self.num_walks]
                start_nodes = graph_nodes[indices]
            
            # 生成随机游走路径
            for start_node in start_nodes:
                try:
                    # 将起始节点转换为张量并确保是长度为1的一维张量
                    start = start_node.view(-1)
                    
                    # 使用torch_cluster.random_walk生成路径
                    # 返回值是[batch_size, walk_length]的张量
                    walk = random_walk(row, col, start, walk_length=self.walk_length-1)[0]
                    
                    # 确保路径长度正确
                    if walk.size(0) < self.walk_length:
                        # 如果路径太短，用最后一个节点填充
                        last_node = walk[-1] if walk.size(0) > 0 else start_node
                        padding = torch.full((self.walk_length - walk.size(0),), 
                                           last_node.item(), 
                                           dtype=torch.long, device=self.device)
                        walk = torch.cat([walk, padding])
                    elif walk.size(0) > self.walk_length:
                        # 如果路径太长，截断
                        walk = walk[:self.walk_length]
                    
                    walks.append(walk)
                    walk_batch.append(graph_idx)  # 记录该路径属于哪个图
                except Exception as e:
                    print(f"随机游走出错: {e}")
                    # 创建一个备用路径 - 只包含起始节点，其余填充相同的值
                    walk = torch.full((self.walk_length,), start_node.item(), 
                                     dtype=torch.long, device=self.device)
                    walks.append(walk)
                    walk_batch.append(graph_idx)  # 记录该路径属于哪个图
        
        # 如果没有有效的随机游走，创建一个空的占位符
        if len(walks) == 0:
            return torch.zeros(1, self.walk_length, device=self.device, dtype=torch.long), torch.zeros(1, device=self.device, dtype=torch.long)
            
        # 将所有游走路径堆叠为一个批次
        walk_tensor = torch.stack(walks)
        walk_batch = torch.tensor(walk_batch, device=self.device)
        return walk_tensor, walk_batch
    
    def forward(self, x, edge_index, batch):
        # 生成随机游走路径
        walks, walk_batch = self.sample_random_walks(edge_index, batch, x.size(0))
        
        # 获取路径中节点的特征
        x_encoded = self.feature_encoder(x)
        
        # 为每个游走路径创建特征序列
        batch_size = walks.size(0)
        seq_len = walks.size(1)
        walk_features = torch.zeros(batch_size, seq_len, self.hidden_dim, device=self.device)
        
        # 收集每个路径节点的特征
        for i in range(batch_size):
            for j in range(seq_len):
                node_idx = walks[i, j].item()  # 获取整数索引
                if 0 <= node_idx < x.size(0):  # 确保索引有效
                    walk_features[i, j] = x_encoded[node_idx]
        
        # 使用GRU编码路径
        _, h_n = self.path_encoder(walk_features)
        path_encodings = h_n.squeeze(0)  # [batch_size, hidden_dim]
        
        # 返回路径编码和批次信息，用于后续聚合
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
        
        # 随机游走编码器
        self.walk_encoder = ExplicitRandomWalkEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            walk_length=max_step + 1,  # +1 包括起始节点
            num_walks=hidden_graphs,   # 每个图采样的随机游走数量
            device=device
        )
        
        # 学习的隐藏图参数 - 用于与随机游走路径交互
        self.adj_hidden = Parameter(torch.FloatTensor(hidden_graphs, 
                                    (size_hidden_graphs*(size_hidden_graphs-1))//2))
        self.features_hidden = Parameter(torch.FloatTensor(hidden_graphs, 
                                         size_hidden_graphs, hidden_dim))
        
        # 批归一化层
        self.bn = BatchNorm1d(hidden_dim)
        
        # 输出层
        self.fc1 = Linear(hidden_dim, penultimate_dim)
        self.fc2 = Linear(penultimate_dim, num_classes)
        
        # 正则化层
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        
        # 初始化参数
        self.init_weights()
    
    def init_weights(self):
        self.adj_hidden.data.uniform_(-1, 1)
        self.features_hidden.data.uniform_(0, 1)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 使用随机游走编码器获取路径编码和批次信息
        walk_encodings, walk_batch = self.walk_encoder(x, edge_index, batch)
        
        # 计算每个图的唯一ID
        unique_batch_ids = torch.unique(batch)
        batch_size = unique_batch_ids.size(0)
        
        # 为每个图聚合其所有随机游走路径的编码
        graph_encodings = torch.zeros(batch_size, walk_encodings.size(1), device=self.device)
        
        # 对每个图，平均其所有随机游走路径的编码
        for i, graph_idx in enumerate(unique_batch_ids):
            # 找到属于当前图的所有路径
            mask = (walk_batch == graph_idx)
            if mask.sum() > 0:  # 确保有路径
                # 计算平均编码
                graph_encodings[i] = walk_encodings[mask].mean(dim=0)
        
        # 应用批归一化
        graph_encodings = self.bn(graph_encodings)
        
        # 下游网络处理
        out = self.relu(self.fc1(graph_encodings))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return F.log_softmax(out, dim=1)

# 训练函数
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

# 测试函数
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
    # 参数设置
    parser = argparse.ArgumentParser(description='RW_GNN with PyTorch Geometric')
    parser.add_argument('--dataset', default='IMDB-BINARY', help='Dataset name')
    parser.add_argument('--use-node-labels', action='store_true', default=False, 
                        help='Whether to use node labels')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--hidden-graphs', type=int, default=16, help='Number of random walks per graph')
    parser.add_argument('--size-hidden-graphs', type=int, default=5, 
                        help='Parameter for internal representation')
    parser.add_argument('--hidden-dim', type=int, default=32, help='Hidden dimension')
    parser.add_argument('--penultimate-dim', type=int, default=64, 
                        help='Penultimate layer dimension')
    parser.add_argument('--max-step', type=int, default=4, help='Maximum walk length')
    parser.add_argument('--normalize', action='store_true', default=False, 
                        help='Whether to normalize features')
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据集
    dataset = TUDataset(root='data', name=args.dataset, use_node_attr=args.use_node_labels)
    
    if dataset.num_node_features == 0:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())
        
        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
    
    # 10折交叉验证
    kf = KFold(n_splits=10, shuffle=True, random_state=13)
    accs = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(range(len(dataset)))):
        print(f'Fold {fold+1}/10')
        
        # 分割数据集
        train_idx, val_idx = train_idx[:int(len(train_idx)*0.9)], train_idx[int(len(train_idx)*0.9):]
        
        train_dataset = dataset[train_idx.tolist()]
        val_dataset = dataset[val_idx.tolist()]
        test_dataset = dataset[test_idx.tolist()]
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        
        # 创建模型
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
        
        # 优化器和学习率调度器
        optimizer = Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        
        # 训练模型
        best_val_acc = 0
        best_model = None
        patience = 20
        patience_counter = 0
        
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            train_loss, train_acc = train(model, train_loader, optimizer, device)
            val_loss, val_acc = test(model, val_loader, device)
            scheduler.step()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Time: {time.time() - start:.4f}s')
            
            if patience_counter >= patience:
                print(f'Early stopping after {epoch} epochs')
                break
        
        # 测试最佳模型
        model.load_state_dict(best_model)
        test_loss, test_acc = test(model, test_loader, device)
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
        accs.append(test_acc)
    
    # 输出最终结果
    print(f'平均测试准确率: {np.mean(accs):.4f} ± {np.std(accs):.4f}')

if __name__ == '__main__':
    main() 