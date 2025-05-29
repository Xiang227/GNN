import argparse
import time
import numpy as np
from math import ceil
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Parameter
from torch.optim import Adam

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops, degree
from torch_geometric import transforms as T

class RWConv(MessagePassing):
    """随机游走卷积层，基于PyG的MessagePassing框架"""
    def __init__(self, in_channels, out_channels):
        super(RWConv, self).__init__(aggr='add')
        self.lin = Linear(in_channels, out_channels)
        
    def forward(self, x, edge_index):
        # 添加自环
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # 计算归一化系数
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv = deg.pow(-1.0)
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[row]
        
        # 开始消息传递
        return self.propagate(edge_index, x=x, norm=norm)
    
    def message(self, x_j, norm):
        # 定义消息函数 - 随机游走中每个邻居的消息乘以归一化系数
        return norm.view(-1, 1) * x_j
    
    def update(self, aggr_out):
        # 更新函数 - 可选的线性变换
        return self.lin(aggr_out)

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
        
        # 学习的隐藏图参数
        self.adj_hidden = Parameter(torch.FloatTensor(hidden_graphs, 
                                    (size_hidden_graphs*(size_hidden_graphs-1))//2))
        self.features_hidden = Parameter(torch.FloatTensor(hidden_graphs, 
                                         size_hidden_graphs, hidden_dim))
        
        # 特征转换层
        self.fc = Linear(input_dim, hidden_dim)
        
        # 随机游走层
        self.rw_layers = nn.ModuleList([
            RWConv(hidden_dim, hidden_dim) for _ in range(max_step)
        ])
        
        # 批归一化层 
        self.bn = None  # 将在forward中动态创建
        
        # 输出层 - 动态创建
        self.fc1 = None
        self.fc2 = None
        
        # 正则化层
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # 初始化参数
        self.init_weights()
    
    def init_weights(self):
        self.adj_hidden.data.uniform_(-1, 1)
        self.features_hidden.data.uniform_(0, 1)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 获取图的数量和每个图的节点数
        unique, counts = torch.unique(batch, return_counts=True)
        n_graphs = unique.size(0)
        
        if self.normalize:
            norm = counts.unsqueeze(1).repeat(1, self.hidden_graphs)
        
        # 创建隐藏图的邻接矩阵
        adj_hidden_norm = torch.zeros(self.hidden_graphs, self.size_hidden_graphs, 
                                     self.size_hidden_graphs, device=self.device)
        idx = torch.triu_indices(self.size_hidden_graphs, self.size_hidden_graphs, 1, 
                                device=self.device)
        adj_hidden_norm[:,idx[0],idx[1]] = self.relu(self.adj_hidden)
        adj_hidden_norm = adj_hidden_norm + torch.transpose(adj_hidden_norm, 1, 2)
        
        # 初始特征转换
        x = self.sigmoid(self.fc(x))
        z = self.features_hidden
        zx = torch.einsum("abc,dc->abd", (z, x))
        
        # 多步随机游走
        out = []
        current_x = x
        for i in range(self.max_step):
            if i == 0:
                eye = torch.eye(self.size_hidden_graphs, device=self.device)
                eye = eye.repeat(self.hidden_graphs, 1, 1)              
                o = torch.einsum("abc,acd->abd", (eye, z))
                t = torch.einsum("abc,dc->abd", (o, current_x))
            else:
                # 对输入图执行随机游走
                current_x = self.rw_layers[i-1](current_x, edge_index)
                
                # 对隐藏图执行随机游走
                z = torch.einsum("abc,acd->abd", (adj_hidden_norm, z))
                t = torch.einsum("abc,dc->abd", (z, current_x))
            
            t = self.dropout(t)
            t = torch.mul(zx, t)
            
            # 将结果聚合到图级别
            t_graphs = torch.zeros(t.size(0), t.size(1), n_graphs, device=self.device)
            for node_idx in range(t.size(2)):
                graph_idx = batch[node_idx]
                t_graphs[:, :, graph_idx] += t[:, :, node_idx]
            
            # 转置并归一化
            t = torch.transpose(t_graphs, 0, 1)
            if self.normalize:
                t = t / norm.unsqueeze(2)
            
            t = torch.transpose(t, 1, 2)
            t = t.reshape(n_graphs, -1)
            out.append(t)
        
        # 连接所有步骤的结果
        out = torch.cat(out, dim=1)
        
        # 动态创建或使用BatchNorm1d层
        if self.bn is None or self.bn.num_features != out.size(1):
            self.bn = BatchNorm1d(out.size(1)).to(self.device)
            
        out = self.bn(out)
        
        # 动态创建全连接层
        if self.fc1 is None or self.fc1.in_features != out.size(1):
            self.fc1 = Linear(out.size(1), self.penultimate_dim).to(self.device)
            self.fc2 = Linear(self.penultimate_dim, self.num_classes).to(self.device)
            
        out = self.relu(self.fc1(out))
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
    parser.add_argument('--hidden-graphs', type=int, default=16, help='Number of hidden graphs')
    parser.add_argument('--size-hidden-graphs', type=int, default=5, 
                        help='Number of nodes in each hidden graph')
    parser.add_argument('--hidden-dim', type=int, default=4, help='Hidden dimension')
    parser.add_argument('--penultimate-dim', type=int, default=32, 
                        help='Penultimate layer dimension')
    parser.add_argument('--max-step', type=int, default=2, help='Maximum walk length')
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