from walker import anonymise_walks
from utils import squareplus
from rnn import GRU

from torch_geometric.utils import degree, softmax
from torch import nn
import torch

class RUMLayer(torch.nn.Module):
    def __init__(self, nin, nout, original_features,
            num_samples,
            length,
            dropout = 0.2,
            rnn = GRU,
            walker = None,
            activation = torch.nn.Identity(),
            edge_features = 0,
            binary = True,
            directed = False,
            degrees = True,
            self_supervise = True,
            rnn_nlayer=1,
    ):
        super().__init__()
        # nout = nout // 2
        # self.fc = torch.nn.Linear(nin + 2 * nout + 1, nout, bias=False)
        self.rnn = rnn(nin + 2 * nout + int(degrees), nout, num_layers=rnn_nlayer)
        self.rnn_walk = rnn(2, nout, bidirectional=True, num_layers=rnn_nlayer)
        if edge_features > 0:
            self.fc_edge = torch.nn.Linear(edge_features, int(degrees) + nin + 2 * nout, bias=False)
        self.nin = nin
        self.nout = nout
        self.walker = walker
        self.num_samples = num_samples
        self.length = length
        self.dropout = torch.nn.Dropout(dropout)
        if self_supervise:
            self.self_supervise = SelfSupervise(nin, original_features, binary=binary)
        else:
            self.self_supervise = None
        self.activation = activation
        self.directed = directed
        self.degrees = degrees

    def forward(self, x, edge_index, y0, e=None):
        walks = self.walker.uniform_random_walk(
            start_nodes=edge_index[0],
            walk_length=self.length,
            num_walks_per_node=self.num_samples,
        )
            

        if self.directed:
            walks = torch.where(
                walks == -1,
                walks[..., 0:1],
                walks,
            )

    
        uniqueness_walk = anonymise_walks(walks)
        walks, uniqueness_walk = walks.flip(-1), uniqueness_walk.flip(-1)
        uniqueness_walk = uniqueness_walk / uniqueness_walk.shape[-1]
        uniqueness_walk = uniqueness_walk * torch.pi * 2.0
        uniqueness_walk = torch.cat(
            [
                uniqueness_walk.sin().unsqueeze(-1),
                uniqueness_walk.cos().unsqueeze(-1),
            ],
            dim=-1,
        )

        x = x[walks]
        
        # rnn_walk.bidirectional is almost hardcoded to True
        num_directions = 2 if self.rnn_walk.bidirectional else 1 
        h0 = torch.zeros(self.rnn_walk.num_layers * num_directions, *x.shape[:-2], self.nout, device=x.device)

        y_walk, h = self.rnn_walk(uniqueness_walk, h0)

        h = h.mean(0, keepdim=True) # why use this as init RNN
        
        if self.rnn.num_layers > 1:
            h = h.repeat(self.rnn.num_layers, 1, 1, 1)

        if self.degrees:
            # I hope g.in_degrees is equivalent to degrees(edge_index[1])
            # degrees = g.in_degrees(walks.flatten()).float().reshape(*walks.shape).unsqueeze(-1)
            degrees = self.walker.dst_degrees[walks.flatten()].float().reshape(*walks.shape).unsqueeze(-1)
            degrees = degrees / degrees.max()
            x = torch.cat([x, y_walk, degrees], dim=-1)
        else:
            x = torch.cat([x, y_walk], dim=-1) 

        # x = self.fc(x) # original code
        # x = self.activation(x) # original code

        # TODO: eids not implemented
        if e is not None:
            _h = torch.empty(
                *x.shape[:-2],
                2 * x.shape[-2] - 1,
                x.shape[-1],
                device=x.device,
                dtype=x.dtype,
            )
            _h[..., ::2, :] = x
            _h[..., 1::2, :] = self.fc_edge(e)[eids]
            x = _h

        y, x = self.rnn(x, h)

        if self.training and self.self_supervise:
            if e is not None:
                y = y[..., ::2, :]
            self_supervise_loss = self.self_supervise(y, y0[walks])
        else:
            self_supervise_loss = 0.0
        x = self.activation(x)
        x = x.mean(0)
        x = self.dropout(x)
        return x, self_supervise_loss

class Consistency(torch.nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, probs):
        avg_probs = probs.mean(0)
        sharpened_probs = avg_probs.pow(1 / self.temperature)
        sharpened_probs = sharpened_probs / sharpened_probs.sum(-1, keepdim=True)
        loss = (sharpened_probs - avg_probs).pow(2).sum(-1).mean()
        return loss

class SelfSupervise(torch.nn.Module):
    def __init__(self, nin, nout, subsample_size=100, binary=True):
        super().__init__()
        self.fc = torch.nn.Linear(nin, nout)
        self.subsample_size = subsample_size
        self.binary = binary

    def forward(self, y_hat, y):
        idxs = torch.randint(high=y_hat.shape[-3], size=(self.subsample_size, ), device=y.device)
        y, y_hat = y.flatten(0, -3), y_hat.flatten(0, -3)
        y = y[..., idxs, 1:, :].contiguous()
        y_hat = y_hat[..., idxs, :-1, :].contiguous()
        y_hat = self.fc(y_hat)
        if self.binary:
            loss = torch.nn.BCEWithLogitsLoss(
                pos_weight=y.detach().mean().pow(-1)
            )(y_hat, y)
        else:
            # loss = torch.nn.CrossEntropyLoss()(y_hat, y) # original code
            loss = torch.nn.MSELoss()(y_hat, y)
        return loss 
    



class SpGraphTransAttention(nn.Module):
    """
    Sparse version GAT layer, code adapted from: https://github.com/twitter-research/graph-neural-pde/blob/main/src/function_transformer_attention.py
    """

    def __init__(self,  nin, natt, nhead, edge_weights=None):
        self.alpha = opt['leaky_relu_slope']
        self.edge_weights = edge_weights

        super(SpGraphTransAttention, self).__init__()
        self.nin = nin
        self.nhead = nhead
        self.natt = natt

        assert self.natt % self.nhead == 0, "Number of nhead ({}) must be a factor of the dimension size ({})".format(
        self.nhead, self.natt)
        self.d_k = self.natt // self.nhead

        self.Q = nn.Linear(nin, self.natt)
        self.K = nn.Linear(nin, self.natt)
        self.V = nn.Linear(nin, self.natt)

        self.activation = nn.Sigmoid()  

        self.init_weights(self.Q)
        self.init_weights(self.K)
        self.init_weights(self.V)

        self.Wout = nn.Linear(self.d_k, nin)
        self.init_weights(self.Wout)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            # nn.init.xavier_uniform_(m.weight, gain=1.414)
            # m.bias.data.fill_(0.01)
            nn.init.constant_(m.weight, 1e-5)

    def forward(self, x, edge):
        """
        x might be [features, augmentation, positional encoding, labels]
        """
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)

        # perform linear operation and split into nhead
        k = k.view(-1, self.nhead, self.d_k)
        q = q.view(-1, self.nhead, self.d_k)
        v = v.view(-1, self.nhead, self.d_k)

        # transpose to get dimensions [n_nodes, attention_dim, n_heads]
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        src = q[edge[0, :], :, :]
        dst_k = k[edge[1, :], :, :]

        if self.opt['attention_type'] == "scaled_dot":
            prods = torch.sum(src * dst_k, dim=1) / torch.sqrt(self.d_k)
        elif self.opt['attention_type'] == "cosine_sim":
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-5)
            prods = cos(src, dst_k)
        elif self.opt['attention_type'] == "pearson":
            src_mu = torch.mean(src, dim=1, keepdim=True)
            dst_mu = torch.mean(dst_k, dim=1, keepdim=True)
            src = src - src_mu
            dst_k = dst_k - dst_mu
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-5)
            prods = cos(src, dst_k)

        # reweighs attention based on edge_weights
        if self.edge_weights is not None:
            prods = prods * self.edge_weights.unsqueeze(dim=1)
        if self.opt['square_plus']:
            attention = squareplus(prods, edge[self.opt['attention_norm_idx']])
        else:
            attention = softmax(prods, edge[self.opt['attention_norm_idx']])
        return attention, (v, prods)






