from layer import RUMLayer, Consistency
from rnn import GRU

import torch.nn.functional as F
import torch


class RUMModel(torch.nn.Module):
    def __init__(self, nin, nhid, nout,
                 nlayer,
                 num_samples=2,
                 length=4,
                 temperature=0.1,
                 dropout=0.2,
                 rnn_nlayer=1,
                 self_supervise_weight=0.05,
                 consistency_weight=0.01,
                 activation=torch.nn.ELU(),
                 edge_features=0.,
                 walker=None,
                 walk_type='uniform',
                 levy_alpha=1.5,
                 levy_ratio=0.5,
                 adaptive_levy=False,
                 alpha_schedule=None,
                 ):
        
        super().__init__()
        self.enc = torch.nn.Linear(nin, nhid, bias=True)
        self.dec = torch.nn.Linear(nhid, nout, bias=True)
        self.nin = nin
        self.nhid = nhid
        self.nout = nout
        self.nlayer = nlayer
        self.activation = activation
        self.consistency = Consistency(temperature=temperature)
        self.self_supervise_weight = self_supervise_weight
        self.consistency_weight = consistency_weight
        
        self.walk_type = walk_type
        self.levy_alpha = levy_alpha
        self.levy_ratio = levy_ratio
        self.adaptive_levy = adaptive_levy
        self.alpha_schedule = alpha_schedule

        self.layers = torch.nn.ModuleList()
        for _ in range(nlayer):
            self.layers.append(RUMLayer(nhid, nhid, nin, num_samples, length,
                                        dropout=dropout, 
                                        rnn=GRU, 
                                        walker=walker,
                                        activation=activation,
                                        edge_features=edge_features,
                                        rnn_nlayer=rnn_nlayer,
                                        walk_type=walk_type,
                                        levy_ratio=levy_ratio,
                                        adaptive_levy=adaptive_levy,
                                        alpha_schedule=alpha_schedule,
                                        ))
            

    def forward(self, data, e=None):
        x, edge_index = data.x, data.edge_index
        h0 = x.clone() 
        x = self.enc(x)

        loss = 0.0

        for idx, layer in enumerate(self.layers):
            if idx > 0:
                x = x.mean(0) 
            x, self_supervise_loss = layer(x, edge_index, h0, e=e)
            loss = loss + self.self_supervise_weight * self_supervise_loss

        x = self.dec(x)
        x = F.softmax(x, dim=-1)

        if self.training:
            _loss = self.consistency(x)
            _loss = _loss * self.consistency_weight
            loss = loss + _loss
        return x, loss

class RUMGraphRegressionModel(RUMModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dec = torch.nn.Sequential(
            # torch.nn.BatchNorm1d(self.nhid),
            self.activation,
            torch.nn.Linear(self.nhid, self.nhid),
            self.activation,
            torch.nn.Dropout(kwargs['dropout']),
            torch.nn.Linear(self.nhid, self.nout),
        )


    def forward(self, g, h, e=None, subsample=None):
        g = g.local_var()
        h0 = h
        h = self.enc(h)
        loss = 0.0
        for idx, layer in enumerate(self.layers):
            if idx > 0:
                # h = torch.nn.functional.tanh(h)
                h = torch.nn.SiLU()(h)
                h = h.mean(0)
            h, _loss = layer(g, h, h0, e=e, subsample=subsample)
            loss = loss + self.self_supervise_weight * _loss
        # h = self.activation(h)
        h = h.mean(0)
        g.ndata['h'] = h
        # h = dgl.sum_nodes(g, 'h')
        h = dgl.mean_nodes(g, 'h')
        h = self.dec(h)
        return h, loss