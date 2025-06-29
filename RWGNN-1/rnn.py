import torch

class GRU(torch.nn.GRU):
    def __init__(self, *args, **kwargs):
        kwargs['batch_first'] = True
        # kwargs['bidirectional'] = True
        super().__init__(*args, **kwargs)
    
    def forward(self, x, h0):
        num_direction = 2 if self.bidirectional else 1
        batch_shape = x.shape[:-2]
        event_shape_x = x.shape[-2:]
        event_shape_h0 = h0.shape[-1:]
        x = x.view(-1, *event_shape_x)
        h0 = h0.view(num_direction * self.num_layers, -1, *event_shape_h0)
        
        output, h_n = super().forward(x, h0)

        output = output.view(*batch_shape, *output.shape[-2:])
        h_n = h_n.view(num_direction * self.num_layers, *batch_shape, *h_n.shape[-1:])
        return output, h_n

class LSTM(torch.nn.LSTM):
    def __init__(self, *args, **kwargs):
        kwargs['batch_first'] = True
        super().__init__(*args, **kwargs)

    def forward(self, x):
        batch_shape = x.shape[:-2]
        event_shape_x = x.shape[-2:]
        x = x.view(-1, *event_shape_x)
        output, h_n = super().forward(x)
        output = output.view(*batch_shape, *output.shape[-2:])
        h_n = h_n.view(self.num_layers, *batch_shape, *h_n.shape[-1:])
        return output, h_n
