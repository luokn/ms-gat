import torch
from torch import FloatTensor, LongTensor
from torch.nn import Conv2d, Embedding, LayerNorm, Module, ModuleList, Parameter, Sequential


class GACN(Module):
    def __init__(self, in_channels, out_channels, in_timesteps, adj):
        super(GACN, self).__init__()
        self.adj = adj
        self.W = Parameter(torch.zeros(out_channels, in_channels), requires_grad=True)

    def forward(self, x: FloatTensor) -> FloatTensor:
        # [n_nodes, n_nodes] × [batch_size, n_timesteps, n_nodes, in_channels] × [in_channels, out_channels]
        x_out = self.adj @ x.transpose(1, 3) @ self.W.T  # -> [batch_size, n_timesteps, n_nodes, out_channels]
        return x_out.transpose(1, 3)  # -> [batch_size, out_channels, n_timesteps, n_nodes]


class Chomp(Module):
    def __init__(self, chomp_size):
        super(Chomp, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: FloatTensor) -> FloatTensor:
        return x[..., :-self.chomp_size]


class TACN(Module):
    def __init__(self, in_channels, out_channels, dilations, n_nodes):
        super(TACN, self).__init__()
        channels = [in_channels] + [out_channels] * len(dilations)
        seq = []
        for i, dilation in enumerate(dilations):
            seq += [
                Conv2d(channels[i], channels[i + 1], [1, 2], padding=[0, dilation], dilation=[1, dilation]),
                Chomp(dilation)
            ]
        self.seq = Sequential(*seq)

    def forward(self, x: FloatTensor) -> FloatTensor:
        return self.seq(x)  # -> [batch_size, out_channels, n_nodes, in_timesteps]


class CACN(Module):
    def __init__(self, in_channels, out_channels, in_timesteps, n_nodes):
        super(CACN, self).__init__()
        self.conv = Conv2d(in_channels, out_channels, 1)

    def forward(self, x: FloatTensor) -> FloatTensor:
        return self.conv(x)  # -> [batch_size, out_channels, n_nodes, in_timesteps]


class TGACN(Module):
    def __init__(self, in_channels, out_channels, in_timesteps, tcn_dilations, n_nodes, **kwargs):
        super(TGACN, self).__init__()
        self.ln = LayerNorm([in_timesteps])
        self.acns = ModuleList([
            CACN(in_channels, out_channels // 3, in_timesteps, n_nodes),
            GACN(in_channels, out_channels // 3, in_timesteps, kwargs['adj']),
            TACN(in_channels, out_channels // 3, tcn_dilations, n_nodes)
        ])
        self.res = Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: FloatTensor) -> FloatTensor:
        out = self.ln(x)  # -> [batch_size, in_channels, n_nodes, in_timesteps]
        # -> [batch_size, out_channels, n_nodes, in_timesteps]
        out = torch.cat([f(out) for f in self.acns], dim=1) + self.res(x)
        return torch.relu(out)  # -> [batch_size, out_channels, n_nodes, in_timesteps]


class MSGCNLayer(Module):
    def __init__(self, blocks, **kwargs):
        super(MSGCNLayer, self).__init__()
        self.seq = Sequential(*[
            TGACN(**block, **kwargs) for block in blocks
        ], LayerNorm([kwargs['in_timesteps']]))
        self.fc = Conv2d(kwargs['in_timesteps'], kwargs['out_timesteps'], [1, blocks[-1]['out_channels']])

    def forward(self, x: FloatTensor) -> FloatTensor:
        out = self.seq(x)  # -> [batch_size, out_channels, n_nodes, in_timesteps]
        out = self.fc(out.transpose(1, 3))  # -> [batch_size, out_timesteps, n_nodes, 1]
        return out[..., 0].transpose(1, 2)  # -> [batch_size, n_nodes, out_timesteps]


class MSGCN(Module):
    def __init__(self, layers, **kwargs):
        super(MSGCN, self).__init__()
        n_nodes, out_timesteps = kwargs['n_nodes'], kwargs['out_timesteps']
        self.d_ebd = Embedding(7, len(layers) * n_nodes * out_timesteps)
        self.h_ebd = Embedding(24, len(layers) * n_nodes * out_timesteps)
        self.layers = ModuleList([MSGCNLayer(**layer, **kwargs) for layer in layers])

    def forward(self, X: FloatTensor, H: LongTensor, D: LongTensor) -> FloatTensor:
        G = self.h_ebd(H) + self.d_ebd(D)  # -> [batch_size, n_layers × n_nodes × out_timesteps]
        G = G.view(X.size(0), X.size(1), X.size(3), -1)  # -> [batch_size, n_layers, n_nodes, out_timesteps]
        # -> [batch_size, n_nodes, out_timesteps]
        return sum(map(self.gate_fusion, self.layers, X.unbind(1), G.unbind(1)))

    @staticmethod
    def gate_fusion(layer, x, gate):
        return layer(x) * gate
