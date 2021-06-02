import torch
from torch.nn import Conv2d, Embedding, LayerNorm, Module, ModuleList, Parameter, Sequential


class GAttention(Module):
    def __init__(self, n_channels, in_timesteps, adj):
        super(GAttention, self).__init__()
        self.W = Parameter(torch.zeros(in_timesteps, in_timesteps), requires_grad=True)
        self.alpha = Parameter(torch.zeros(n_channels), requires_grad=True)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        """
        Args:
            x (FloatTensor): shape is [batch_size, n_nodes, n_nodes, in_timesteps]

        Returns:
            [FloatTensor]: shape is [batch_size, n_nodes, n_nodes]
        """
        # k_{n,t} = q_{n,t} = x_{i,n,t} \alpha_{i}
        k = q = torch.einsum('bint,i->bnt', x, self.alpha)  # -> [batch_size, n_nodes, in_timesteps]
        att = torch.softmax(k @ self.W @ q.transpose(1, 2), dim=-1)  # -> [batch_size, n_nodes, n_nodes]
        return att * adj  # -> [batch_size, n_nodes, n_nodes]


class GACN(Module):
    def __init__(self, in_channels, out_channels, in_timesteps):
        super(GACN, self).__init__()
        self.gatt = GAttention(n_channels=in_channels, in_timesteps=in_timesteps)
        self.W = Parameter(torch.zeros(out_channels, in_channels), requires_grad=True)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        """
        Args:
            x (FloatTensor): shape is [batch_size, in_channels, n_nodes, in_timesteps]

        Returns:
            [FloatTensor]: shape is [batch_size, out_channels, n_nodes, in_timesteps]
        """
        # [batch_size, n_nodes, n_nodes] @ [in_timesteps, batch_size, n_nodes, in_channels] @ [in_channels, out_channels]
        # -> [in_timesteps, batch_size, in_timesteps, out_channels]
        x_out = self.gatt(x, adj) @ x.permute(3, 0, 2, 1) @ self.W.T
        return x_out.permute(1, 3, 2, 0)  # -> [batch_size, out_channels, n_nodes, in_timesteps]


class TAttention(Module):
    def __init__(self, n_channels, n_nodes):
        super(TAttention, self).__init__()
        self.W1 = Parameter(torch.zeros(10, n_nodes), requires_grad=True)
        self.W2 = Parameter(torch.zeros(10, n_nodes), requires_grad=True)
        self.alpha = Parameter(torch.zeros(n_channels), requires_grad=True)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (FloatTensor): shape is [batch_size, in_channels, n_nodes, in_timesteps]

        Returns:
            [FloatTensor]: shape is [batch_size, in_channels, n_nodes, in_timesteps]
        """
        # k_{t,n} = q_{t,n} = x_{i,n,t} \alpha_{i}
        k = q = torch.einsum('bint,i->btn', x, self.alpha)  # -> [batch_size, in_timesteps, n_nodes]
        # -> [batch_size, in_timesteps, in_timesteps]
        att = torch.softmax((k @ self.W1.T) @ (q @ self.W2.T).transpose(1, 2), dim=-1)
        # y_{c,n,t} = a_{t,i} x_{c,n,i}
        return torch.einsum('bti,bcni->bcnt', att, x)  # -> [batch_size, in_channels, n_nodes, in_timesteps]


class Chomp(Module):
    def __init__(self, chomp_size):
        super(Chomp, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor):
        return x[..., :-self.chomp_size]


class TACN(Module):
    def __init__(self, in_channels, out_channels, dilations, n_nodes):
        super(TACN, self).__init__()
        channels = [in_channels] + [out_channels] * len(dilations)
        seq = [TAttention(n_channels=in_channels, n_nodes=n_nodes)]
        for i, dilation in enumerate(dilations):
            seq += [
                Conv2d(channels[i], channels[i + 1], [1, 2], padding=[0, dilation], dilation=[1, dilation]),
                Chomp(dilation)
            ]
        self.seq = Sequential(*seq)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (FloatTensor): [batch_size, in_channels, n_nodes, in_timesteps]

        Returns:
            [type]: [batch_size, out_channels, n_nodes, in_timesteps]
        """
        return self.seq(x)  # -> [batch_size, out_channels, n_nodes, in_timesteps]


class CAttention(Module):
    def __init__(self, n_nodes, in_timesteps):
        super(CAttention, self).__init__()
        self.W = Parameter(torch.zeros(in_timesteps, in_timesteps), requires_grad=True)
        self.alpha = Parameter(torch.zeros(n_nodes), requires_grad=True)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (FloatTensor): shape is [batch_size, in_channels, n_nodes, in_timesteps]

        Returns:
            FloatTensor: shape is [batch_size, in_channels, n_nodes, in_timesteps]
        """
        # k_{c,t} = q_{c,t} = x_{c,i,t} \alpha_{i}
        k = q = torch.einsum('bcit,i->bct', x, self.alpha)  # -> [batch_size, in_channels, in_timesteps]
        att = torch.softmax(k @ self.W @ q.transpose(1, 2), dim=-1)  # [batch_size, in_channels, in_channels]
        # y_{c,n,t} = a_{c,i} x_{i,n,t}
        return torch.einsum('bci,bint->bcnt', att, x)  # -> [batch_size, in_channels, n_nodes, in_timesteps]


class CACN(Module):
    def __init__(self, in_channels, out_channels, in_timesteps, n_nodes):
        super(CACN, self).__init__()
        self.catt = CAttention(n_nodes, in_timesteps)
        self.conv = Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (FloatTensor): shape is [batch_size, in_channels, n_nodes, in_timesteps]

        Returns:
            FloatTensor: shape is [batch_size, out_channels, n_nodes, in_timesteps]
        """
        out = self.catt(x)  # -> [batch_size, in_channels, n_nodes, in_timesteps]
        return self.conv(out)  # -> [batch_size, out_channels, n_nodes, in_timesteps]


class TGACN(Module):
    def __init__(self, in_channels, out_channels, in_timesteps, tcn_dilations, n_nodes):
        super(TGACN, self).__init__()
        self.ln = LayerNorm([in_timesteps])
        self.cacn = CACN(in_channels, out_channels // 3, in_timesteps, n_nodes)
        self.tacn = TACN(in_channels, out_channels // 3, tcn_dilations, n_nodes)
        self.gacn = GACN(in_channels, out_channels // 3, in_timesteps)
        self.res = Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        """
        Args:
            x (FloatTensor): shape is [batch_size, in_channels, n_nodes, in_timesteps]

        Returns:
            [FloatTensor]: shape is [batch_size, out_channels, n_nodes, out_timesteps]
        """
        out = self.ln(x)  # -> [batch_size, in_channels, n_nodes, in_timesteps]
        # -> [batch_size, out_channels, n_nodes, in_timesteps]
        out = torch.cat([self.cacn(out),  self.tacn(out), self.gacn(out, adj)], dim=1)
        return torch.relu(out + self.res(x))  # -> [batch_size, out_channels, n_nodes, in_timesteps]


class TPC(Module):
    def __init__(self, blocks, **kwargs):
        super(TPC, self).__init__()
        self.seq = Sequential(*[
            TGACN(**block, **kwargs) for block in blocks
        ], LayerNorm([kwargs['in_timesteps']]))
        self.fc = Conv2d(kwargs['in_timesteps'], kwargs['out_timesteps'], [1, blocks[-1]['out_channels']])

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (FloatTensor): shape is [batch_size, in_channels, n_nodes, in_timesteps]

        Returns:
            [FloatTensor]: shape is [batch_size, n_nodes, out_timesteps]
        """
        out = self.seq(x)  # -> [batch_size, out_channels, n_nodes, in_timesteps]
        out = self.fc(out.transpose(1, 3))  # -> [batch_size, out_timesteps, n_nodes, 1]
        return out[..., 0].transpose(1, 2)  # -> [batch_size, n_nodes, out_timesteps]


class TE(Module):
    def __init__(self, n_components, n_nodes, n_timesteps):
        self.sizes = [n_components, n_nodes, n_timesteps]
        self.d_ebd = Embedding(7, n_components * n_nodes * n_timesteps)
        self.h_ebd = Embedding(24, n_components * n_nodes * n_timesteps)

    def forward(self, H: torch.Tensor, D: torch.Tensor):
        G = self.h_ebd(H) + self.d_ebd(D)  # [(batch_size * n_components * n_nodes * n_timesteps)]
        return G.view(len(G), *self.sizes)  # [batch_size, n_components, n_nodes, n_timesteps]


class MSGAT(Module):
    def __init__(self, components, **kwargs):
        super(MSGAT, self).__init__()
        n_nodes, out_timesteps = kwargs['n_nodes'], kwargs['out_timesteps']
        if kwargs['te']:
            self.te = TE(len(components), n_nodes, out_timesteps)
        else:
            self.W = Parameter(torch.zeros(len(components), n_nodes, out_timesteps))
        self.tpcs = ModuleList([TPC(**component, **kwargs) for component in components])

    def forward(self, X: FloatTensor, H: LongTensor, D: LongTensor) -> FloatTensor:
        """
        Args:
            X (FloatTensor): shape is [batch_size, n_layers, in_channels, n_nodes, in_timesteps]
            H (LongTensor):  shape is [batch_size]
            D (LongTensor):  shape is [batch_size]

        Returns:
            [FloatTensor]: shape is [batch_size, n_nodes, out_timesteps]
        """
        if self.te:
            G = self.te(H, D)
            return sum((f(x) * g for f, x, g in zip(self.tpcs, X.unbind(1), G.unbind(1))))
        else:
            return sum((f(x) * w for f, x, w in zip(self.tpcs, X.unbind(1), self.W.unbind(0))))


def msgat(n_components, in_channels, in_timesteps, out_timesteps, n_nodes, adj, te):
    layers = [{
        "blocks": [
            {
                'in_channels': in_channels,
                'out_channels': 72,
                'tcn_dilations': [1, 2]
            },
            {
                'in_channels': 72,
                'out_channels': 72,
                'tcn_dilations': [2, 4]
            },
        ]
    }] * n_components
    net = MSGAT(components=layers, adj=adj, n_nodes=n_nodes,
                in_timesteps=in_timesteps, out_timesteps=out_timesteps, te=te)
    return net
