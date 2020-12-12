from models.msgcn import MSGCN
from torch.nn import init


def make_msgat(in_timesteps, out_timesteps, n_nodes, adj, device):
    layers = [{
        "blocks": [
            {
                'in_channels': 3,
                'out_channels': 72,
                'tcn_dilations': [1, 2]
            },
            {
                'in_channels': 72,
                'out_channels': 72,
                'tcn_dilations': [2, 4]
            },
        ]
    }] * 5
    model = MSGCN(layers=layers, adj=adj, n_nodes=n_nodes,
                  in_timesteps=in_timesteps, out_timesteps=out_timesteps).to(device)
    for param in model.parameters():
        if param.ndim >= 2:
            init.xavier_normal_(param)
        else:
            init.uniform_(param)
    return model
