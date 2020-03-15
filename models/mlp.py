import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 128), activation='tanh', is_dropout=False):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'leaky':
            self.activation = torch.nn.LeakyReLU()

        self.affine_layers = nn.ModuleList()
        self.out_dim = hidden_dims[-1]
        last_dim = input_dim
        for idx, nh in enumerate(hidden_dims):
            self.affine_layers.append(nn.Linear(last_dim, nh))  
            if idx == 0 and is_dropout:
                self.affine_layers.append(nn.Dropout(p=0.5))
            last_dim = nh

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        return x


class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 128), activation='tanh', is_dropout=False):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'leaky':
            self.activation = torch.nn.LeakyReLU()

        self.affine_layers = nn.ModuleList()
        self.out_dim = hidden_dims[-1]
        last_dim = input_dim
        for idx, nh in enumerate(hidden_dims):
            if idx == 0:
                self.affine_layers.append(nn.Linear(input_dim, nh))  
            else:
                self.affine_layers.append(nn.Linear(last_dim + 2, nh))  
            last_dim = nh
        self.dropout = is_dropout
        self.dp = nn.Dropout(p=0.5)

    def forward(self, _input):
        prob = _input[:, -2:]
        for idx, affine in enumerate(self.affine_layers):
            if idx == 0:
                x = self.activation(affine(_input))
            else:
                x = self.activation(affine(torch.cat([x, prob], dim=-1)))
            if idx == 1:
                x = self.dp(x)
        return x