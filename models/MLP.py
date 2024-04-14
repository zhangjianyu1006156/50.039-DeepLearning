import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_features, dropout_rate, hidden_units, output_features=3):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_features, hidden_units[0])
        self.dropout1 = nn.Dropout(dropout_rate)
        self.act1 = nn.ReLU()

        self.hidden_layers = nn.ModuleList()
        for i in range(1, len(hidden_units)):
            self.hidden_layers.append(nn.Linear(hidden_units[i-1], hidden_units[i]))
            self.hidden_layers.append(nn.Dropout(dropout_rate))
            self.hidden_layers.append(nn.ReLU())

        self.output = nn.Linear(hidden_units[-1], output_features)

    def forward(self, x):
        x = torch.flatten(x, 1)
        
        x = self.dropout1(self.act1(self.fc1(x)))
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                x = layer(x)
            else:
                x = layer(x)

        x = self.output(x)
        return x