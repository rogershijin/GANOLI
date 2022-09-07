import torch
from transformers import DistilBertModel
from torch.nn import Linear, ReLU

class MLP(torch.nn.Module):
    
    def __init__(self, input_dim=116490, output_dim=13431, hidden_dim=768): # todo: cache dir
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.encoder = Linear(self.input_dim, self.hidden_dim)
        self.relu = ReLU()
        self.decoder = Linear(self.hidden_dim, self.output_dim)
        
    def forward(self, batch):
        out = self.encoder(batch)
        out = self.relu(out)
        out = self.decoder(out)
        out = self.relu(out)
        return out
