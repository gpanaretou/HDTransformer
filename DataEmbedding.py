import torch
import torch.nn as nn

from TokenEmbedding import TokenEmbedding
from PositionalEmbedding import PositionalEmbedding

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.4):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)