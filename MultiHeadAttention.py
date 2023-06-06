import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, n_heads, dropout):
    super().__init__()
    self.head_dim = d_model // n_heads
    self.n_heads = n_heads
    self.d_model = d_model
    # self.scale = d_model ** 0.5
    self.scale = None

    self.keys = nn.Linear(d_model, d_model)
    self.queries = nn.Linear(d_model, d_model)
    self.values = nn.Linear(d_model, d_model)
    self.projection = nn.Linear(d_model, d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, q, k, v, mask=None):
    N = q.size(0)         # batch_size
    Q = self.queries(q)   # shape: [N, query_len, d_model]
    K = self.keys(k)      # shape: [N, key_len, d_model]
    V = self.values(v)    # shape: [N, value_len, d_model]

    Q = Q.view(N, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # shape: [N, n_heads, query_len, head_dim]
    K = K.view(N, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # shape: [N, n_heads, key_len, head_dim]
    V = V.view(N, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # shape: [N, n_heads, value_len, head_dim]

    B, L, H, E = Q.shape
    scale =  1. / math.sqrt(E)

    energy = (Q @ K.permute(0, 1, 3, 2)) / 1.

    attention = energy.softmax(-1)          # shape: [N, n_heads, query_len, key_len]
    x = self.dropout(attention) @ V         # shape: [N, n_heads, query_len, key_len]
    x = x.permute(0, 2, 1, 3).contiguous()  # shape: [N, query_len, n_heads, head_dim]
    x = x.view(N, -1, self.d_model)          # shape: [N, query_len, d_model]
    x = self.projection(x)

    return x