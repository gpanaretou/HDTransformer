import torch.nn as nn

from MultiHeadAttention import MultiHeadAttention

class EncoderLayer(nn.Module):
  def __init__(self, d_model, n_heads, ff_hid_dim, dropout):
    super().__init__()
    self.attention = MultiHeadAttention(d_model, n_heads, dropout)
    self.norm1 = nn.LayerNorm(d_model)
    self.mlp = nn.Sequential(
        nn.Linear(d_model, ff_hid_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(ff_hid_dim, d_model)
    )
    self.dropout = nn.Dropout(dropout)
    self.norm2 = nn.LayerNorm(d_model)

  def forward(self, src, mask=None):
    attention = self.attention(src, src, src, mask)
    x = self.norm1(attention + self.dropout(src))
    out = self.mlp(x)
    out = self.norm2(out + self.dropout(x))
    return out