import torch.nn as nn

from EncoderLayer import EncoderLayer

class Encoder(nn.Module):
  def __init__(self, d_model, n_blocks, n_heads, ff_hid_dim, max_length, dropout, device):
    super().__init__()
    self.device = device
    self.scale = d_model ** 0.5
    # self.tok_emb = nn.Embedding(input_size, d_model)
    # self.pos_emb = nn.Embedding(max_length, d_model)
    self.blocks = nn.ModuleList([EncoderLayer(
            d_model, 
            n_heads, 
            ff_hid_dim,
            dropout)] * n_blocks)

    self.dropout = nn.Dropout(dropout)
    self.norm = nn.LayerNorm(d_model)

  def forward(self, x, mask=None):
    # N, seq_len = src.shape
    # positions = torch.arange(0, seq_len).expand(N, seq_len.to(self.device)
    # pos_embeddings = self.pos_emb(positions)
    # tok_embeddings = self.tok_emb(src) * self.scale
    # out = self.dropout(pos_embeddings + tok_embeddings)

    for block in self.blocks:
      out = block(x, mask)
      out = self.norm(out)

    return out