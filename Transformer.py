import torch.nn as nn
import torch

from DataEmbedding import DataEmbedding
from Encoder import Encoder


class Transformer(nn.Module):
    def __init__(self,
                 input_size,
                 embed_dim,
                 n_blocks,
                 n_heads,
                 ff_hid_dim,
                 max_length,
                 dropout,
                 device):
        super().__init__()

        self.enc_in = input_size+1

        # linear mapper
        self.linear_mapper = nn.Linear(input_size, self.enc_in)

        # adding classification token
        self.class_token = nn.Parameter(torch.rand(1, self.enc_in))

        #Embedding
        self.embedding = DataEmbedding(self.enc_in, embed_dim, dropout)

        self.encoder = Encoder(embed_dim,
                               n_blocks,
                               n_heads,
                               ff_hid_dim,
                               max_length,
                               dropout,
                               device)

        self.projection = nn.Linear(embed_dim, input_size, bias=True)

        # Classification MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 1),
            nn.Sigmoid()
        )

    def src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).bool().to(self.device) & trg_pad_mask
        return trg_mask.to(self.device)

    def forward(self, src):
        # src_mask = self.src_mask(src)
        # trg_mask = self.trg_mask(trg)
        x = self.linear_mapper(src)
        x = torch.stack([torch.vstack((self.class_token, x[i])) for i in range(len(x))])        # apply classification to the first position
        enc_out = self.embedding(x)
        enc_out = self.encoder(enc_out)
        enc_out = self.projection(enc_out)

        # seperate class token and recostruction, send the class token through the MLP
        class_token = enc_out[:, :1, :]
        enc_out = enc_out[:, 1:, :]
        class_token = self.mlp(class_token)

        return enc_out, class_token