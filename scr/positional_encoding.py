import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
  def __init__(self, embedding_dim, max_len=5000):
    super().__init__()

    self.embedding_dim = embedding_dim
    self.max_len = max_len

    #position encoding matrix
    pe = torch.zeros(max_len, embedding_dim)


    position = torch.arange(0, max_len).unsqueeze(1)

    #frequence
    div_term = torch.exp( torch.arange(0, embedding_dim, 2)*(-math.log(10000.0) / embedding_dim))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    pe=pe.unsqueeze(0)

    #save without train
    self.register_buffer("pe", pe)

  def forward(self, x):
    seq_len = x.size(1)

    x = x + self.pe[:, : seq_len, :]

    return x

