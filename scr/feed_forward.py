import torch
import torch.nn as nn

class FeedForward(nn.Module):
  def __init__(self, embedding_dim, expansion):
    super().__init__()
    self.embedding_dim = embedding_dim
    self.expansion = expansion
    self.fc1 = nn.Linear(embedding_dim, embedding_dim * expansion)
    self.gelu = nn.GELU()
    self.fc2 = nn.Linear(embedding_dim * expansion, embedding_dim)

  def forward(self, x):
    x = self.fc1(x)
    x = self.gelu(x)
    x = self.fc2(x)

    return x
