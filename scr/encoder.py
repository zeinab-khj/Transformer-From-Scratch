import torch
import torch.nn as nn

class TransformerEncoderLayer(nn.Module):
  def __init__(self, embedding_dim, num_head, expansion=4, dropout=0.1):
    super().__init__()
    self.emdebbing_dim = embedding_dim
    self.num_head = num_head

    self.attention = Multi_Head_Attention(embedding_dim, num_head)
    self.norm1 = nn.LayerNorm(embedding_dim)
    self.ff = FeedForward(embedding_dim, expansion)
    self.norm2 = nn.LayerNorm(embedding_dim)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):

    attention_output = self.attention(x)

    x = self.norm1(x + self.dropout(attention_output))

    ff_output = self.ff(x)

    x = self.norm2(x + self.dropout(ff_output))

    return x




class TransformerEncoder(nn.Module):
  def __init__(self, vocab_size, embedding_dim, num_head, num_layer,  expansion=4, dropout=0.1):
    super().__init__()
    self.vocab_size = vocab_size
    self.embedding_dim = embedding_dim
    self.num_head = num_head
    self.num_layer = num_layer
    self.expansion = expansion


    #transformer Block
    self.layers = nn.ModuleList(
        [
            TransformerEncoderLayer(embedding_dim, num_head, expansion, dropout)for _ in range(num_layer)
        ]
    )

    self.norm = nn.LayerNorm(embedding_dim)

  def forward(self, x):


    for layer in self.layers:
      x = layer(x)

    x = self.norm(x)

    return x
