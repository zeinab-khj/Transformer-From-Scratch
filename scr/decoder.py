import torch
import torch.nn as nn
from scr.attention import MultiHeadAttention
from scr.feed_forward import FeedForward

class TransformerDecoderLayer(nn.Module):
  def __init__(self, embedding_dim, num_head, expansion=4, dropout=0.1):
    super().__init__()
    self.embedding_dim = embedding_dim
    self.num_head = num_head
    self.expansion = expansion

    self.self_attention = MultiHeadAttention(embedding_dim, num_head)

    self.cross_attention = MultiHeadAttention(embedding_dim, num_head)

    self.ff = FeedForward(embedding_dim, expansion)

    self.norm1 = nn.LayerNorm(embedding_dim)
    self.norm2 = nn.LayerNorm(embedding_dim)
    self.norm3 = nn.LayerNorm(embedding_dim)

    self.dropout = nn.Dropout(dropout)

  def forward(self, x, encoder_output, mask=None):

    # mask self attention
    self_attention = self.self_attention(x, x, x, mask)

    x = self.norm1(x + self.dropout(self_attention))

    # cross attention
    cross_attention = self.cross_attention(x, encoder_output, encoder_output)

    x = self.norm2(x + self.dropout(cross_attention))

    # feed forward
    ff = self.ff(x)

    output = self.norm3(x + self.dropout(ff))

    return output


class TransformerDecoder(nn.Module):
  def __init__(self, embedding_dim, num_head, num_layer, expansion=4, dropout=0.1):
    super().__init__()

    self.num_head = num_head
    self.num_layer = num_layer


    self.layers = nn.ModuleList(
        [
            TransformerDecoderLayer(embedding_dim, num_head, expansion, dropout) for _ in range(num_layer)
        ]
    )

  def forward(self, x, encoder_output, mask=None):

    for layer in self.layers:
      x = layer(x, encoder_output, mask)

    return x

