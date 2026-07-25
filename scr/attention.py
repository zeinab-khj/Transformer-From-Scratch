import math
import torch
import torch.nn as nn

class self_Attention(nn.Module):
  def __init__(self, embedding_dim):
    super().__init__()
    self.Wq = nn.Linear(embedding_dim, embedding_dim)
    self.Wk = nn.Linear(embedding_dim, embedding_dim)
    self.Wv = nn.Linear(embedding_dim, embedding_dim)

  def forward(self, x):
    Q = self.Wq(x)
    K = self.Wk(x)
    V = self.Wv(x)

    scores = Q @ K.transpose(-2, -1)
    scores = scores / math.sqrt(Q.size(-1))
    weights = torch.softmax(scores, dim = -1)
    output = weights @ V
    return output, weights

class MultiHeadAttention(nn.Module):

    def __init__(self, embedding_dim, num_heads):

        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        self.head_dim = embedding_dim // num_heads


        self.query = nn.Linear( embedding_dim, embedding_dim)

        self.key = nn.Linear( embedding_dim, embedding_dim)

        self.value = nn.Linear(embedding_dim, embedding_dim)

        self.fc_out = nn.Linear( embedding_dim, embedding_dim)

    def forward(self, query, key, value, mask=None):
       batch_size = query.shape[0]
       query_len = query.shape[1]
       key_len = key.shape[1]
       value_len = value.shape[1]

       Q = self.query(query)
       K = self.key(key)
       V = self.value(value)

       Q = Q.view(batch_size, query_len, self.num_heads, self.head_dim)
       K = K.view(batch_size, key_len, self.num_heads, self.head_dim)
       V = V.view(batch_size, value_len, self.num_heads, self.head_dim)

       Q = Q.transpose(1, 2)
       K = K.transpose(1, 2)
       V = V.transpose(1, 2)


       scores = Q @ K.transpose(-2, -1)
       scores = scores / math.sqrt(Q.size(-1))

       if mask is not None:
         scores = scores.masked_fill( mask == 0, float("-inf") )

       weights = torch.softmax(scores, dim = -1)
       output = weights @ V

       output = output.transpose(1, 2)

       output = output.contiguous().view(batch_size, query_len, self.embedding_dim)

       output = self.fc_out(output)

       return output
