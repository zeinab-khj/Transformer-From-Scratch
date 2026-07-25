import torch
import torch.nn as nn
from src.transformer import Transformer



src_vocab_size = 100
tgt_vocab_size = 100

embedding_dim = 32
num_heads = 4

num_encoder_layers = 2
num_decoder_layers = 2


model = Transformer(
    scr_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    embedding_dim=embedding_dim,
    num_head=num_heads,
    num_encoder_layer=num_encoder_layers,
    num_decoder_layer=num_decoder_layers
)



batch_size = 2

src_length = 5
tgt_length = 6



src = torch.randint(
    0,
    src_vocab_size,
    (batch_size, src_length)
)


tgt = torch.randint(
    0,
    tgt_vocab_size,
    (batch_size, tgt_length)
)


output = model(
    src,
    tgt
)


print("Output shape:")
print(output.shape)
