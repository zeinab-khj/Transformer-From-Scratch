import torch
import torch.nn as nn
from torch.optim import Adam

from src.transformer import Transformer


# =========================
# Hyperparameters
# =========================

src_vocab_size = 20
tgt_vocab_size = 20

embedding_dim = 32
num_heads = 4

num_encoder_layers = 2
num_decoder_layers = 2

batch_size = 4

src_len = 5
tgt_len = 6

epochs = 200


# =========================
# Device
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# =========================
# Model
# =========================

model = Transformer(
    scr_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    embedding_dim=embedding_dim,
    num_head=num_heads,
    num_encoder_layer=num_encoder_layers,
    num_decoder_layer=num_decoder_layers
)


model = model.to(device)


# =========================
# Toy Dataset
# =========================

# Encoder input
src = torch.randint(
    0,
    src_vocab_size,
    (batch_size, src_len)
)


# Target sentence
target = torch.randint(
    0,
    tgt_vocab_size,
    (batch_size, tgt_len)
)


src = src.to(device)
target = target.to(device)


# Decoder input
decoder_input = target[:, :-1]


target_output = target[:, 1:]


# =========================
# Loss & Optimizer
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = Adam(
    model.parameters(),
    lr=0.001
)


# =========================
# Training Loop
# =========================

model.train()


for epoch in range(epochs):

    optimizer.zero_grad()


    output = model(
        src,
        decoder_input
    )


    # output:
    # (batch, seq_len, vocab_size)

    loss = criterion(
        output.reshape(-1, tgt_vocab_size),
        target_output.reshape(-1)
    )


    loss.backward()


    optimizer.step()


    if epoch % 20 == 0:
        print(
            f"Epoch {epoch} | Loss: {loss.item():.4f}"
        )
