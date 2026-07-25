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
# Dataset
# =========================

dataset = ToyTranslationDataset(
    num_samples=100
)


dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
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

    total_loss = 0


    for batch in dataloader:


        # -------------------------
        # Get data
        # -------------------------

        src = batch["src"].to(device)

        tgt = batch["tgt"].to(device)



        # -------------------------
        # Decoder shifting
        # -------------------------

        decoder_input = tgt[:, :-1]

        target_output = tgt[:, 1:]



        # -------------------------
        # Forward
        # -------------------------

        output = model(
            src,
            decoder_input
        )


        # output:
        # (batch, seq_len, vocab_size)



        # -------------------------
        # Loss
        # -------------------------

        loss = criterion(
            output.reshape(-1, tgt_vocab_size),
            target_output.reshape(-1)
        )



        # -------------------------
        # Backprop
        # -------------------------

        optimizer.zero_grad()


        loss.backward()


        optimizer.step()



        total_loss += loss.item()



    avg_loss = total_loss / len(dataloader)


    if epoch % 20 == 0:

        print(
            f"Epoch {epoch} | Loss: {avg_loss:.4f}"
        )
