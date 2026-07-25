import torch
import torch.nn as nn
from scr.positional_encoding import PositionalEncoding
from scr.encoder import TransformerEncoder
from scr.decoder import TransformerDecoder



class Transformer(nn.Module):
  def __init__(
      self,
      scr_vocab_size,
      tgt_vocab_size,
      embedding_dim,
      num_head,
      num_encoder_layer,
      num_decoder_layer,
      max_length=5000,
      expansion=4,
      dropout=0.1
  ):
    super().__init__()

    self.scr_embedding = nn.Embedding(scr_vocab_size, embedding_dim)
    self.scr_pos_encoding = PositionalEncoding(embedding_dim, max_length)

    self.tgt_embedding = nn.Embedding(tgt_vocab_size, embedding_dim)
    self.tgt_pos_encoding = PositionalEncoding(embedding_dim, max_length)

    self.encoder = TransformerEncoder(
        vocab_size=scr_vocab_size,
        embedding_dim=embedding_dim,
        num_head=num_head,
        num_layer=num_encoder_layer,
        expansion=expansion,
        dropout=dropout

    )

    self.decoder = TransformerDecoder(
        embedding_dim=embedding_dim,
        num_head=num_head,
        num_layer=num_decoder_layer,
        expansion=expansion,
        dropout=dropout
    )

    self.fc_out = nn.Linear(embedding_dim, tgt_vocab_size)

  def forward(self, scr, tgt, scr_mask=None, tgt_mask=None):

    # Encoder input
    scr_embedding = self.scr_embedding(scr)
    scr_pos_embedding = self.scr_pos_encoding(scr_embedding)

    encoder_output = self.encoder(scr_pos_embedding)

    
    ## Decoder 
    tgt_embedding = self.tgt_embedding(tgt)
    tgt_pos_encoding = self.tgt_pos_encoding(tgt_embedding)

    decoder_output = self.decoder(tgt_pos_encoding, encoder_output, tgt_mask)

    #convert features to vocabulary scores
    output = self.fc_out(decoder_output)

    return output
