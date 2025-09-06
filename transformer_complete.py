import torch
import torch.nn as nn  
import math
from transformer_encoder import Encoder, EncoderBlock, InputEmbeddings, PositionalEncoding
from transformer_decoder import Decoder, DecoderBlock, ProjectionLayer

class Transformer(nn.Module):
    def __init__(self, 
    encoder : Encoder, 
    decoder : Decoder, 
    src_embed : InputEmbeddings,
    target_emb : InputEmbeddings,
    src_pos : PositionalEncoding,
    target_pos : PositionalEncoding,
    projection_layer : ProjectionLayer, 
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.target_emb = target_emb
        self.src_pos = src_pos
        self.target_pos = target_pos
        self.projection_layer = projection_layer
    
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, target_sentence, target_mask):
        target_sentence = self.target_emb(target_sentence)
        target_sentence = self.target_pos(target_sentence)
        return self.decoder(target_sentence, encoder_output, src_mask, target_mask)
    
    def project(self, decoder_output):
        final_output = self.projection_layer(decoder_output)
        return final_output


def build_transformer():
    
