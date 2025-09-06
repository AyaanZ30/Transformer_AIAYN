import torch
import torch.nn as nn  
import math

from torch.nn.modules import transformer
from transformer_encoder import Encoder, EncoderBlock, FeedForwardBlock, InputEmbeddings, MultiHeadAttentionBlock, PositionalEncoding
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
    ):
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

# initiializng transformer with all hyperparams used in the original AIAYN paper
def build_transformer(
    src_vocab_size : int, 
    target_vocab_size : int, 
    src_seq_length : int, 
    target_seq_length : int,
    dim_model : int = 512,
    N_blocks : int = 6,     # we will be using 6 encoder + 6 decoder blocks
    n_heads : int = 8,
    dropout : float = 1e-1,
    dim_ff : int = 2048,
    ) -> Transformer:
    # creating the embedding layers 
    src_emb = InputEmbeddings(dim_model, src_vocab_size)
    target_emb = InputEmbeddings(dim_model, target_vocab_size)

    # creating the positional encodings
    src_pos = PositionalEncoding(dim_model, src_seq_length, dropout)
    target_pos = PositionalEncoding(dim_model, target_seq_length, dropout)

    # creating encoder blocks 
    encoder_blocks = []
    for i in range(N_blocks):
        encoder_self_attention_block = MultiHeadAttentionBlock(dim_model, n_heads, dropout)
        feed_forward_block = FeedForwardBlock(dim_model, dim_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # creating decoder blocks
    decoder_blocks = []
    for i in range(N_blocks):
        decoder_self_attention_block = MultiHeadAttentionBlock(dim_model, n_heads, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(dim_model, n_heads, dropout)
        feed_forward_block = FeedForwardBlock(dim_model, dim_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # creating encoder (consisting of all encoder blocks) & decoder (consisting of all decoder blocks)
    encoder = Encoder(layers = nn.ModuleList(encoder_blocks))
    decoder = Decoder(layers = nn.ModuleList(decoder_blocks))

    # projection layer 
    projection_layer = ProjectionLayer(dim_model, target_vocab_size)

    # stacking everything together
    transformer = Transformer(
        encoder, 
        decoder, 
        src_emb,
        target_emb,
        src_pos,
        target_pos,
        projection_layer
    )
    for parameter in transformer.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)
    return transformer



    


    
       

