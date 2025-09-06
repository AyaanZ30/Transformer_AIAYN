import torch 
import torch.nn as nn 
import math
from transformer_encoder import AddandNorm, FeedForwardBlock, MultiHeadAttentionBlock, ResidualConnection 

class DecoderBlock(nn.Module):
    def __init__(self, 
    self_attention_block : MultiHeadAttentionBlock, 
    cross_attention_block : MultiHeadAttentionBlock, 
    feed_forward_block : FeedForwardBlock,
    dropout_p : float,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block

        # single decoder block has 3 residual connections 
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout_p) for _ in range(3)])

    # src_mask(English) : source(input) language mask & target_mask(Italian) : decoding language mask
    def forward(self, x, encoder_output, src_mask, target_mask):
        # primary input to the decoder (Italian translated text..) and hence (q, k, v and mask for self attention) -> (x, x, x, target_mask)
        x = self.residual_connections[0](x, lambda x : self.self_attention_block(x, x, x, target_mask))
        # key and value come from final encoder block output and query from decoder
        x = self.residual_connections[1](x, lambda x : self.cross_attention_block(x, encoder_output, encoder_output, src_mask))   
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, layers : nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = AddandNorm()
    
    def forward(self, x, encoder_output, src_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, target_mask)
        return self.norm(x)

# Linear layer for projection of dim_model -> vocab_size (classification head)
class ProjectionLayer(nn.Module):
    def __init__(self, dim_model : int, vocab_size : int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_model, vocab_size)
    
    def forward(self, x):
        # (batch, seq_len, dim_model) --> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1)
