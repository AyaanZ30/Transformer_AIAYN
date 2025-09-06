import torch 
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init_(self, dim_model : int, vocab_size : int) -> None:
        super().__init__()
        self.dim_model = dim_model
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, dim_model)    # Embedding matrix of size (vocab x 512)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.dim_model)    
    
class PositionalEncoding(nn.Module):
    def __init__(self, dim_model : int, seq_length : int, dropout_p : float) -> None:   # seq length : to create one PE vector for each word in sequences
        super().__init__()
        self.dim_model = dim_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout_p)

        # create a matrix of shape (seq_length, dim_model) => 
        #"I love my cat" (seq_length = 4) ==>
        #I : [...512 embeddings], 
        #love : [...512 embeddings],
        #my : [...512 embeddings], 
        #cat : [...512 embeddings],
        pe = torch.zeros(seq_length, dim_model)

        # create a vector of (seq_length, 1) size to store positions of words in the sequence
        position = torch.arange(0, seq_length, dtype = torch.float)
        position = position.unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000) / dim_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # (1, seq_length, dim_model)
        pe = pe.unsqueeze(0)   # Adding batch dimension (as pe will be computed for batches of sentences)
        self.register_buffer("entrenched_positional_encodings", pe)
    
    def forward(self, x):
        """
        Performs the forward pass by adding positional encodings to the input tensor.
        Ex : "Hello world" -> (for 'Hello')[23,65, .... (512 elements)] + (corresponding pe vector)[3.4, 2.5, ...(512 elements)] .... (same for 'world')
        ARGS:
        x (torch.Tensor): The input tensor representing the current sentence batch.
        Expected shape of x : [batch_size, seq_length, dim_model(= embedding_dim)].
        """
        current_sentence_pe = (self.pe[:, :x.shape[1], :]).requires_grad_(False)     # encondings are not learnable parameters (they are jsut computed once and used for every sentence)
        x = x + current_sentence_pe
        return self.dropout(x)
    
class AddandNorm(nn.Module):
    def __init__(self, epsilon : float = 10**(-6)) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(1))    # multiplicative additional parameter
        self.bias = nn.Parameter(torch.zeros(1))   # additive additional parameter
    
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = math.sqrt(x.variance(dim = -1, keepdim = True))
        return self.alpha * (x - mean) / (std + self.epsilon) * self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, dim_model : int, dim_ff : int, dropout_p : float) -> None:
        super().__init__()
        self.dim_model = dim_model
        self.dim_ff = dim_ff
        self.dropout_p = dropout_p

        self.linear_1 = nn.Linear(in_features = dim_model, out_features = dim_ff)   # W1 and B1 (bias by default)
        self.dropout = nn.Dropout(dropout_p)
        self.linear_2 = nn.Linear(in_features = dim_ff, out_features = dim_model)   # W2 and B2 (bias by default)
    
    def forward(self, x):
        """
        Feed forward neural network to process batches of sentences in dataset
        ARGS:
        x (torch.Tensor): The input tensor representing the current sentence batch.
        Expected shape of x : [batch_size, seq_length, dim_model(= embedding_dim)].
        """
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, dim_model : int, n_heads : int, dropout_p : float) -> None:
        super().__init__()
        self.dim_model = dim_model
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout_p)

        assert(dim_model % n_heads == 0), f"d_model ({dim_model}) not divisible by n_heads ({n_heads})  [{dim_model} / {n_heads} = {float(dim_model / n_heads)}]"

        self.dim_k = (self.dim_model // self.n_heads)   # dim of a single head

        self.weights_q = nn.Linear(self.dim_model, self.dim_model)
        self.weights_k = nn.Linear(self.dim_model, self.dim_model)
        self.weights_v = nn.Linear(self.dim_model, self.dim_model)

        # (d_model = n_heads(total heads) x dim_k(dim of each head))
        self.weights_o = nn.Linear((self.n_heads * self.dim_k), self.dim_model)
    
    @staticmethod
    def attention(query, key, value, mask, dropout_p : nn.Dropout):
        dim_k = query.shape[-1]
        
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(dim_k)
        if mask:
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        # attention scores obtained for each correlation (each word taken with every other word and itself) in the sequence
        attention_scores = attention_scores.softmax(dim = -1)     # shape : (batch, n_heads, seq_length, seq_length)
        if dropout_p:
            attention_scores = dropout_p(attention_scores)

        single_head_output = (attention_scores @ value)
        return single_head_output, attention_scores 

    
    def forward(self, q, k, v, mask):
        query = self.weights_q(q)      # q(batch, seq_length, dim_model) x w_q(dim_model, dim_model) => query(batch, seq_length, dim_model)
        key = self.weights_k(k)        # k(batch, seq_length, dim_model) x w_k(dim_model, dim_model) => key(batch, seq_length, dim_model)
        value = self.weights_v(v)     # v(batch, seq_length, dim_model) x w_v(dim_model, dim_model) => values(batch, seq_length, dim_model)

        # (batch, seq_len, dim_model) --> (batch, seq_length, n_heads, dim_k) --> (batch, n_heads, seq_length, dim_k)
        query_divided_into_heads = query.view(query.shape[0], query.shape[1], self.n_heads, self.dim_k).transpose(1, 2)    # break down the query matrix into 'n' heads of dimension k
        key_divided_into_heads = key.view(key.shape[0], key.shape[1], self.n_heads, self.dim_k).transpose(1, 2)    # break down the key matrix into 'n' heads of dimension k 
        value_divided_into_heads = value.view(value.shape[0], value.shape[1], self.n_heads, self.dim_k).transpose(1, 2)    # break down the value matrix into 'n' heads of dimension k  

        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query_divided_into_heads,
            key_divided_into_heads,
            value_divided_into_heads,
            mask,
            self.dropout
        )

        # (batch, n_heads, seq_len, dim_k) -- transpose --> (batch, seq_len, n_heads, dim_k) --> (batch, seq_len, dim_model)  [dim_model = n_heads * dim_k]
        # we are recombining the outputs from all attention heads into single tensor (concatenation) 
        # (now x contains combined information from all attention heads)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, (self.n_heads * self.dim_k)) 
        # x (batch, seq_len, dim_model) @ weights_o (dim_model, dim_model) => final MH output (batch, seq_len, dim_model)
        return self.weights_o(x)
    
class ResidualConnection(nn.Module):
    def __init__(self, dropout_p : float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.norm = AddandNorm()     # as input will go directly to the AddandNorm Block as well as through the sublayer between input and AddandNorm Block.

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block : MultiHeadAttentionBlock, feed_forward_block : FeedForwardBlock, dropout_p : float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block

        # 2 residual connections block (refer architecture image)
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout_p) for _ in range(2)])
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x : self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block(x))
        return x

class Encoder(nn.Module):
    def __init__(self, layers : nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers 
        self.norm = AddandNorm()
    
    def forward(self, x, mask):
        for current_layer in self.layers:
            x = current_layer(x, mask)
        return self.norm(x)


