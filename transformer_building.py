# Imports
import torch
import torch.nn as nn 
import torch.nn.functional as F

num_embeddings = 32
num_block_layers = 6 # as in the paper
num_heads = 8 # number of heads for multi-headed attention
num_tokens_per_batch_stream = 8 # number of tokens per sequence
num_epochs = 5000
num_tokens = 131 # NOTE: Change this to be the actual number of unique tokens
dropout_probability = .2
masked_attention = True


class Feed_Forward(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        '''
            num_inputs contains the number of input values for the feed forward
        '''
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_inputs, num_outputs),
            nn.ReLU(),
            nn.Linear(num_outputs, num_outputs),
            nn.Dropout(dropout_probability)
        )

    def forward(self, input_data: torch.tensor) -> torch.tensor:
        return self.net(input_data)
    

class Attention(nn.Module):
    def __init__(self, dim_per_head):
        '''
            Attention head
        '''
        super().__init__()
        self.dim_per_head = dim_per_head
        self.q = nn.Linear(num_embeddings, dim_per_head)
        self.k = nn.Linear(num_embeddings, dim_per_head)
        self.v = nn.Linear(num_embeddings, dim_per_head)

        self.register_buffer('masked_weights', torch.tril(torch.ones(num_tokens_per_batch_stream, num_tokens_per_batch_stream)))

    
    def forward(self, input_data):
        query_vals = self.q(input_data)
        key_vals = self.k(input_data)
        value_vals = self.v(input_data)

        mult = (query_vals @ key_vals.transpose(1,2)) / ((self.dim_per_head)**.5) # attention weighting
        mult = mult.masked_fill(self.masked_weights == 0, float('-inf'))
        mult = F.softmax(mult, 2)
        return mult @ value_vals


class MultiAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # Multi-headed attention is taking multiple attention heads and having them function in parallel, concatenating their results at the end. Each attention performs operations on a subset of the embedding dimensionality
        self.attention_head_list = nn.ModuleList([Attention(num_embeddings//num_heads) for _ in range(num_heads)])
        self.lin_layer = nn.Linear(num_embeddings, num_embeddings)

    
    def forward(self, input_data):
        return self.lin_layer(torch.concat([attention_head(input_data) for attention_head in self.attention_head_list], dim=2))


class Block_Layer(nn.Module):
    def __init__(self):
        super().__init__()
        # repeated sets of multi-headed attention
        self.multi_headed_list = nn.Sequential(*[MultiAttention() for i in range(num_block_layers)])
    

    def forward(self, input_data):
        return self.multi_headed_list(input_data)


class Transformer_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.positional_embedding_table = nn.Embedding(num_tokens_per_batch_stream, num_embeddings)
        self.token_embedding_table = nn.Embedding(num_tokens, num_embeddings)
        self.feed_forward = Feed_Forward(num_embeddings, num_embeddings)
        self.lin_layer = nn.Linear(num_embeddings, num_tokens)
        self.blocks = nn.Sequential(*[Block_Layer() for i in range(num_block_layers)])
        self.layer_norm = nn.LayerNorm(num_embeddings)
        self.layer_norm_2 = nn.LayerNorm(num_embeddings)



    def forward(self, input_data, expected_result):
        # first we obtain the positional encoding values
        positional_embeddings = self.positional_embedding_table(torch.arange(num_tokens_per_batch_stream))
        token_embeddings = self.token_embedding_table(input_data)

        #positional encoding
        x = token_embeddings+positional_embeddings
        x_2 = self.blocks(x)
        x = x + x_2
        x =  self.layer_norm(x)
        x_2 = self.feed_forward(x)
        x = x+x_2 
        x = self.layer_norm_2(x)
        x = self.lin_layer(x)
        return x 
    




