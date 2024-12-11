# Imports
import torch
import torch.nn as nn 
import torch.functional as F

num_embeddings = 32
num_block_layers = 6 # as in the paper
num_heads = 8 # number of heads for multi-headed attention
num_tokens_per_batch_stream = 8 # number of tokens per sequence
num_epochs = 5000
num_tokens = 800 # NOTE: Change this to be the actual number of unique tokens
dropout_probability = .2
masked_attention = True


class Feed_Forward(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        '''
            num_inputs contains the number of input values for the feed forward
        '''
        super.__init__()
        self.net = nn.Sequential(
            nn.Linear(num_inputs, num_outputs),
            nn.ReLU(),
            nn.Linear(num_outputs, num_outputs),
            nn.Dropout(dropout_probability)
        )

    def forward(self, input_data: torch.tensor) -> torch.tensor:
        return self.net(input_data)
    

class Attention(nn.Module):
    def __init__(self, dim_per_head, output_dim_per_head):
        '''
            Attention head
        '''
        super.__init__()
        self.dim_per_head = dim_per_head
        self.output_dim_per_head = output_dim_per_head
        self.q = nn.Linear(dim_per_head, output_dim_per_head)
        self.k = nn.Linear(dim_per_head, output_dim_per_head)
        self.v = nn.Linear(dim_per_head, output_dim_per_head)

        self.register_buffer('masked_weights', torch.tril(torch.ones(output_dim_per_head)))
        # self.dropout = nn.Dropout(dropout_probability)

    
    def forward(self, input_data):
        # NOTE: A lot needs to be fixed here
        query_vals = self.q(input_data)
        key_vals = self.k(input_data)
        value_vals = self.v(input_data)

        mult = (query_vals @ key_vals.T) / ((self.dim_per_head)**5)
        masked_weights = mult.masked_fill(self.tril[:self.output_dim_per_head,:self.output_dim_per_head] == 0, float('-inf'))
        mult = F.softmax(mult)
        return mult @ value_vals


class MultiAttention(nn.Module):
    def __init__(self):
        super.__init__()

    
    def forward(self, input_data):
        pass


class Block_Layer(nn.Module):
    def __init__(self):
        super.__init__()
    

    def forward(self, input_data):
        pass 


class Transformer_Model(nn.Module):
    def __init__(self):
        super.__init__()


    def forward(self, input_data):
        pass