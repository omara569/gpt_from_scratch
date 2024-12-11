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


class Feed_Forward(nn.Module):
    def __init__(self):
        super.__init__()
        self.net = nn.Sequential(
            nn.Linear(num_embeddings, num_embeddings),
            nn.ReLU(),
            nn.Linear(num_embeddings, num_tokens),
            nn.Dropout(.2)
        )

    def forward(self, input_data: torch.tensor) -> torch.tensor:
        return self.net(input_data)
    
