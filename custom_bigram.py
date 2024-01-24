import torch
import torch.nn as nn 
from torch.nn import functional as F 
import requests 


# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3

## This is so that we can run the Program on a GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

eval_iters = 200
n_embed = 32 ## Number of embedding dimensions. We want to be flexible here for model performance
# ------------

torch.manual_seed(1337)

## Getting text data - using my code to do this part instead of his!
# Define a function to get text data from a url
def get_text(url: str) -> str:
    response = requests.get(url)
    if response.status_code == 200:
        byte_data = response.content
        text = byte_data.decode('utf-8')
        return text
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
text = get_text("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt") 

## Setting up encoder and decoder
# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:] 


# data loading (batches)
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    ## We need to ensure we move the data to the GPU or CPU accordingly
    x, y = x.to(device), y.to(device)
    return x, y

## New Function - this one will average out the loss over multiple batches. This will result in a much less noisy loss!
## @torch.no_grad() disables gradient calculation for the block of code defined below it. This saves computation time
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval() ## Sets the model in evaluation mode. This is 
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out