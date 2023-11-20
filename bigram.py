import torch
import torch.nn as nn
from torch.nn import functional as F
import requests

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2

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

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) # (B, T, C)
        self.position_embedding_table = nn.Embedding(block_size, n_embed) # A second embedding table that is for the position. We have one of these because before we only considered the character
        self.lm_head = nn.Linear(n_embed, vocab_size) # (B, T, vocab_size) Linear layer that'll map the model outputs to. We need this because our number of embeddings is going to vary

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C) --> Integers from 0 to T-1
        # This addition of the two tensors will result in the positional embedding getting brodcased across the batches
        ## In the context of neural language models, this addition of token embeddings and position embeddings is a common practice to enhance 
        ## the model's ability to capture sequential information. It helps the model distinguish between tokens at different positions in the sequence, 
        ## especially when the model needs to understand the order of the tokens in the input data.
        x = tok_emb + pos_emb #(B, T, C) --> What is being fed in is a combination of the time (i.e. position in token) and the Channel
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
## We need to move the model to the proper device! This ensures that all the calculations happen in the GPU/CPU respectively
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    ## Shows the losses as certain iteration counts are hit
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
## Note that we're generating the context on the device being used (GPU/CPU) - so we need to set "device=device"
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
