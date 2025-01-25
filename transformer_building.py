# Imports
import torch
import torch.nn as nn 
import torch.nn.functional as F

num_embeddings = 32
num_block_layers = 6 # as in the paper
num_heads = 8 # number of heads for multi-headed attention
num_tokens = 118 # NOTE: Change this to be the actual number of unique tokens
dropout_probability = .2
masked_attention = True
block_size = 10 # number of tokens per batch
batch_size = 4 # number of batches


device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
    

# class Attention(nn.Module):
#     def __init__(self, dim_per_head):
#         '''
#             Attention head
#         '''
#         super().__init__()
#         self.dim_per_head = dim_per_head
#         self.q = nn.Linear(num_embeddings, dim_per_head)
#         self.k = nn.Linear(num_embeddings, dim_per_head)
#         self.v = nn.Linear(num_embeddings, dim_per_head)

#         self.register_buffer('masked_weights', torch.tril(torch.ones(block_size, block_size)))

    
#     def forward(self, input_data):
#         query_vals = self.q(input_data)
#         key_vals = self.k(input_data)
#         value_vals = self.v(input_data)

#         batch_count, token_count, embedding_count = input_data.shape
#         mult = (query_vals @ key_vals.transpose(1,2)) / ((self.dim_per_head)**.5) # attention weighting
#         mult = mult.masked_fill(self.masked_weights[:token_count, :token_count] == 0, float('-inf'))
#         mult = F.softmax(mult, 2)
#         return mult @ value_vals


# class MultiAttention(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Multi-headed attention is taking multiple attention heads and having them function in parallel, concatenating their results at the end. Each attention performs operations on a subset of the embedding dimensionality
#         self.attention_head_list = nn.ModuleList([Attention(num_embeddings//num_heads) for _ in range(num_heads)])
#         self.lin_layer = nn.Linear(num_embeddings, num_embeddings)

    
#     def forward(self, input_data):
#         return self.lin_layer(torch.concat([attention_head(input_data) for attention_head in self.attention_head_list], dim=2))


class MultiAttention2(nn.Module):
    '''
        This class is a more efficient implementation of Multi-Headed Attention that instead uses the direct tensor 
        calculation method instead of defining multiple attention heads and concatenating their results
    '''
    
    def __init__(self):
        super().__init__()
        self.query_weights_matrix = nn.Linear(num_embeddings, num_embeddings, device=device) 
        self.key_weights_matrix = nn.Linear(num_embeddings, num_embeddings, device=device) 
        self.value_weights_matrix = nn.Linear(num_embeddings, num_embeddings, device=device) 
        self.register_buffer('masked_weights', torch.tril(torch.ones(block_size, block_size)))


    def forward(self, input_data):
        '''
            Performs the multi-headed attention forward pass
        '''
        query = self.query_weights_matrix(input_data) 
        key = self.key_weights_matrix(input_data) 
        value = self.value_weights_matrix(input_data) 

        # now we have the query, key, and value calculations for each attention head, we want to reshape them for further calculations on each attention head
        embeddings_per_head = num_embeddings//num_heads
        query2 = query.reshape((batch_size, block_size, num_heads, embeddings_per_head))
        key2 = key.reshape((batch_size, block_size, num_heads, embeddings_per_head))
        value2 = value.reshape((batch_size, block_size, num_heads, embeddings_per_head))

        query2 = query2.transpose(1,2)
        key2 = key2.transpose(1,2)
        value2 = value2.transpose(1,2)

        # Calculate the attention scores. Use the masked weights here!
        attention_scores = query2.matmul(key2.transpose(-1, -2)) / torch.sqrt(torch.tensor(embeddings_per_head))
        # now we want to apply the masked weights
        attention_scores = attention_scores.masked_fill(self.masked_weights[:block_size, :block_size]==0, float('-inf'))


        attention_weights = attention_scores.softmax(-1)
        attention_output = attention_weights.matmul(value2)

        # Now we want to return the results by transposing and reshaping
        attention_output2 = attention_output.transpose(1,2).reshape((batch_size, block_size, num_embeddings))

        return attention_output2  


class Block_Layer(nn.Module):
    def __init__(self):
        super().__init__()
        # repeated sets of multi-headed attention
        # self.multi_headed_list = nn.Sequential(*[MultiAttention() for i in range(num_block_layers)])
        self.multi_headed_list = nn.Sequential(*[MultiAttention2() for i in range(num_block_layers)])
    

    def forward(self, input_data):
        return self.multi_headed_list(input_data)


class Transformer_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.positional_embedding_table = nn.Embedding(block_size, num_embeddings, device=device)
        self.token_embedding_table = nn.Embedding(num_tokens, num_embeddings, device=device)
        self.feed_forward = Feed_Forward(num_embeddings, num_embeddings)
        self.lin_layer = nn.Linear(num_embeddings, num_tokens, device=device)
        self.blocks = nn.Sequential(*[Block_Layer() for i in range(num_block_layers)])
        self.layer_norm = nn.LayerNorm(num_embeddings, device=device)
        self.layer_norm_2 = nn.LayerNorm(num_embeddings, device=device)



    def forward(self, input_data, expected_result=None):
        # first we obtain the positional encoding values
        num_batches, num_toks_in_batch = input_data.shape
        positional_embeddings = self.positional_embedding_table(torch.arange(num_toks_in_batch, device=device))
        token_embeddings = self.token_embedding_table(input_data)

        #positional encoding
        x = token_embeddings+positional_embeddings
        x_2 = self.blocks(x)
        x = x + x_2
        x =  self.layer_norm(x)
        x_2 = self.feed_forward(x)
        x = x+x_2 
        x = self.layer_norm_2(x)
        x = self.lin_layer(x) # this final layer calculates our most probable word from the list of words (shape is Batch-Time-Token)

        if expected_result is None: # i.e. we're generating, not training
            return x, None 

        # calculate the loss and return it. for this we're better off just turning the whole output result and expected output result into a 1 by N dimensional value and performing cross entropy
        batch_size = x.shape[0] # we only need the first one, as it is set in the setup. The other two are already avaialble in our code
        logits = x.view(batch_size*block_size, num_tokens) 
        target_values = expected_result.view(batch_size*block_size)
        loss = F.cross_entropy(logits, target_values)

        return logits, loss
         
    
    def generate(self, context: torch.tensor, max_new_tokens: int):
        '''
            Function to generate tokens. This starts by taking a specific token context and then autoregressing to get the output
        '''
        updated_context = context
        for _ in range(max_new_tokens):
            new_context = updated_context[:, -block_size:] # we have a limited context we work with when training and generating
            logits, _ = self(new_context) # we throw away the loss since it's not used in generation
            logits_2 = logits[:, -1, :] # takes the last predicted token's logits in each batch of elements
            probs = F.softmax(logits_2, dim=-1) # run softmax on each batch's logits for the last predicted token
            # here, we use the multinomial distribution
            next_token = torch.multinomial(probs, num_samples=1) # generate the next sample with a degree of randomness, as we don't always want the most probable sample for the sake of textual diversity
            updated_context = torch.cat((updated_context, next_token), dim=1) # updates the time component (list of tokens for the batch) of the given context
        return updated_context
    