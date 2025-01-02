import requests 
from typing import Tuple, List
import torch
from transformer_building import Transformer_Model
import re


def get_text(url: str) -> str:
    '''
        Obtain the text for training purposes
    '''
    content = requests.get(url, timeout=None)
    # filter text to desired starting and ending points
    text = content.text
    tmp = list(re.finditer(r'(BOOK I\.)', text))
    start_idx_text = tmp[1].span()[0]
    text = text[start_idx_text:]
    tmp = list(re.finditer(r'(END OF THE PROJECT GUTENBERG EBOOK THE ILIAD)', text))
    end_idx_text = tmp[0].span()[0] - 4
    text = text[:end_idx_text]
    return text


def create_vocabulary(input_text: str) -> Tuple[set, int]:
    '''Takes in the text source as an input and creates a set of each unique character seen'''  
    set_text = set(input_text)
    return set_text, len(set_text)


def create_str_to_num(input_set: set) -> dict:
    '''Creates a character-to-number mapping dictionary'''
    return {letter:number for number, letter in enumerate(input_set)}


def create_num_to_str(input_dict: dict) -> dict:
    '''Creates a letter-to-character mapping dictionary'''
    return {input_dict[key]:key for key in input_dict.keys()}


def convert_input_to_nums(input_text: str, mapper: dict) -> List[int]:
    '''Takes the input data and converts it into a list of integers based on the str-to-num mapper'''
    return [mapper[s] for s in input_text]


def convert_output_to_strs(input_list: list, mapper: dict) -> List[str]:
    '''Takes the output data integer list and cconverts it into a list of characters based on the num-to-str mapper'''
    return [mapper[i] for i in input_list]


def get_batch(data: torch.tensor, block_size=8, batch_size=4) -> Tuple[torch.tensor, torch.tensor]:
    '''
        This will create a batch for processing (either training or testing). 
    '''
    nums_to_produce = torch.randint(low=0, high=len(data)-block_size+1, size=(batch_size, 1)) # produce 4 random integers for the positions of each index of the batch
    X = torch.vstack([data[i:i+block_size] for i in nums_to_produce])
    y = torch.vstack([data[i+1:i+block_size+1] for i in nums_to_produce])
    return X, y


if __name__=='__main__':
    book_text = get_text('https://www.gutenberg.org/cache/epub/3059/pg3059.txt')

    vocab, length_vocab = create_vocabulary(book_text)
    
    # create mappings
    str_to_num_mapping = create_str_to_num(vocab)
    num_to_str_mapping = create_num_to_str(str_to_num_mapping)

    # map the data
    input_to_nums = convert_input_to_nums(book_text, str_to_num_mapping)

    # split training and test data
    training_end_index = int(.9 * len(input_to_nums)) # we'll use the first 90% of the data as the training set
    training_data, testing_data = input_to_nums[:training_end_index], input_to_nums[training_end_index:]
    training_set = torch.tensor(training_data, dtype=torch.long)
    test_set = torch.tensor(testing_data, dtype=torch.long)

    # create data loader
    torch.manual_seed(42)
    block_size, batch_size = 8, 4

    # model creation, training
    model = Transformer_Model()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # optimizer setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)

    for iteration in range(10000):
        xb, yb = get_batch(training_set, block_size, batch_size)

        # evaluation of the loss . . . 
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward() # backpropogation step
        optimizer.step()
    
    # model text generation
    context = torch.zeros((1,1), dtype=torch.long, device=device) # this is to be our starting context
    generated_text = model.generate(context, max_new_tokens=500)
    print(''.join(convert_output_to_strs(generated_text[0].tolist(), mapper=num_to_str_mapping)))