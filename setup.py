import requests 
from typing import Tuple, List
import torch



def get_text(url: str) -> str:
    '''
        Obtain the text for training purposes
    '''
    content = requests.get(url)
    return content.text


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


if __name__=='__main__':
    book_text = get_text('https://www.gutenberg.org/cache/epub/59306/pg59306.txt')
    vocab, length_vocab = create_vocabulary(book_text)
    
    # create mappings
    str_to_num_mapping = create_str_to_num(vocab)
    num_to_str_mapping = create_num_to_str(str_to_num_mapping)

    # map the data
    input_to_nums = convert_input_to_nums(book_text, str_to_num_mapping)

    # split training and test data
    training_end_index = int(.9 * len(input_to_nums)) # we'll use the first 90% of the data as the training set
    training_set, testing_set = input_to_nums[:training_end_index], input_to_nums[training_end_index:]
    training_set = torch.tensor(training_set, dtype=torch.long)
    test_set = torch.tensor(testing_set, dtype=torch.long)

    # create data loader
    