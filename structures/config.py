from dataclasses import dataclass


@dataclass
class Params:
    tiny_shakespeare: str


params_default = {
    'tiny_shakespeare': 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
}


def get_params():
    return Params(**params_default)