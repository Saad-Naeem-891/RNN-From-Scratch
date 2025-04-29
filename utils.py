import numpy as np

def one_hot(index, vocab_size):
    vec = np.zeros((vocab_size, 1))
    vec[index] = 1
    return vec

def create_vocab(text):
    vocab = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(vocab)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    return vocab, char_to_idx, idx_to_char
