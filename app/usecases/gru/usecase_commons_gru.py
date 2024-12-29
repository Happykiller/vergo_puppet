# app\usecases\gru\usecase_commons_gru.py
from typing import List

def process_input(tokens: List[str], word2idx):
    """
    Transforms a list of tokens into a tensor of indices with padding.
    :param tokens: List of tokens in the sequence to be classified.
    :param word2idx: Dictionary mapping words to their respective indices.
    :return: Tensor-like list representing the prepared sequence with padding.
    """
    # Convert each token to its index; if the token is not found, use the '<PAD>' index
    seq = [word2idx.get(token, word2idx['<PAD>']) for token in tokens]
    
    # Define the maximum sequence length (in this case, the original length of the sequence)
    max_seq_length = len(seq)
    
    # Add padding to ensure the sequence length matches max_seq_length
    seq += [word2idx['<PAD>']] * (max_seq_length - len(seq))
    
    return seq
