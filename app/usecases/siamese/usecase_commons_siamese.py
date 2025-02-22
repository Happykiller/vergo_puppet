# app\usecases\siamese\usecase_commons_siamese.py
from collections import Counter
from typing import Dict, List, Optional, Tuple

def create_glossary_from_training_data(training_data: List[Tuple[List[str], List[str], float]]) -> List[str]:
    """
    Creates a glossary from the training data, including unique tokens from all sequences.
    :param training_data: List of tuples containing pairs of sequences and similarity scores.
    :return: A sorted list of unique tokens with <PAD> added as the first element.
    """
    unique_tokens = set()
    for seq1, seq2, _ in training_data:
        unique_tokens.update(seq1)
        unique_tokens.update(seq2)
        
    # Convert to sorted list and add special tokens
    return ["<PAD>", "UNK"] + sorted(unique_tokens)

def create_glossary_from_dictionary(dictionary: List[List[str]]) -> List[str]:
    """
    Creates a glossary from a dictionary, adding all unique tokens across sequences.
    
    :param dictionary: List of token sequences.
    :return: A sorted list of unique tokens with <PAD> and UNK at the beginning.
    """
    unique_tokens = set()
    
    # Collect all unique tokens from dictionary
    for sequence in dictionary:
        unique_tokens.update(sequence)

    # Convert to sorted list and add special tokens
    return ["<PAD>", "UNK"] + sorted(unique_tokens)

def create_indexed_glossary(glossary: List[str]) -> Dict[str, int]:
    """
    Creates an indexed glossary where each token is mapped to a unique index.
    :param glossary: List of tokens to index.
    :return: A dictionary mapping each token to a unique index, with <PAD> mapped to 0.
    """
    # Start with the special token <PAD> assigned to index 0
    indexed_glossary = {"<PAD>": 0}
    
    # Add words from the glossary, starting at index 1
    for idx, word in enumerate(sorted(glossary[1:]), 1):  # Skip <PAD> during sorting
        indexed_glossary[word] = idx
    
    return indexed_glossary

def tokens_to_indices(tokens: List[str], glossary: List[str]) -> List[Optional[int]]:
    """
    Converts a list of tokens into indices based on the glossary.
    :param tokens: List of tokens to convert.
    :param glossary: Glossary list containing unique tokens.
    :return: List of indices for each token, with unknown tokens mapped to the 'UNK' token index.
    """
    # Create a unique glossary keeping only the first occurrence of each token
    unique_glossary = []
    seen = set()
    for word in glossary:
        if word not in seen:
            seen.add(word)
            unique_glossary.append(word)

    # Create a dictionary to map each token to its first index in the unique glossary
    glossary_dict = {word: idx for idx, word in enumerate(unique_glossary)}

    # Use the dictionary to retrieve the indices of tokens
    return [glossary_dict.get(token, glossary_dict.get("UNK")) for token in tokens]

def calculate_word_representation(dictionary: List[List[str]]) -> Dict[str, float]:
    """
    Calculates the representation rate of words used in the dictionary vectors.
    
    :param dictionary: List of token sequences representing dictionary entries.
    :return: Dictionary where keys are words and values are their representation percentage.
    """
    word_counts = Counter()
    total_words = 0

    # Count occurrences of each word
    for sequence in dictionary:
        word_counts.update(sequence)
        total_words += len(sequence)

    # Compute percentage representation of each word
    word_representation = {word: (count / total_words) * 100 for word, count in word_counts.items()}

    # Sort by representation percentage in descending order
    sorted_word_representation = dict(sorted(word_representation.items(), key=lambda item: item[1], reverse=True))

    return sorted_word_representation