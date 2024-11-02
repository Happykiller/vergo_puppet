#app\usecases\gru\usecase_commons_gru_test.py
import pytest
from app.usecases.gru.usecase_commons_gru import process_input

def test_process_input():
    """
    Test the process_input function to ensure it correctly converts tokens to indices,
    handles unknown words with padding, and verifies padding behavior for shorter sequences.
    """
    # Dictionary mapping words to indices, including the padding token
    word2idx = {'hello': 1, 'world': 2, '<PAD>': 0}
    
    # List of tokens to convert
    tokens = ['hello', 'unknown', 'world']
    
    # Call the process_input function
    processed_seq = process_input(tokens, word2idx)
    
    # Verify the output matches expected indices
    expected_seq = [1, 0, 2]  # 'hello' -> 1, 'unknown' -> <PAD> (0), 'world' -> 2
    assert processed_seq == expected_seq, f"Expected {expected_seq} but got {processed_seq}"
    
    # Additional test to check padding behavior with a shorter sequence
    tokens_short = ['hello']
    processed_seq_short = process_input(tokens_short, word2idx)
    
    # Since max_seq_length = 1 here, no additional padding is added
    expected_seq_short = [1]
    assert processed_seq_short == expected_seq_short, f"Expected {expected_seq_short} but got {processed_seq_short}"

    # Test with a sequence containing only an unknown word
    tokens_unknown = ['unknown']
    processed_seq_unknown = process_input(tokens_unknown, word2idx)
    
    # Should return the <PAD> token index for unknown words
    expected_seq_unknown = [0]
    assert processed_seq_unknown == expected_seq_unknown, f"Expected {expected_seq_unknown} but got {processed_seq_unknown}"
