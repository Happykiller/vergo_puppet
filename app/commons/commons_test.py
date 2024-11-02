#app\commons\commons_test.py
import pytest
from app.commons.commons import tokens_to_indices

# Test de la fonction tokens_to_indices
def test_tokens_to_indices():
    tokens = ["apple", "banana", "unknown"]
    word2idx = {"<PAD>": 0, "apple": 1, "banana": 2}
    result = tokens_to_indices(tokens, word2idx)
    expected = [1, 2, 0]  # "unknown" doit retourner l'index 0 correspondant à <PAD>
    assert result == expected, f"Expected {expected}, but got {result}"