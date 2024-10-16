import pytest

from app.usecases.indices_to_tokens import indices_to_tokens

glossary = [""] + ["UNK"] + ["token1", "token2", "token3"]

def test_indices_to_tokens_success():
    # Cas standard avec indices valides
    indices = [2, 3, 4]
    result = indices_to_tokens(indices, glossary)
    assert result == ["token1", "token2", "token3"], f"Expected ['token1', 'token2', 'token3'], but got {result}"

def test_indices_to_tokens_with_none():
    # Cas avec des None dans les indices
    indices = [2, None, 4]
    result = indices_to_tokens(indices, glossary)
    assert result == ["token1", "UNK", "token3"], f"Expected ['token1', 'UNK', 'token3'], but got {result}"

def test_indices_to_tokens_out_of_range():
    # Cas avec des indices hors de portée
    indices = [2, 6]
    result = indices_to_tokens(indices, glossary)
    assert result == ["token1", "UNK"], f"Expected ['token1', 'UNK'], but got {result}"

def test_indices_to_tokens_empty_list():
    # Cas avec une liste vide d'indices
    indices = []
    result = indices_to_tokens(indices, glossary)
    assert result == [], f"Expected [], but got {result}"

def test_indices_to_tokens_empty_glossary():
    # Cas avec un glossaire vide
    indices = [0, 1, 2]
    glossary = []
    result = indices_to_tokens(indices, glossary)
    assert result == ["UNK", "UNK", "UNK"], f"Expected ['UNK', 'UNK', 'UNK'], but got {result}"
