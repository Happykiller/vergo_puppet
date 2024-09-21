import pytest

from app.usecases.indices_to_tokens import indices_to_tokens

def test_indices_to_tokens_success():
    # Cas standard avec indices valides
    indices = [0, 1, 2]
    glossary = ["token1", "token2", "token3"]
    result = indices_to_tokens(indices, glossary)
    assert result == ["token1", "token2", "token3"], f"Expected ['token1', 'token2', 'token3'], but got {result}"

def test_indices_to_tokens_with_none():
    # Cas avec des None dans les indices
    indices = [0, None, 2]
    glossary = ["token1", "token2", "token3"]
    result = indices_to_tokens(indices, glossary)
    assert result == ["token1", None, "token3"], f"Expected ['token1', None, 'token3'], but got {result}"

def test_indices_to_tokens_out_of_range():
    # Cas avec des indices hors de portée
    indices = [0, 3, 2]
    glossary = ["token1", "token2", "token3"]
    result = indices_to_tokens(indices, glossary)
    assert result == ["token1", None, "token3"], f"Expected ['token1', None, 'token3'], but got {result}"

def test_indices_to_tokens_empty_list():
    # Cas avec une liste vide d'indices
    indices = []
    glossary = ["token1", "token2", "token3"]
    result = indices_to_tokens(indices, glossary)
    assert result == [], f"Expected [], but got {result}"

def test_indices_to_tokens_empty_glossary():
    # Cas avec un glossaire vide
    indices = [0, 1, 2]
    glossary = []
    result = indices_to_tokens(indices, glossary)
    assert result == [None, None, None], f"Expected [None, None, None], but got {result}"
