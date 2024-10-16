import pytest
from app.commons.commons import pad_vector, create_glossary_from_training_data, create_glossary_from_dictionary, create_indexed_glossary, tokens_to_indices

# Test de la fonction pad_vector
def test_pad_vector():
    vector = [1, 2, 3]
    max_length = 5
    result = pad_vector(vector, max_length)
    assert result == [1, 2, 3, 0, 0], f"Expected [1, 2, 3, 0, 0], but got {result}"

    # Cas où le vecteur est déjà plus long que max_length
    result = pad_vector(vector, 2)
    assert result == [1, 2, 3], f"Expected [1, 2, 3], but got {result}"

# Test de la fonction create_glossary_from_training_data
def test_create_glossary_from_training_data():
    training_data = [
        (["apple", "banana"], ["cherry", "date"], 0.8),
        (["elderberry", "fig"], ["grape", "apple"], 0.6)
    ]
    result = create_glossary_from_training_data(training_data)
    expected = ["<PAD>", "apple", "banana", "cherry", "date", "elderberry", "fig", "grape"]
    assert result == expected, f"Expected {expected}, but got {result}"

# Test de la fonction create_glossary_from_dictionary
def test_create_glossary_from_dictionary():
    dictionary = [["apple", "banana"], ["cherry", "date"], ["elderberry", "fig"]]
    result = create_glossary_from_dictionary(dictionary)
    expected = ["<PAD>", "apple", "banana", "cherry", "date", "elderberry", "fig"]
    assert result == expected, f"Expected {expected}, but got {result}"

# Test de la fonction create_indexed_glossary
def test_create_indexed_glossary():
    glossary = ["<PAD>", "apple", "banana", "cherry", "date"]
    result = create_indexed_glossary(glossary)
    expected = {"<PAD>": 0, "apple": 1, "banana": 2, "cherry": 3, "date": 4}
    assert result == expected, f"Expected {expected}, but got {result}"

# Test de la fonction tokens_to_indices
def test_tokens_to_indices():
    tokens = ["apple", "banana", "unknown"]
    word2idx = {"<PAD>": 0, "apple": 1, "banana": 2}
    result = tokens_to_indices(tokens, word2idx)
    expected = [1, 2, 0]  # "unknown" doit retourner l'index 0 correspondant à <PAD>
    assert result == expected, f"Expected {expected}, but got {result}"

