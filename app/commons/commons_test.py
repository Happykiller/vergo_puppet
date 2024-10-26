import pytest
import numpy as np
from app.commons.commons import min_max_normalize, pad_vector, create_glossary_from_training_data, create_glossary_from_dictionary, create_indexed_glossary, tokens_to_indices

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

# Test de la fonction min_max_normalize
def test_min_max_normalize():
    data = [1, 2, 3, 4, 5]
    result = min_max_normalize(data)
    expected = [0.0, 0.25, 0.5, 0.75, 1.0]
    assert np.allclose(result, expected), f"Expected {expected}, but got {result}"

    # Cas où tous les éléments sont identiques
    data_identical = [2, 2, 2]
    result_identical = min_max_normalize(data_identical)
    expected_identical = [2, 2, 2]  # Aucun changement si min == max
    assert np.allclose(result_identical, expected_identical), f"Expected {expected_identical}, but got {result_identical}"

    # Cas avec des données négatives
    data_negative = [-5, -3, -1, 1, 3]
    result_negative = min_max_normalize(data_negative)
    expected_negative = [0.0, 0.25, 0.5, 0.75, 1.0]
    assert np.allclose(result_negative, expected_negative), f"Expected {expected_negative}, but got {result_negative}"