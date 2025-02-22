#app\usecases\siamese\usecase_commons_siamese_test.py
from app.usecases.siamese.usecase_commons_siamese import (
    calculate_word_representation,
    create_glossary_from_dictionary,
    create_glossary_from_training_data,
    create_indexed_glossary,
    tokens_to_indices
)

# Test for the create_glossary_from_training_data function
def test_create_glossary_from_training_data():
    training_data = [
        (["apple", "banana"], ["cherry", "date"], 0.8),
        (["elderberry", "fig"], ["grape", "apple"], 0.6)
    ]
    result = create_glossary_from_training_data(training_data)
    expected = ["<PAD>", "UNK", "apple", "banana", "cherry", "date", "elderberry", "fig", "grape"]
    assert result == expected, f"Expected {expected}, but got {result}"

# Test for the create_glossary_from_dictionary function
def test_create_glossary_from_dictionary():
    dictionary = [["apple", "banana"], ["cherry", "date"], ["elderberry", "fig"]]
    result = create_glossary_from_dictionary(dictionary)
    expected = ["<PAD>", "UNK", "apple", "banana", "cherry", "date", "elderberry", "fig"]
    assert result == expected, f"Expected {expected}, but got {result}"

# Test for the create_indexed_glossary function
def test_create_indexed_glossary():
    glossary = ["<PAD>", "UNK", "apple", "banana", "cherry", "date"]
    result = create_indexed_glossary(glossary)
    expected = {"<PAD>": 0, "UNK": 1, "apple": 2, "banana": 3, "cherry": 4, "date": 5}
    assert result == expected, f"Expected {expected}, but got {result}"

# Test when all tokens are found in the glossary
def test_all_tokens_found():
    tokens = ["token1", "token2", "token3"]
    glossary = ["<PAD>"] + ["UNK"] + ["token1", "token2", "token3"]
    expected = [2, 3, 4]
    result = tokens_to_indices(tokens, glossary)
    assert result == expected, f"Expected {expected}, but got {result}"

# Test when some tokens are not found in the glossary
def test_some_tokens_not_found():
    tokens = ["token1", "token4", "token2"]
    glossary = ["<PAD>"] + ["UNK"] + ["token1", "token2", "token3"]
    expected = [2, 1, 3]
    result = tokens_to_indices(tokens, glossary)
    assert result == expected, f"Expected {expected}, but got {result}"

# Test with an empty token list
def test_empty_tokens_list():
    tokens = []
    expected = []
    glossary = ["<PAD>"] + ["UNK"] + ["token1", "token2", "token3"]
    result = tokens_to_indices(tokens, glossary)
    assert result == expected, f"Expected {expected}, but got {result}"

# Test with an empty glossary
def test_empty_glossary():
    tokens = ["token1", "token2", "token3"]
    glossary = ["<PAD>"] + ["UNK"] + []
    expected = [1, 1, 1]
    result = tokens_to_indices(tokens, glossary)
    assert result == expected, f"Expected {expected}, but got {result}"

# Test with duplicate tokens in the list
def test_duplicate_tokens():
    tokens = ["token1", "token1", "token2"]
    expected = [2, 2, 3]
    glossary = ["<PAD>"] + ["UNK"] + ["token1", "token2", "token3"]
    result = tokens_to_indices(tokens, glossary)
    assert result == expected, f"Expected {expected}, but got {result}"

# Test when the glossary has duplicate tokens
def test_duplicate_glossary():
    tokens = ["token1", "token2", "token3"]
    glossary = ["<PAD>"] + ["UNK"] + ["token1", "token2", "token3", "token1"]
    expected = [2, 3, 4]  # The glossary should be unique after processing
    result = tokens_to_indices(tokens, glossary)
    assert result == expected, f"Expected {expected}, but got {result}"

def test_calculate_word_representation():
    """
    Test the calculation of word representation percentages.
    """
    dictionary = [
        ["chat", "chien", "oiseau"],
        ["chien", "souris", "chat"],
        ["chat", "chat", "chien"]
    ]

    expected_output = {
        "chat": 44.44,   # 4 occurrences / 9 mots
        "chien": 33.33,  # 3 occurrences / 9 mots
        "oiseau": 11.11, # 1 occurrence / 9 mots
        "souris": 11.11  # 1 occurrence / 9 mots
    }

    result = calculate_word_representation(dictionary)

    # Vérification que les clés correspondent
    assert set(result.keys()) == set(expected_output.keys())

    # Vérification des valeurs arrondies
    for key in result:
        assert round(result[key], 2) == expected_output[key], f"Mismatch for {key}: expected {expected_output[key]}, got {round(result[key], 2)}"

def test_calculate_word_representation_empty():
    """
    Test the behavior when dictionary is empty.
    """
    dictionary = []
    result = calculate_word_representation(dictionary)
    
    assert result == {}, "The output should be an empty dictionary for an empty input."
