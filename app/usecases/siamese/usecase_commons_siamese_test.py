#app\usecases\siamese\usecase_commons_siamese_test.py
from app.usecases.siamese.usecase_commons_siamese import (
    create_glossary_from_dictionary,
    create_glossary_from_training_data,
    create_indexed_glossary,
    tokens_to_indices
)

glossary = [""] + ["UNK"] + ["token1", "token2", "token3"]

# Test for the create_glossary_from_training_data function
def test_create_glossary_from_training_data():
    training_data = [
        (["apple", "banana"], ["cherry", "date"], 0.8),
        (["elderberry", "fig"], ["grape", "apple"], 0.6)
    ]
    result = create_glossary_from_training_data(training_data)
    expected = ["<PAD>", "apple", "banana", "cherry", "date", "elderberry", "fig", "grape"]
    assert result == expected, f"Expected {expected}, but got {result}"

# Test for the create_glossary_from_dictionary function
def test_create_glossary_from_dictionary():
    dictionary = [["apple", "banana"], ["cherry", "date"], ["elderberry", "fig"]]
    result = create_glossary_from_dictionary(dictionary)
    expected = ["<PAD>", "apple", "banana", "cherry", "date", "elderberry", "fig"]
    assert result == expected, f"Expected {expected}, but got {result}"

# Test for the create_indexed_glossary function
def test_create_indexed_glossary():
    glossary = ["<PAD>", "apple", "banana", "cherry", "date"]
    result = create_indexed_glossary(glossary)
    expected = {"<PAD>": 0, "apple": 1, "banana": 2, "cherry": 3, "date": 4}
    assert result == expected, f"Expected {expected}, but got {result}"

# Test when all tokens are found in the glossary
def test_all_tokens_found():
    tokens = ["token1", "token2", "token3"]
    expected = [2, 3, 4]
    result = tokens_to_indices(tokens, glossary)
    assert result == expected, f"Expected {expected}, but got {result}"

# Test when some tokens are not found in the glossary
def test_some_tokens_not_found():
    tokens = ["token1", "token4", "token2"]
    expected = [2, 1, 3]
    result = tokens_to_indices(tokens, glossary)
    assert result == expected, f"Expected {expected}, but got {result}"

# Test with an empty token list
def test_empty_tokens_list():
    tokens = []
    expected = []
    result = tokens_to_indices(tokens, glossary)
    assert result == expected, f"Expected {expected}, but got {result}"

# Test with an empty glossary
def test_empty_glossary():
    tokens = ["token1", "token2", "token3"]
    glossary = [""] + ["UNK"] + []
    expected = [1, 1, 1]
    result = tokens_to_indices(tokens, glossary)
    assert result == expected, f"Expected {expected}, but got {result}"

# Test with duplicate tokens in the list
def test_duplicate_tokens():
    tokens = ["token1", "token1", "token2"]
    expected = [2, 2, 3]
    result = tokens_to_indices(tokens, glossary)
    assert result == expected, f"Expected {expected}, but got {result}"

# Test when the glossary has duplicate tokens
def test_duplicate_glossary():
    tokens = ["token1", "token2", "token3"]
    glossary = [""] + ["UNK"] + ["token1", "token2", "token3", "token1"]
    expected = [2, 3, 4]  # The glossary should be unique after processing
    result = tokens_to_indices(tokens, glossary)
    assert result == expected, f"Expected {expected}, but got {result}"
