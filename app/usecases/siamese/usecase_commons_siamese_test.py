#app\usecases\siamese\usecase_commons_siamese_test.py
from app.usecases.siamese.usecase_commons_siamese import create_glossary_from_dictionary, create_glossary_from_training_data, create_indexed_glossary

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