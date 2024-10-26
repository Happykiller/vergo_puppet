import pytest
from typing import List, Optional

# Importer la fonction que nous voulons tester
from app.usecases.tokens_to_indices import tokens_to_indices

glossary = [""] + ["UNK"] + ["token1", "token2", "token3"]

# Exemple de la fonction à tester si elle n'est pas dans un fichier séparé
# def tokens_to_indices(tokens: List[str], glossary: List[str]) -> List[Optional[int]]:
#     glossary_dict = {word: idx for idx, word in enumerate(glossary)}
#     return [glossary_dict.get(token, None) for token in tokens]

# Test 1: Tous les tokens sont trouvés dans le glossaire
def test_all_tokens_found():
    tokens = ["token1", "token2", "token3"]
    expected = [2, 3, 4]
    result = tokens_to_indices(tokens, glossary)
    assert result == expected, f"Expected {expected}, but got {result}"

# Test 2: Certains tokens ne sont pas dans le glossaire
def test_some_tokens_not_found():
    tokens = ["token1", "token4", "token2"]
    expected = [2, 1, 3]
    result = tokens_to_indices(tokens, glossary)
    assert result == expected, f"Expected {expected}, but got {result}"

# Test 3: Liste de tokens vide
def test_empty_tokens_list():
    tokens = []
    expected = []
    result = tokens_to_indices(tokens, glossary)
    assert result == expected, f"Expected {expected}, but got {result}"

# Test 4: Glossaire vide
def test_empty_glossary():
    tokens = ["token1", "token2", "token3"]
    glossary = [""] + ["UNK"] + []
    expected = [1, 1, 1]
    result = tokens_to_indices(tokens, glossary)
    assert result == expected, f"Expected {expected}, but got {result}"

# Test 5: Tokens en double
def test_duplicate_tokens():
    tokens = ["token1", "token1", "token2"]
    expected = [2, 2, 3]
    result = tokens_to_indices(tokens, glossary)
    assert result == expected, f"Expected {expected}, but got {result}"

# Test 6: Glossaire avec des tokens en double
def test_duplicate_glossary():
    tokens = ["token1", "token2", "token3"]
    glossary = [""] + ["UNK"] + ["token1", "token2", "token3", "token1"]
    expected = [2, 3, 4]  # Le glossaire doit être unique dans le traitement
    result = tokens_to_indices(tokens, glossary)
    assert result == expected, f"Expected {expected}, but got {result}"
