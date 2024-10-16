from typing import Dict, List, Tuple

def pad_vector(vector, max_length):
    """Ajoute du padding à un vecteur pour qu'il ait la taille max_length."""
    return vector + [0] * (max_length - len(vector))

def create_glossary_from_training_data(training_data: List[Tuple[List[str], List[str], float]]) -> List[str]:
    unique_tokens = set()
    for seq1, seq2, _ in training_data:
        unique_tokens.update(seq1)
        unique_tokens.update(seq2)
    # Ajouter éventuellement <PAD> si nécessaire
    return ["<PAD>"] + sorted(unique_tokens)

def create_glossary_from_dictionary(dictionary: List[List[str]]) -> List[str]:
    unique_tokens = set()
    for seq in dictionary:
        unique_tokens.update(seq)
    # Ajouter éventuellement <PAD> si nécessaire
    return ["<PAD>"] + sorted(unique_tokens)

def create_indexed_glossary(glossary: List[str]) -> Dict[str, int]:
    # Démarrer avec le token spécial <PAD> associé à l'index 0
    indexed_glossary = {"<PAD>": 0}
    
    # Ajouter les mots du glossaire en commençant à l'index 1
    for idx, word in enumerate(sorted(glossary[1:]), 1):  # On saute <PAD> lors du tri
        indexed_glossary[word] = idx
    
    return indexed_glossary

# Fonction pour convertir une séquence de tokens en indices
def tokens_to_indices(tokens: List[str], word2idx: dict) -> List[int]:
    # Convertit chaque token en son indice dans le vocabulaire
    return [word2idx.get(token, 0) for token in tokens]