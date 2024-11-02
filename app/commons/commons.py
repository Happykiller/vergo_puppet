#app\commons\commons.py
from typing import List

# Fonction pour convertir une séquence de tokens en indices
def tokens_to_indices(tokens: List[str], word2idx: dict) -> List[int]:
    # Convertit chaque token en son indice dans le vocabulaire
    return [word2idx.get(token, 0) for token in tokens]