from typing import List, Optional

def tokens_to_indices(tokens: List[str], glossary: List[str]) -> List[Optional[int]]:
    # Créer un dictionnaire pour mapper chaque token à son premier index dans le glossaire
    glossary_dict = {}
    for idx, word in enumerate(glossary):
        if word not in glossary_dict:
            glossary_dict[word] = idx  # Seule la première occurrence est sauvegardée

    # Utiliser le dictionnaire pour retrouver les indices des tokens
    return [glossary_dict.get(token, None) for token in tokens]