from typing import List, Optional

def tokens_to_indices(tokens: List[str], glossary: List[str]) -> List[Optional[int]]:
    # Créer un glossaire unique en conservant uniquement la première occurrence
    unique_glossary = []
    seen = set()
    for word in glossary:
        if word not in seen:
            seen.add(word)
            unique_glossary.append(word)

    # Créer un dictionnaire pour mapper chaque token à son premier index dans le glossaire unique
    glossary_dict = {word: idx for idx, word in enumerate(unique_glossary)}

    # Utiliser le dictionnaire pour retrouver les indices des tokens
    return [glossary_dict.get(token, glossary_dict.get("UNK")) for token in tokens]