from typing import List, Optional

def indices_to_tokens(indices: List[Optional[int]], glossary: List[str]) -> List[Optional[str]]:
    # Liste pour stocker les résultats
    result = []

    # Pour chaque indice dans la liste d'indices
    for idx in indices:
        if idx is not None and 0 <= idx < len(glossary):
            # Ajouter le token correspondant si l'indice est valide
            result.append(glossary[idx])
        else:
            # Ajouter None si l'indice est None ou hors de portée
            result.append("UNK")

    return result
