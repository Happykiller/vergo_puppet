from typing import List, Tuple

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