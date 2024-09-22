def pad_vector(vector, max_length):
    """Ajoute du padding à un vecteur pour qu'il ait la taille max_length."""
    return vector + [0] * (max_length - len(vector))