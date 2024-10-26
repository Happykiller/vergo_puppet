from typing import List

def process_input(tokens: List[str], word2idx):
    """
    Transforme la liste de tokens en tenseur d'indices avec padding.
    :param tokens: Liste de tokens de la séquence à classer.
    :param word2idx: Dictionnaire de mapping mot->indice.
    :return: Tenseur de la séquence préparée.
    """
    seq = [word2idx.get(token, word2idx['<PAD>']) for token in tokens]
    max_seq_length = len(seq)
    seq += [word2idx['<PAD>']] * (max_seq_length - len(seq))  # Padding si nécessaire
    return seq
