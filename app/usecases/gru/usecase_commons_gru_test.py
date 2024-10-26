import pytest
from app.usecases.gru.usecase_commons_gru import process_input

def test_process_input():
    # Dictionnaire de mapping mot -> indice, incluant le token de padding
    word2idx = {'hello': 1, 'world': 2, '<PAD>': 0}
    
    # Liste de tokens à transformer
    tokens = ['hello', 'unknown', 'world']
    
    # Appel de la fonction process_input
    processed_seq = process_input(tokens, word2idx)
    
    # Vérification des résultats
    expected_seq = [1, 0, 2]  # 'hello' -> 1, 'unknown' -> <PAD> (0), 'world' -> 2
    assert processed_seq == expected_seq, f"Expected {expected_seq} but got {processed_seq}"
    
    # Test supplémentaire pour vérifier le padding avec une séquence plus courte
    tokens_short = ['hello']
    processed_seq_short = process_input(tokens_short, word2idx)
    
    # Comme max_seq_length = 1 dans ce cas, pas de padding additionnel
    expected_seq_short = [1]
    assert processed_seq_short == expected_seq_short, f"Expected {expected_seq_short} but got {processed_seq_short}"

    # Test avec une séquence contenant seulement un mot inconnu
    tokens_unknown = ['unknown']
    processed_seq_unknown = process_input(tokens_unknown, word2idx)
    
    # Doit retourner le token <PAD> pour les mots inconnus
    expected_seq_unknown = [0]
    assert processed_seq_unknown == expected_seq_unknown, f"Expected {expected_seq_unknown} but got {processed_seq_unknown}"

