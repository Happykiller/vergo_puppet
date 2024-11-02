#app\neural_network\nn_siamese_test.py
import torch
import pytest
import random
import numpy as np
from app.neural_network.nn_siamese import SiameseLSTM, train_siamese_model_nn, evaluate_similarity
from app.usecases.siamese.usecase_commons_siamese import create_glossary_from_dictionary, create_glossary_from_training_data, tokens_to_indices

def set_seed(seed=42):
    """Sets the random seed for reproducibility across PyTorch, numpy, and Python."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Verify the structure of the Siamese LSTM model
def test_siamese_lstm_structure():
    vocab_size = 100
    embedding_dim = 128
    hidden_dim = 256
    nn_model = SiameseLSTM(vocab_size, embedding_dim, hidden_dim)
    
    # Check that the model has defined layers
    assert isinstance(nn_model.embedding, torch.nn.Embedding), "Embedding layer is not correctly defined"
    assert nn_model.embedding.num_embeddings == vocab_size, "Vocabulary size in the embedding layer is incorrect"
    assert nn_model.embedding.embedding_dim == embedding_dim, "Embedding dimension is incorrect"
    
    assert isinstance(nn_model.lstm, torch.nn.LSTM), "LSTM layer is not correctly defined"
    assert nn_model.lstm.input_size == embedding_dim, "LSTM layer input size is incorrect"
    assert nn_model.lstm.hidden_size == hidden_dim, "LSTM layer output size is incorrect"

# Verify training of the Siamese LSTM model
def test_train_siamese_model_nn():
    training_data = [
        (["dog", "cat", "bird"], ["dog", "cat", "bird"], 1.0),
        (["frog", "lion", "tiger"], ["frog", "lion", "tiger"], 1.0),
        (["dog", "cat", "bird"], ["frog", "lion", "tiger"], 0.5),
        (["elephant", "giraffe", "zebra"], ["elephant", "giraffe", "zebra"], 1.0),
        (["dog", "cat", "bird"], ["elephant", "giraffe", "zebra"], 0.4),
        (["frog", "lion", "tiger"], ["elephant", "giraffe", "zebra"], 0.3)
    ]

    glossary = create_glossary_from_training_data(training_data)
    word2idx = {word: idx for idx, word in enumerate(glossary)}
    vocab_size = len(glossary)
    transformed_data = []
    for source_tokens, target_tokens, score in training_data:
        source_indices = tokens_to_indices(source_tokens, word2idx)
        target_indices = tokens_to_indices(target_tokens, word2idx)
        transformed_data.append((source_indices, target_indices, score))
    
    # Train the model
    nn_model, losses = train_siamese_model_nn(transformed_data, vocab_size, num_epochs=5)
    
    # Check that the model has been created
    assert nn_model is not None, "Siamese LSTM model was not trained correctly."
    # Check that loss decreases
    assert losses[0] > losses[-1], "Loss did not decrease during training."

# Verify the similarity evaluation function
def test_evaluate_similarity():
    training_data = [
        (["dog", "cat", "bird"], ["dog", "cat", "bird"], 1.0),
        (["frog", "lion", "tiger"], ["frog", "lion", "tiger"], 1.0),
        (["dog", "cat", "bird"], ["frog", "lion", "tiger"], 0.5),
        (["elephant", "giraffe", "zebra"], ["elephant", "giraffe", "zebra"], 1.0),
        (["dog", "cat", "bird"], ["elephant", "giraffe", "zebra"], 0.4),
        (["frog", "lion", "tiger"], ["elephant", "giraffe", "zebra"], 0.3)
    ]

    training_glossary = create_glossary_from_training_data(training_data)
    training_word2idx = {word: idx for idx, word in enumerate(training_glossary)}
    vocab_size = len(training_glossary)

    transformed_data = []
    for source_tokens, target_tokens, score in training_data:
        source_indices = tokens_to_indices(source_tokens, training_word2idx)
        target_indices = tokens_to_indices(target_tokens, training_word2idx)
        transformed_data.append((source_indices, target_indices, score))
    
    # Train the model
    nn_model, _ = train_siamese_model_nn(transformed_data, vocab_size, num_epochs=5)
    
    # Evaluate similarity between two identical sequences
    seq1 = ["man", "sit", "up"]
    seq2 = ["man", "sit", "up"]
    seq3 = ["woman", "exercise"]
    glossary = create_glossary_from_dictionary([seq1, seq2, seq3])
    word2idx = {word: idx for idx, word in enumerate(glossary)}
    seq1_indices = tokens_to_indices(seq1, word2idx)
    seq2_indices = tokens_to_indices(seq2, word2idx)
    seq3_indices = tokens_to_indices(seq3, word2idx)
    
    similarity = evaluate_similarity(nn_model, seq1_indices, seq2_indices)
    assert similarity > 0.9, "Similarity between identical sequences should be high."
    
    # Evaluate similarity between two different sequences
    similarity_diff = evaluate_similarity(nn_model, seq1_indices, seq3_indices)
    assert similarity > similarity_diff, "Similarity should be lower for different sequences."

# Verify loss decrease with continuous labels
def test_loss_decreases_with_continuous_labels():
    set_seed(42)
    training_data = [
        (["dog", "cat", "bird"], ["dog", "cat", "bird"], 1.0),
        (["dog", "cat", "bird"], ["frog", "cat", "bird"], 0.9),
        (["dog", "cat", "bird"], ["frog", "lion"], 0.5)
    ]
    glossary = create_glossary_from_training_data(training_data)
    word2idx = {word: idx for idx, word in enumerate(glossary)}
    vocab_size = len(glossary)

    transformed_data = []
    for source_tokens, target_tokens, score in training_data:
        source_indices = tokens_to_indices(source_tokens, word2idx)
        target_indices = tokens_to_indices(target_tokens, word2idx)
        transformed_data.append((source_indices, target_indices, score))
    
    # Train the model
    nn_model, losses = train_siamese_model_nn(transformed_data, vocab_size, num_epochs=10)
    
    # Check that loss decreases
    assert losses[0] > losses[-1], "Loss did not decrease during training with continuous labels."

# Verify stability of search results
def test_search_stability():
    training_data = [
        (["dog", "cat", "bird"], ["dog", "cat", "bird"], 1.0),
        (["frog", "lion", "tiger"], ["frog", "lion", "tiger"], 1.0),
        (["dog", "cat", "bird"], ["frog", "lion", "tiger"], 0.5),
        (["elephant", "giraffe", "zebra"], ["elephant", "giraffe", "zebra"], 1.0),
        (["dog", "cat", "bird"], ["elephant", "giraffe", "zebra"], 0.4),
        (["frog", "lion", "tiger"], ["elephant", "giraffe", "zebra"], 0.3)
    ]

    training_glossary = create_glossary_from_training_data(training_data)
    training_word2idx = {word: idx for idx, word in enumerate(training_glossary)}
    vocab_size = len(training_glossary)

    transformed_data = []
    for source_tokens, target_tokens, score in training_data:
        source_indices = tokens_to_indices(source_tokens, training_word2idx)
        target_indices = tokens_to_indices(target_tokens, training_word2idx)
        transformed_data.append((source_indices, target_indices, score))
    
    # Train the model
    nn_model, _ = train_siamese_model_nn(transformed_data, vocab_size, num_epochs=5)
    
    # Create search vector and dictionary for stability test
    search_vector = ["man", "lifting", "weights"]
    dictionary = [
        ["man", "lifting", "weights"],
        ["woman", "lifting", "weights"],
        ["man", "sit", "up"]
    ]
    glossary = create_glossary_from_dictionary(dictionary)
    word2idx = {word: idx for idx, word in enumerate(glossary)}
    search_indices = tokens_to_indices(search_vector, word2idx)
    
    # Repeat search to check stability of results
    for _ in range(5):
        similarities = []
        for vector in dictionary:
            vector_indices = tokens_to_indices(vector, word2idx)
            similarity = evaluate_similarity(nn_model, search_indices, vector_indices)
            similarities.append((vector, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        best_match = similarities[0][0]
        assert best_match == ["man", "lifting", "weights"], "Search result is not stable."

# Verify the similarity evaluation function
def test_evaluate_similarity_seq_varia():
    training_data = [
        (["dog", "cat", "bird"], ["dog", "cat", "bird"], 1.0),
        (["dog", "cat"], ["dog", "cat", "bird"], 0.6),
        (["dog", "cat", "tiger"], ["dog", "cat"], 0.6),
        (["dog", "cat"], ["dog", "bird", "cat"], 0.5),
    ]

    training_glossary = create_glossary_from_training_data(training_data)
    training_word2idx = {word: idx for idx, word in enumerate(training_glossary)}
    vocab_size = len(training_glossary)

    transformed_data = []
    for source_tokens, target_tokens, score in training_data:
        source_indices = tokens_to_indices(source_tokens, training_word2idx)
        target_indices = tokens_to_indices(target_tokens, training_word2idx)
        transformed_data.append((source_indices, target_indices, score))
    
    # Train the model
    nn_model, _ = train_siamese_model_nn(transformed_data, vocab_size, num_epochs=5)
    
    # Evaluate similarity between varied sequences
    seq1 = ["man", "sit"]
    seq2 = ["man", "sit", "up"]
    glossary = create_glossary_from_dictionary([["man", "sit"], ["man", "sit", "up"]])
    word2idx = {word: idx for idx, word in enumerate(glossary)}
    seq1_indices = tokens_to_indices(seq1, word2idx)
    seq2_indices = tokens_to_indices(seq2, word2idx)
    similarity = evaluate_similarity(nn_model, seq1_indices, seq2_indices)
    
    assert similarity > 0.5, "Similarity between varied sequences should be above 0.5."
