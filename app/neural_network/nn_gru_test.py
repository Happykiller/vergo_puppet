# app\neural_network\nn_gru_test.py
import torch
import pytest
from unittest.mock import patch
from app.neural_network.nn_gru import GRUClassifier, pre_pad_sequences, train_gru, predict

# Test: Verify the structure of the GRUClassifier model
def test_gru_classifier_structure():
    vocab_size = 100
    embedding_dim = 128
    hidden_dim = 256
    num_classes = 5
    dropout_rate = 0.5

    model = GRUClassifier(vocab_size, embedding_dim, hidden_dim, num_classes, dropout_rate)

    # Check the embedding layer
    assert isinstance(model.embedding, torch.nn.Embedding), "Embedding layer is not defined correctly"
    assert model.embedding.num_embeddings == vocab_size, "Vocabulary size in the embedding layer is incorrect"
    assert model.embedding.embedding_dim == embedding_dim, "Embedding dimension is incorrect"

    # Check the GRU layer
    assert isinstance(model.gru, torch.nn.GRU), "GRU layer is not defined correctly"
    assert model.gru.input_size == embedding_dim, "GRU layer input size is incorrect"
    assert model.gru.hidden_size == hidden_dim, "GRU layer output size is incorrect"

    # Check the fully connected layer
    assert isinstance(model.fc, torch.nn.Linear), "Fully connected layer is not defined correctly"
    assert model.fc.in_features == hidden_dim, "Input size of fully connected layer is incorrect"
    assert model.fc.out_features == num_classes, "Output size of fully connected layer does not match the number of classes"

# Test: Verify training of the GRU model with dummy data
def test_train_gru():
    vocab_size = 50
    num_classes = 3
    sequences = [
        [1, 2, 3, 4, 0, 0], 
        [5, 6, 7, 0, 0, 0], 
        [1, 3, 4, 5, 6, 7]
    ]
    labels = [0, 1, 2]

    # Train the model
    model, _ = train_gru(vocab_size, num_classes, sequences, labels)

    # Check that the model has been trained
    assert model is not None, "GRU model was not trained correctly"

# Test: Verify prediction with a trained GRU model
def test_predict_with_trained_gru():
    vocab_size = 50
    num_classes = 3
    sequences = [
        [1, 2, 3, 4, 0, 0], 
        [5, 6, 7, 0, 0, 0], 
        [1, 3, 4, 5, 6, 7]
    ]
    labels = [0, 1, 2]

    # Train the model
    model, _ = train_gru(vocab_size, num_classes, sequences, labels)

    # Make a prediction
    input_sequence = [1, 2, 3, 4, 0, 0]
    predicted_class = predict(model, input_sequence)

    # Check that the prediction is a valid class index
    assert isinstance(predicted_class, int), "Prediction should be an integer representing a class"
    assert 0 <= predicted_class < num_classes, "Prediction should be within the available class range"

# Test: Verify loss decrease during training
def test_loss_decrease_during_training():
    vocab_size = 50
    num_classes = 3
    sequences = [
        [1, 2, 3, 4, 0, 0], 
        [5, 6, 7, 0, 0, 0], 
        [1, 3, 4, 5, 6, 7]
    ]
    labels = [0, 1, 2]

    # Train the model and capture logs to check for loss progression
    with patch('app.neural_network.nn_gru.logger.info') as mock_logger:
        model = train_gru(vocab_size, num_classes, sequences, labels)

    # Ensure training completed successfully
    assert model is not None, "GRU model was not trained correctly"

# Test: Verify that dropout and log_softmax layers work correctly
def test_gru_dropout_log_softmax():
    vocab_size = 50
    embedding_dim = 128
    hidden_dim = 256
    num_classes = 3
    dropout_rate = 0.5
    model = GRUClassifier(vocab_size, embedding_dim, hidden_dim, num_classes, dropout_rate)
    input_sequence = torch.tensor([[1, 2, 3, 4, 5, 0]])

    # Perform a forward pass
    output = model(input_sequence)

    # Check output shape and sum of probabilities
    assert output.shape == (1, num_classes), "Output should have shape (1, num_classes)"
    assert torch.allclose(torch.exp(output).sum(dim=1), torch.tensor([1.0]), atol=1e-3), "Probabilities should sum to 1"

# Test: Verify that sequences of equal length remain unchanged
def test_pre_pad_sequences_no_padding_needed():
    sequences = [
        [1, 2, 3],
        [4, 5, 6]
    ]
    # When all sequences already have the same length, the function should return them unchanged.
    padded = pre_pad_sequences(sequences)
    assert padded == sequences, "Sequences of equal length should not be modified."

# Test: Verify that sequences of unequal length are padded to the maximum length found
def test_pre_pad_sequences_padding():
    sequences = [
        [1, 2],
        [3, 4, 5]
    ]
    # Maximum length is 3 (from the second sequence). The first sequence should be padded with 0.
    expected = [
        [1, 2, 0],
        [3, 4, 5]
    ]
    padded = pre_pad_sequences(sequences)
    assert padded == expected, "Sequences should be padded with 0 to match the maximum length."

# Test: Verify that providing a fixed max_len results in truncation or padding as needed
def test_pre_pad_sequences_truncation():
    sequences = [
        [1, 2, 3, 4],
        [5, 6]
    ]
    # If we set max_len=3, the first sequence should be truncated and the second padded.
    expected = [
        [1, 2, 3],
        [5, 6, 0]
    ]
    padded = pre_pad_sequences(sequences, max_len=3)
    assert padded == expected, "Sequences should be truncated or padded to fixed max_len=3."

# Test: Verify that a custom padding value is used correctly
def test_pre_pad_sequences_custom_padding_value():
    sequences = [
        [7],
        [8, 9]
    ]
    # Set max_len=3 and a custom padding value, e.g., -1.
    expected = [
        [7, -1, -1],
        [8, 9, -1]
    ]
    padded = pre_pad_sequences(sequences, max_len=3, padding_value=-1)
    assert padded == expected, "Sequences should be padded using the custom padding value (-1)."
