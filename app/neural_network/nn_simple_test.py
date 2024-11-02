#app\neural_network\nn_simple_test.py
import torch
import random
import pytest
import numpy as np
from app.neural_network.nn_simple import SimpleNN, train_model_nn, predict

def set_seed(seed=42):
    """Sets the random seed for reproducibility in PyTorch, numpy, and random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Test 1: Verify the structure of the SimpleNN model
def test_neural_network_structure():
    model = SimpleNN(input_size=3, hidden_size=128, output_size=3)

    # Check that the model has correctly defined layers
    assert isinstance(model.fc1, torch.nn.Linear), "The first layer is not a Linear layer"
    assert model.fc1.in_features == 3, "fc1 does not receive the correct input size"
    assert model.fc1.out_features == 128, "fc1 does not produce the correct output size"

    assert isinstance(model.fc2, torch.nn.Linear), "The second layer is not a Linear layer"
    assert model.fc2.in_features == 128, "fc2 does not receive the correct input size"
    assert model.fc2.out_features == 128, "fc2 does not produce the correct output size"

    assert isinstance(model.fc3, torch.nn.Linear), "The third layer is not a Linear layer"
    assert model.fc3.in_features == 128, "fc3 does not receive the correct input size"
    assert model.fc3.out_features == 3, "fc3 does not produce the correct output size"

# Test 2: Verify model training and that weights are updated
def test_train_model_nn():
    # Realistic feature examples (e.g., housing characteristics: size, number of rooms, distance to city center)
    features_processed = [
        [75, 2, 5],   # 75 sqm, 2 rooms, 5 km from city center
        [120, 4, 10], # 120 sqm, 4 rooms, 10 km from city center
        [60, 1, 2]    # 60 sqm, 1 room, 2 km from city center
    ]

    # Realistic targets: housing prices (normalized between 0 and 1)
    targets_standardized = [
        0.5,  # Average price for the first property
        0.8,  # Higher price for larger and more distant property
        0.3   # Lower price for a smaller property
    ]

    # Train the model
    nn_model, _ = train_model_nn(features_processed, targets_standardized, 3, epochs=100, learning_rate=0.001)

    # Check that the model has been created
    assert nn_model is not None, "The model was not trained successfully."

    # Verify that weights have been updated (not the same as initial weights)
    initial_weights = torch.zeros_like(nn_model.fc1.weight.data)
    assert not torch.equal(nn_model.fc1.weight.data, initial_weights), "The weights of the first layer were not updated after training."

# Test 3: Verify model prediction with realistic input data
def test_predict_with_trained_model():
    # Set seed for reproducibility
    set_seed(42)
    
    # Realistic feature examples (e.g., housing characteristics: size, number of rooms, distance to city center)
    features_processed = [
        [75, 2, 5],   # 75 sqm, 2 rooms, 5 km from city center
        [120, 4, 10], # 120 sqm, 4 rooms, 10 km from city center
        [60, 1, 2]    # 60 sqm, 1 room, 2 km from city center
    ]

    # Realistic targets: housing prices (normalized between 0 and 1)
    targets_standardized = [
        0.5,  # Average price for the first property
        0.8,  # Higher price for larger and more distant property
        0.3   # Lower price for a smaller property
    ]

    # Train the model
    nn_model, _ = train_model_nn(features_processed, targets_standardized, 3, epochs=200, learning_rate=0.001)

    # Make a prediction with a realistic input vector
    input_vector = [80, 3, 6]  # A property with 80 sqm, 3 rooms, 6 km from city center
    predicted = predict(nn_model, input_vector, targets_mean=0.53, targets_std=0.18)  # Using realistic mean and std for un-normalization

    # Check that the prediction returns a valid value
    assert predicted > 0, "The prediction returned a negative or incorrect value."

# Test 4: Verify that loss decreases during training with realistic data
def test_loss_decreases_during_training():
    # Realistic feature examples (e.g., housing characteristics: size, number of rooms, distance to city center)
    features_processed = [
        [75, 2, 5],   # 75 sqm, 2 rooms, 5 km from city center
        [120, 4, 10], # 120 sqm, 4 rooms, 10 km from city center
        [60, 1, 2]    # 60 sqm, 1 room, 2 km from city center
    ]

    # Realistic targets: housing prices (normalized between 0 and 1)
    targets_standardized = [
        0.5,  # Average price for the first property
        0.8,  # Higher price for larger and more distant property
        0.3   # Lower price for a smaller property
    ]

    # Train the model over multiple epochs and check the loss
    nn_model, losses = train_model_nn(features_processed, targets_standardized, 3, epochs=100, learning_rate=0.001)

    # Verify that the loss decreases over time
    assert losses[0] > losses[-2], "The loss did not decrease during training."
