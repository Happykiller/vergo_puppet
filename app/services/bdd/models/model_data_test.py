# app\services\bdd\models\model_data_test.py
import torch
import joblib
import pytest
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.services.bdd.models.model_data import ModelData

class DummyModel(torch.nn.Module):
    """A dummy PyTorch model for testing purposes."""
    def __init__(self, input_size=10, hidden_size=20, output_size=1):
        super(DummyModel, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size)
        # Simulate the args attribute from SimpleNN
        self.args = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "output_size": output_size,
        }

    def forward(self, x):
        return self.fc(x)

def test_model_data_creation_with_basic_fields():
    """Test creating a ModelData instance with basic fields only."""
    model_data = ModelData(
        name="test_model",
        neural_network_type="SimpleNN",
        dictionary=[["word1", "word2"]],
        indexed_dictionary=[[1, 2]],
        glossary=["word1", "word2"]
    )

    serialized_data = model_data.serialize()

    # Validate serialized fields
    assert serialized_data["name"] == "test_model"
    assert serialized_data["neural_network_type"] == "SimpleNN"
    assert serialized_data["dictionary"] == [["word1", "word2"]]
    assert serialized_data["indexed_dictionary"] == [[1, 2]]
    assert serialized_data["glossary"] == ["word1", "word2"]
    assert serialized_data["nn_model"] is None
    assert serialized_data["encoder"] is None
    assert serialized_data["scaler"] is None
    assert serialized_data["indices"] is None


def test_model_data_retrieval_for_training():
    """Test retrieving a ModelData instance prepared for training."""
    dummy_model = DummyModel()
    encoder = OneHotEncoder()
    scaler = StandardScaler()
    indices = {"categorical_indices": [0, 1], "numerical_indices": [2, 3]}

    model_data = ModelData(
        name="test_model",
        neural_network_type="SimpleNN",
        nn_model=dummy_model,
        encoder=encoder,
        scaler=scaler,
        indices=indices,
        targets_mean=0.5,
        targets_std=0.2
    )

    serialized_data = model_data.serialize()

    # Deserialize the serialized data
    deserialized_model_data = ModelData.deserialize(serialized_data, model_class=DummyModel)

    # Validate deserialized fields
    assert deserialized_model_data.name == "test_model"
    assert deserialized_model_data.neural_network_type == "SimpleNN"
    assert deserialized_model_data.nn_model is not None
    assert isinstance(deserialized_model_data.nn_model, DummyModel)
    assert isinstance(deserialized_model_data.encoder, OneHotEncoder)
    assert isinstance(deserialized_model_data.scaler, StandardScaler)
    assert deserialized_model_data.indices == {"categorical_indices": [0, 1], "numerical_indices": [2, 3]}
    assert deserialized_model_data.targets_mean == 0.5
    assert deserialized_model_data.targets_std == 0.2


def test_model_data_update_after_training():
    """Test updating a ModelData instance after training."""
    initial_model_data = ModelData(
        name="test_model",
        neural_network_type="SimpleNN"
    )

    initial_serialized_data = initial_model_data.serialize()

    # Validate initial state
    assert initial_serialized_data["nn_model"] is None
    assert initial_serialized_data["encoder"] is None
    assert initial_serialized_data["scaler"] is None
    assert initial_serialized_data["indices"] is None

    # Create objects after training
    dummy_model = DummyModel()
    encoder = OneHotEncoder()
    scaler = StandardScaler()
    indices = {"categorical_indices": [0, 1], "numerical_indices": [2, 3]}

    # Update the model with post-training data
    updated_model_data = ModelData(
        name="test_model",
        neural_network_type="SimpleNN",
        nn_model=dummy_model,
        encoder=encoder,
        scaler=scaler,
        indices=indices,
        targets_mean=0.5,
        targets_std=0.2
    )

    updated_serialized_data = updated_model_data.serialize()

    # Deserialize the updated data
    deserialized_model_data = ModelData.deserialize(updated_serialized_data, model_class=DummyModel)

    # Validate updated fields
    assert deserialized_model_data.name == "test_model"
    assert deserialized_model_data.nn_model is not None
    assert isinstance(deserialized_model_data.nn_model, DummyModel)
    assert isinstance(deserialized_model_data.encoder, OneHotEncoder)
    assert isinstance(deserialized_model_data.scaler, StandardScaler)
    assert deserialized_model_data.indices == {"categorical_indices": [0, 1], "numerical_indices": [2, 3]}
    assert deserialized_model_data.targets_mean == 0.5
    assert deserialized_model_data.targets_std == 0.2


def test_serialization_with_missing_optional_fields():
    """Test serialization and deserialization when optional fields are missing."""
    model_data = ModelData(
        name="test_model",
        neural_network_type="SimpleNN"
    )

    serialized_data = model_data.serialize()

    # Validate that optional fields are None in the serialized output
    assert serialized_data["dictionary"] is None
    assert serialized_data["indexed_dictionary"] is None
    assert serialized_data["glossary"] is None
    assert serialized_data["nn_model"] is None

    # Deserialize and validate fields
    deserialized_model_data = ModelData.deserialize(serialized_data)
    assert deserialized_model_data.dictionary is None
    assert deserialized_model_data.indexed_dictionary is None
    assert deserialized_model_data.glossary is None
    assert deserialized_model_data.nn_model is None


def test_invalid_deserialization_should_raise_exception():
    """Test that deserialization raises an exception for invalid input."""
    invalid_serialized_data = {
        "name": "test_model",
        "neural_network_type": "SimpleNN",
        "encoder": "invalid_data"  # Incorrect serialization
    }

    with pytest.raises(Exception):
        ModelData.deserialize(invalid_serialized_data, model_class=DummyModel)


def test_serialization_with_empty_dictionary():
    """Test serialization and deserialization with an empty dictionary."""
    model_data = ModelData(
        name="test_model",
        neural_network_type="SimpleNN",
        dictionary=[],
        glossary=[]
    )

    serialized_data = model_data.serialize()

    # Validate that dictionary and glossary are serialized as empty lists
    assert serialized_data["dictionary"] == []
    assert serialized_data["glossary"] == []

    # Deserialize and validate fields
    deserialized_model_data = ModelData.deserialize(serialized_data)
    assert deserialized_model_data.dictionary == []
    assert deserialized_model_data.glossary == []