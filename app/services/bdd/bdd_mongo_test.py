# app\services\bdd\bdd_mongo_test.py
import pytest
import mongomock
from datetime import datetime, timezone

from app.services.bdd.bdd_mongo import MongoBDDService
from app.services.bdd.models.model_data import ModelData
from app.services.bdd.models.training_result import TrainingResult

@pytest.fixture
def mongo_bdd_service():
    """Fixture for initializing MongoBDDService with a mocked in-memory MongoDB instance."""
    mock_client = mongomock.MongoClient()
    mock_db = mock_client["test_db"]

    service = MongoBDDService(mongo_client=mock_client, database_name="test_db")
    service.database = mock_db

    return service

def test_save_and_get_model(mongo_bdd_service):
    """Test saving and retrieving a model in MongoDB."""
    model_data = ModelData(name="test_model", neural_network_type="SimpleNN")

    mongo_bdd_service.save_model(model_data)
    retrieved_model = mongo_bdd_service.get_model("test_model")

    assert retrieved_model is not None
    assert retrieved_model.name == "test_model"
    assert retrieved_model.neural_network_type == "SimpleNN"

def test_model_exists(mongo_bdd_service):
    """Test checking if a model exists in MongoDB."""
    model_data = ModelData(name="existing_model", neural_network_type="LSTM")
    mongo_bdd_service.save_model(model_data)

    assert mongo_bdd_service.model_exists("existing_model")
    assert not mongo_bdd_service.model_exists("non_existing_model")

def test_update_model(mongo_bdd_service):
    """Test updating an existing model in MongoDB."""
    model_data = ModelData(name="update_model", neural_network_type="LSTM")
    mongo_bdd_service.save_model(model_data)

    updated_model = ModelData(name="update_model", neural_network_type="LSTM")
    mongo_bdd_service.update_model(updated_model)

    retrieved_model = mongo_bdd_service.get_model("update_model")
    assert retrieved_model is not None
    assert retrieved_model.name == "update_model"

def test_get_all_models(mongo_bdd_service):
    """Test retrieving all models from MongoDB."""
    model1 = ModelData(name="model1", neural_network_type="GRU")
    model2 = ModelData(name="model2", neural_network_type="LSTM")

    mongo_bdd_service.save_model(model1)
    mongo_bdd_service.save_model(model2)

    models = mongo_bdd_service.get_all_models()
    assert len(models) == 2
    assert models[0].name in ["model1", "model2"]
    assert models[1].name in ["model1", "model2"]

def test_save_and_get_training_results(mongo_bdd_service):
    """Test saving and retrieving training results from MongoDB."""
    training_result = TrainingResult(
        model_name="test_model",
        metrics={"accuracy": 0.95},
        timestamp=datetime.now(timezone.utc),
    )

    mongo_bdd_service.save_training_result(training_result)
    retrieved_results = mongo_bdd_service.get_training_results("test_model")

    assert len(retrieved_results) == 1
    assert retrieved_results[0].model_name == "test_model"
    assert retrieved_results[0].metrics["accuracy"] == 0.95
