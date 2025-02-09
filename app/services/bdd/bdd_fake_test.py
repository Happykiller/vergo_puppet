# app\services\bdd\bdd_fake_test.py
import pytest
from datetime import datetime, timezone

from app.services.bdd.bdd_fake import FakeBDDService
from app.services.bdd.models.model_data import ModelData
from app.services.bdd.models.model_metrics import MetricsModel

@pytest.fixture
def fake_bdd():
    """Fixture for initializing the FakeBDDService."""
    return FakeBDDService()

def test_save_and_get_model(fake_bdd):
    """Test saving and retrieving a model."""
    model = ModelData(name="test_model", neural_network_type="SimpleNN")
    fake_bdd.save_model(model)

    retrieved_model = fake_bdd.get_model("test_model")

    assert retrieved_model is not None
    assert retrieved_model.name == model.name
    assert retrieved_model.neural_network_type == model.neural_network_type

def test_model_exists(fake_bdd):
    """Test checking if a model exists."""
    model = ModelData(name="existing_model", neural_network_type="LSTM")
    fake_bdd.save_model(model)

    assert fake_bdd.model_exists("existing_model")
    assert not fake_bdd.model_exists("non_existing_model")

def test_update_model(fake_bdd):
    """Test updating a model's data."""
    model = ModelData(name="updatable_model", neural_network_type="GRU")
    fake_bdd.save_model(model)

    updated_model = ModelData(name="updatable_model", neural_network_type="LSTM")
    fake_bdd.update_model(updated_model)

    retrieved_model = fake_bdd.get_model("updatable_model")
    assert retrieved_model.neural_network_type == "LSTM"

def test_get_all_models(fake_bdd):
    """Test retrieving all models from the fake database."""
    model1 = ModelData(name="model1", neural_network_type="SimpleNN")
    model2 = ModelData(name="model2", neural_network_type="LSTM")

    fake_bdd.save_model(model1)
    fake_bdd.save_model(model2)

    all_models = fake_bdd.get_all_models()
    assert len(all_models) == 2
    assert any(model.name == "model1" for model in all_models)
    assert any(model.name == "model2" for model in all_models)

def test_save_and_get_search_result(fake_bdd):
    """Test saving and retrieving search results."""
    model_name = "search_model"
    query = "find_best_match"
    result = {"match": "example_result"}

    fake_bdd.save_search_result(model_name, query, result)
    retrieved_result = fake_bdd.get_search_result(model_name, query)

    assert retrieved_result == result

def test_clear_search_buffer(fake_bdd):
    """Test clearing the search buffer for a specific model."""
    model_name = "clear_test"
    query = "query1"
    result = {"data": "some_result"}

    fake_bdd.save_search_result(model_name, query, result)
    fake_bdd.clear_search_buffer(model_name)

    assert fake_bdd.get_search_result(model_name, query) is None

def test_save_and_get_metrics(fake_bdd):
    """Test saving and retrieving training results."""
    training_result1 = MetricsModel(model_name="model1", metrics={"accuracy": 0.95}, timestamp=datetime.now(timezone.utc))
    training_result2 = MetricsModel(model_name="model2", metrics={"accuracy": 0.98}, timestamp=datetime.now(timezone.utc))

    fake_bdd.save_metrics(training_result1)
    fake_bdd.save_metrics(training_result2)

    all_results = fake_bdd.get_metrics()
    assert len(all_results) == 2

    filtered_results = fake_bdd.get_metrics("model1")
    assert len(filtered_results) == 1
    assert filtered_results[0].model_name == "model1"
