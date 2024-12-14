import pytest

from app.repositories.memory import (
    save_model,
    get_model,
    model_exists,
    update_model,
    get_all_models,
    save_search_result,
    get_search_result,
    clear_search_buffer,
)

@pytest.fixture
def setup_models():
    # Reset models and search buffer before each test
    from app.repositories.memory import models, search_buffer
    models.clear()
    search_buffer.clear()


def test_save_and_get_model(setup_models):
    # Test saving and retrieving a model
    save_model("model1", {"key": "value"})
    model = get_model("model1")
    assert model == {"key": "value"}, "The saved and retrieved model does not match"


def test_model_exists(setup_models):
    # Test checking if a model exists
    save_model("model1", {"key": "value"})
    assert model_exists("model1"), "The model should exist"
    assert not model_exists("model2"), "A non-existent model should not be found"


def test_update_model(setup_models):
    # Test updating an existing model
    save_model("model1", {"key": "value"})
    update_model("model1", {"new_key": "new_value"})
    model = get_model("model1")
    assert model == {"key": "value", "new_key": "new_value"}, "Model update failed"


def test_get_all_models(setup_models):
    # Test retrieving all models
    save_model("model1", {"key": "value"})
    save_model("model2", {"key": "another_value"})
    models = get_all_models()
    assert len(models) == 2, "The number of retrieved models is incorrect"
    assert "model1" in models and "model2" in models, "Not all saved models are present"


def test_save_and_get_search_result(setup_models):
    # Test saving and retrieving search results
    save_search_result("model1", "query1", {"result": "value"})
    result = get_search_result("model1", "query1")
    assert result == {"result": "value"}, "The saved and retrieved search result does not match"

    # Test retrieving a non-existent search result
    result = get_search_result("model1", "query2")
    assert result is None, "A non-existent search result should not be found"


def test_clear_search_buffer(setup_models):
    # Test clearing the search buffer
    save_search_result("model1", "query1", {"result": "value"})
    save_search_result("model2", "query2", {"result": "another_value"})

    # Clear the buffer for a specific model
    clear_search_buffer("model1")

    result = get_search_result("model1", "query1")
    assert result is None, "The buffer for model1 should have been cleared"

    result = get_search_result("model2", "query2")
    assert result == {"result": "another_value"}, "The buffer for model2 should not have been cleared"


def test_buffer_integration_with_model(setup_models):
    # Test integration between models and the buffer
    save_model("model1", {"key": "value"})
    save_search_result("model1", "query1", {"result": "value"})

    # Ensure the model and buffer work together
    assert model_exists("model1"), "The model should exist"
    cached_result = get_search_result("model1", "query1")
    assert cached_result == {"result": "value"}, "The cached search result should have been found"
