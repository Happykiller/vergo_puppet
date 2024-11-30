# app\usecases\siamese\prepare_cache_siamese_test.py
import pytest
from unittest.mock import patch
from app.repositories.memory import save_model, models
from app.usecases.siamese.prepare_cache_siamese import prepare_cache

@pytest.fixture(autouse=True)
def reset_memory():
    """
    Reset the in-memory storage before each test.
    """
    models.clear()
    save_model("mock_model", {
        "nn_model": "mock_nn_model",
        "dictionary": [["token1", "token2"], ["token3", "token4"]],
        "glossary": ["token1", "token2", "token3", "token4"]
    })

@patch("app.usecases.siamese.prepare_cache_siamese.search_model_siamese")
def test_prepare_cache_success(mock_search_model_siamese):
    """
    Test successful preparation of the cache for a SIAMESE model.
    """
    # Mock the behavior of search_model_siamese
    mock_search_model_siamese.side_effect = lambda name, vector: {"search": vector, "find": vector, "stats": {"accuracy": 0.9}}

    search_vectors = [["token1", "token2"], ["token3", "token4"]]

    # Prepare the cache
    response = prepare_cache("mock_model", search_vectors)

    # Validate the response
    assert response["status"] == "cache prepared"
    assert response["model_name"] == "mock_model"
    assert response["vectors_processed"] == len(search_vectors)

    # Validate results
    assert len(response["results"]) == len(search_vectors)
    for i, vector in enumerate(search_vectors):
        result = response["results"][i]
        assert result["search_vector"] == vector
        assert "result" in result
        assert result["result"] == {"search": vector, "find": vector, "stats": {"accuracy": 0.9}}

    # Validate that the mock was called
    mock_search_model_siamese.assert_called()

@patch("app.usecases.siamese.prepare_cache_siamese.search_model_siamese")
def test_prepare_cache_model_not_found(mock_search_model_siamese):
    """
    Test cache preparation with a non-existent model raises an error.
    """
    # Mock behavior: raise an exception when the model is not found
    mock_search_model_siamese.side_effect = Exception("Model not found")

    with pytest.raises(Exception, match="Model not found"):
        prepare_cache("non_existent_model", [["token1", "token2"]])

    # Ensure the mock was not called with any vectors
    mock_search_model_siamese.assert_not_called()

@patch("app.usecases.siamese.prepare_cache_siamese.search_model_siamese")
def test_prepare_cache_with_empty_vectors(mock_search_model_siamese):
    """
    Test cache preparation with an empty list of search vectors.
    """
    response = prepare_cache("mock_model", [])

    # Validate the response
    assert response["status"] == "cache prepared"
    assert response["model_name"] == "mock_model"
    assert response["vectors_processed"] == 0
    assert response["results"] == []

    # Ensure the mock was not called
    mock_search_model_siamese.assert_not_called()

@patch("app.usecases.siamese.prepare_cache_siamese.search_model_siamese")
def test_prepare_cache_with_invalid_vector(mock_search_model_siamese):
    """
    Test cache preparation with a search vector that causes an error.
    """
    # Mock behavior: raise an exception for invalid vectors
    def mock_side_effect(name, vector):
        if "invalid_token" in vector:
            raise Exception("Invalid vector")
        return {"search": vector, "find": vector, "stats": {"accuracy": 0.9}}
    
    mock_search_model_siamese.side_effect = mock_side_effect

    search_vectors = [["token1", "token2"], ["invalid_token", "token4"]]

    response = prepare_cache("mock_model", search_vectors)

    # Validate the response
    assert response["status"] == "cache prepared"
    assert response["model_name"] == "mock_model"
    assert response["vectors_processed"] == len(search_vectors)

    for result in response["results"]:
        if result["search_vector"] == ["invalid_token", "token4"]:
            assert "error" in result
            assert result["error"] == "Invalid vector"
        else:
            assert "result" in result
            assert result["result"] == {"search": ["token1", "token2"], "find": ["token1", "token2"], "stats": {"accuracy": 0.9}}

    # Validate that the mock was called the correct number of times
    assert mock_search_model_siamese.call_count == len(search_vectors)
