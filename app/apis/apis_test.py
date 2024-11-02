# app\apis\apis_test.py
import pytest
from fastapi.testclient import TestClient   # type: ignore  # Import FastAPI test client to simulate HTTP requests
from app.main import app

# Initialize test client for making requests to the API
client = TestClient(app)

# Sample data for testing
dictionary = [
    ["token1", "token2", "token3"], 
    ["token1", "token2", "token3", "token4"], 
    ["token1", "token2", "token5"]
]
glossary = ["token1", "token2", "token3", "token4", "token5"]
search_vector = ["token1", "token2", "token3"]
training_data = [
    [["token1", "token2", "token3"], ["token2", "token3", "token4"], 0.66],
    [["token1", "token2", "token4"], ["token2", "token4", "token5"], 0.33],
    [["token1", "token4"], ["token4", "token5"], 0],
    [["token1", "token2", "token5"], ["token2", "token5", "token1"], 0.5],
    [["token1", "token2", "token3", "token4"], ["token2", "token3", "token4", "token5"], 0.75]
]

# Test for the model creation API
def test_create_model():
    # Data to be sent to the API for creating a new model
    data = {
        "name": "model1",
        "neural_network_type": "SIAMESE",
        "dictionary": dictionary,
        "glossary": glossary
    }
    # Send POST request to /create_model endpoint
    response = client.post("/create_model", json=data)
    # Assert that the response is successful (status code 200)
    assert response.status_code == 200, f"Error during model creation: {response.text}"
    # Verify that the expected response structure and content are returned
    assert response.json() == {"status": "model created", "model_name": "model1"}

# Test for the model training API
def test_train_model():
    # Data to be sent to the API to train an existing model
    data = {
        "name": "model1",
        "neural_network_type": "SIAMESE",
        "training_data": training_data
    }
    # Send POST request to /train_model endpoint
    response = client.post("/train_model", json=data)
    # Assert that the response is successful (status code 200)
    assert response.status_code == 200, f"Error during model training: {response.text}"
    # Verify that the response indicates training has completed
    assert "training completed" in response.json()["status"]

# Test for the model search API
def test_search_model():
    # Data for performing a search query within the model
    data = {
        "name": "model1",
        "neural_network_type": "SIAMESE",
        "vector": search_vector
    }
    # Send POST request to /search endpoint
    response = client.post("/search", json=data)
    # Assert that the response is successful (status code 200)
    assert response.status_code == 200, f"Error during model search: {response.text}"
    # Verify response includes expected search output structure
    result = response.json()
    assert result["search"] == search_vector
    assert "find" in result  # Check for presence of a 'find' key in the response
    assert "stats" in result # Check for presence of a 'stats' key in the response

# Test for the API version endpoint
def test_get_version():
    # Send GET request to /version endpoint to retrieve the current API version
    response = client.get("/version")
    # Assert that the response is successful (status code 200)
    assert response.status_code == 200, f"Error retrieving API version: {response.text}"
    # Verify response includes version information
    version_info = response.json()
    assert "version" in version_info
