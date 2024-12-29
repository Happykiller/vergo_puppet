# app\apis\apis_test.py
import pytest
from fastapi.testclient import TestClient # type: ignore 

from app.main import app
from unittest.mock import patch
from app.generate_token import create_token

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

@pytest.fixture(scope="module", autouse=True)
def mocked_env_vars():
    # Mock environment variables
    with patch("app.apis.apis.load_env_vars", return_value={
        "secret_key": "mocked-secret-key",
        "mode": "test"
    }), patch("app.generate_token.load_env_vars", return_value={
        "secret_key": "mocked-secret-key",
        "mode": "test"
    }):
        yield

@pytest.fixture(scope="module")
def get_headers(mocked_env_vars):
    # Génération d'un token JWT pour un utilisateur fictif
    token = create_token(user_id="test_user")
    # Retourne les en-têtes nécessaires
    return {"Authorization": f"Bearer {token}"}

# Test for the model creation API
def test_create_model(mocked_env_vars, get_headers):
    # Data to be sent to the API for creating a new model
    data = {
        "name": "model1",
        "neural_network_type": "SIAMESE",
        "dictionary": dictionary,
        "glossary": glossary
    }
    # Send POST request to /create_model endpoint
    response = client.post("/create_model", json=data, headers=get_headers)
    # Assert that the response is successful (status code 200)
    assert response.status_code == 200, f"Error during model creation: {response.text}"
    # Verify that the expected response structure and content are returned
    assert response.json()['status'] == "model created"

# Test for the model training API
def test_train_model(mocked_env_vars, get_headers):
    # Data to be sent to the API to train an existing model
    data = {
        "name": "model1",
        "neural_network_type": "SIAMESE",
        "training_data": training_data
    }
    # Send POST request to /train_model endpoint
    response = client.post("/train_model", json=data, headers=get_headers)
    # Assert that the response is successful (status code 200)
    assert response.status_code == 200, f"Error during model training: {response.text}"
    # Verify that the response indicates training has completed
    assert "training completed" in response.json()["status"]

def test_prepare_cache_with_results(mocked_env_vars, get_headers):
    # Input data
    data = {
        "name": "model1",
        "neural_network_type": "SIAMESE",
        "search_vectors": [["token1", "token2"], ["token3", "token4"]]
    }

    # Send request to the /prepare_cache endpoint
    response = client.post("/prepare_cache", json=data, headers=get_headers)
    assert response.status_code == 200, f"Error during cache preparation: {response.text}"
    response_data = response.json()

    # Validate response structure and results
    assert response_data["status"] == "cache prepared"
    assert response_data["model_name"] == "model1"
    assert response_data["vectors_processed"] == 2
    assert len(response_data["results"]) == 2

    # Validate individual results
    assert response_data["results"][0]["search_vector"] == ["token1", "token2"]
    assert "result" in response_data["results"][0]

    assert response_data["results"][1]["search_vector"] == ["token3", "token4"]
    assert "result" in response_data["results"][1]

# Test for the model search API
def test_search_model(mocked_env_vars, get_headers):
    # Data for performing a search query within the model
    data = {
        "name": "model1",
        "neural_network_type": "SIAMESE",
        "vector": search_vector
    }
    # Send POST request to /search endpoint
    response = client.post("/search", json=data, headers=get_headers)
    # Assert that the response is successful (status code 200)
    assert response.status_code == 200, f"Error during model search: {response.text}"
    # Verify response includes expected search output structure
    result = response.json()
    assert result["search"] == search_vector
    assert "find" in result  # Check for presence of a 'find' key in the response
    assert "stats" in result # Check for presence of a 'stats' key in the response

def test_update_model(mocked_env_vars, get_headers):
    # Data for updating the model
    update_data = {
        "name": "model1",
        "neural_network_type": "SIAMESE",
        "dictionary": dictionary,
        "glossary": glossary
    }

    # Send a PATCH request to the /update_model endpoint
    response = client.patch("/update_model", json=update_data, headers=get_headers)
    
    # Assert that the response is successful (status code 200)
    assert response.status_code == 200, f"Error during model update: {response.text}"
    
    # Validate the response content
    response_data = response.json()
    assert response_data["status"] == "model updated"
    assert response_data["model_name"] == "model1"

# Test for the API version endpoint
def test_get_version():
    # Send GET request to /version endpoint to retrieve the current API version
    response = client.get("/version")
    # Assert that the response is successful (status code 200)
    assert response.status_code == 200, f"Error retrieving API version: {response.text}"
    # Verify response includes version information
    version_info = response.json()
    assert "version" in version_info