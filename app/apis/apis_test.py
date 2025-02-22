# app\apis\apis_test.py
from app.services.bdd.models.model_data import ModelData
import pytest
from fastapi.testclient import TestClient # type: ignore 

from app.main import app
from unittest.mock import mock_open, patch
from app.generate_token import create_token

# Initialize test client for making requests to the API
client = TestClient(app)

# Sample data for testing
dictionary = [
    ["token1", "token2", "token3"], 
    ["token1", "token2", "token3", "token4"], 
    ["token1", "token2", "token5"]
]
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
        "dictionary": dictionary
    }
    with patch("app.apis.apis.create_model_siamese", return_value={"status": "model created", "model_name": "model1", "missing_tokens": ["broomstick"]}):
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
    with patch("app.apis.apis.train_model_siamese", return_value={"status": "training completed", "model_name": "model1", "training_report": { "total_epochs": 221, "possible_epochs": 1000, "total_time": 119.67791843414307, "final_loss": 0.00016467317659847158, "best_loss": 0.0001383969918023004, "num_parameters": 460672}}):
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

    with patch("app.apis.apis.prepare_cache_siamese", return_value={
  "status": "cache prepared",
  "model_name": "model1",
  "vectors_processed": 2,
  "results": [
    {
      "search_vector": [
        "token1",
        "token2"
      ],
      "result": {
        "search": [
          "token1",
          "token2"
        ],
        "find": [
          "token1",
          "token2"
        ],
        "stats": {
          "accuracy": 0.942383348941803
        }
      }
    },
    {
      "search_vector": [
        "token3",
        "token4"
      ],
      "result": {
        "search": [
          "token3",
          "token4"
        ],
        "find": [
          "token3",
          "token4"
        ],
        "stats": {
          "accuracy": 0.9148389399051666
        }
      }
    }
  ]
}):
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

        assert response_data["results"][1]["search_vector"] == ["token3", "token4"]

# Test for the model search API
def test_search_model(mocked_env_vars, get_headers):
    # Data for performing a search query within the model
    data = {
        "name": "model1",
        "neural_network_type": "SIAMESE",
        "vector": search_vector
    }

    with patch("app.apis.apis.search_model_siamese", return_value={
  "search": [
    "token1",
    "token2",
    "token3"
  ],
  "find": [
    "token1",
    "token2",
    "token3"
  ],
  "stats": {
    "accuracy": 0.9253618717193604
  }
}):
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
        "dictionary": dictionary
    }

    with patch("app.apis.apis.update_model_siamese", return_value={"status": "model updated", "model_name": "model1"}):
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

# Test: Get all models
def test_get_models(mocked_env_vars, get_headers):
    with patch("app.apis.apis.get_all_models_usecase", return_value=[]):
        response = client.get("/models", headers=get_headers)
        assert response.status_code == 200

# Test: Tokenize
def test_tokenize(mocked_env_vars, get_headers):
    data = {
        "data": [
            {"incidentId": "INC123", "description": "Hello world"},
            {"incidentId": "INC124", "description": "This is a test incident"}
        ]
    }
    with patch("app.apis.apis.usecase_tokenize", return_value={"tokens": ["hello", "world"]}):
        response = client.post("/tokenize", json=data, headers=get_headers)
        assert response.status_code == 200

# Test: Secure endpoint access
def test_secure_endpoint(mocked_env_vars, get_headers):
    response = client.post("/secure_endpoint", headers=get_headers)
    assert response.status_code == 200

# Test: Secure endpoint without token
def test_secure_endpoint_without_token():
    response = client.post("/secure_endpoint")
    assert response.status_code == 401

# Test: Create data for Puppet-O4
def test_create_data_puppet_o4(mocked_env_vars, get_headers):
    with patch("app.apis.apis.usecase_create_data_puppeto4", return_value="data_created"):
        response = client.get("/create_data_puppet-o4", headers=get_headers)
        assert response.status_code == 200

# Error case: Unknown neural network type
def test_create_model_with_invalid_type(mocked_env_vars, get_headers):
    data = {"name": "model1", "neural_network_type": "UNKNOWN"}
    response = client.post("/create_model", json=data, headers=get_headers)
    assert response.status_code == 500

def test_train_model_from_file(mocked_env_vars, get_headers):
    data = {"name": "model1", "neural_network_type": "SIAMESE", "file_name": "test_data.json"}
    with patch("app.apis.apis.train_model_from_file_background", return_value=None), \
         patch("app.apis.apis.get_model_usecase", return_value=ModelData(name='test', neural_network_type="LSTM")), \
         patch("pathlib.Path.exists", return_value=True):
        response = client.post("/train_model_from_file", json=data, headers=get_headers)
        assert response.status_code == 200

def test_train_model_from_file_missing(mocked_env_vars, get_headers):
    data = {"name": "model1", "neural_network_type": "SIAMESE", "file_name": "missing.json"}
    with patch("pathlib.Path.exists", return_value=False):
        response = client.post("/train_model_from_file", json=data, headers=get_headers)
        assert response.status_code == 500

# Test: Super train model
def test_super_train_model(mocked_env_vars, get_headers):
    data = {"name": "model1", "neural_network_type": "SIAMESE", "train_file": "test_data.json", "test_data": []}
    with patch("app.apis.apis.super_train_model_background", return_value=None), \
         patch("app.apis.apis.get_model_usecase", return_value=ModelData(name='test', neural_network_type="LSTM")), \
         patch("app.apis.apis.parse_input_data", return_value=[]):
        response = client.post("/super_train_model", json=data, headers=get_headers)
        assert response.status_code in [200, 404]

# Test: Search brut multi
def test_search_brut_multi(mocked_env_vars, get_headers):
    data = {"name": "model1", "neural_network_type": "GRU", "documents": [
    {
      "incidentId": "1",
      "description": "Pourriez vous le creer les acces. Merci"
    }]}
    with patch("app.apis.apis.search_multi_brut_model_gru", return_value={"status": "success", "results": []}):
        response = client.post("/search_brut_multi", json=data, headers=get_headers)
        assert response.status_code == 200
        assert response.json()["status"] == "success"