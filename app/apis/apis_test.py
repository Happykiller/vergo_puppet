import pytest
from fastapi.testclient import TestClient # type: ignore
from app.main import app  # Assurez-vous que 'app' est l'application principale FastAPI

# Initialiser le client de test
client = TestClient(app)

# Données de test
dictionary = [
    ["token1", "token2", "token3"], 
    ["token1", "token2", "token3", "token4"], 
    ["token1", "token2", "token5"]
]
glossary = ["token1", "token2", "token3", "token4", "token5"]
search_vector = ["token1", "token2", "token3"]
training_data = [
    [["token1", "token2", "token3"], ["token2", "token3", "token4"]],
    [["token1", "token2", "token4"], ["token2", "token4", "token5"]],
    [["token1", "token4"], ["token4", "token5"]],
    [["token1", "token2", "token5"], ["token2", "token5", "token1"]],
    [["token1", "token2", "token3", "token4"], ["token2", "token3", "token4", "token5"]]
]

# Test de l'API de création de modèle
def test_create_model():
    data = {
        "name": "model1",
        "neural_network_type": "LSTMNN",
        "dictionary": dictionary,
        "glossary": glossary
    }
    response = client.post("/create_model", json=data)
    assert response.status_code == 200, f"Erreur lors de la création du modèle : {response.text}"
    assert response.json() == {"status": "model created", "model_name": "model1"}

# Test de l'API d'entraînement du modèle
def test_train_model():
    data = {
        "name": "model1",
        "neural_network_type": "LSTMNN",
        "training_data": training_data
    }
    response = client.post("/train_model", json=data)
    assert response.status_code == 200, f"Erreur lors de l'entraînement du modèle : {response.text}"
    assert "training completed" in response.json()["status"]

# Test de l'API de recherche dans le modèle
def test_search_model():
    data = {
        "name": "model1",
        "neural_network_type": "LSTMNN",
        "vector": search_vector
    }
    response = client.post("/search", json=data)
    assert response.status_code == 200, f"Erreur lors de la recherche dans le modèle : {response.text}"
    result = response.json()
    assert result["search"] == search_vector
    assert "find" in result
    assert "stats" in result

# Test de l'API de version
def test_get_version():
    response = client.get("/version")
    assert response.status_code == 200, f"Erreur lors de la récupération de la version : {response.text}"
    version_info = response.json()
    assert "version" in version_info
