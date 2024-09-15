import pytest
from app.usecases.create_model import create_model
from app.usecases.train_model import train_model
from app.usecases.search import search_model
from app.repositories.memory import models
from fastapi import HTTPException # type: ignore

# Fonction de configuration pour réinitialiser la mémoire avant chaque test
def setup_function():
    models.clear()

# Test 1: Recherche réussie avec un réseau de neurones
def test_search_model_success():
    # Créer et entraîner un modèle
    create_model("model1", [[1, 2, 3]])

    # Entraîner le modèle avec des vecteurs non vides
    train_model("model1", [[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Recherche avec un vecteur valide
    result = search_model("model1", [1, 2, 3])

    # Assertions
    assert result["search"] == [1, 2, 3]
    assert result["find"] == [1, 2, 3]  # Le vecteur recherché devrait correspondre exactement
    assert result["stats"]["accuracy"] > 0  # La précision devrait être supérieure à 0

# Test 2: Recherche dans un modèle sans dictionnaire de vecteurs
def test_search_model_no_dictionary():
    # Créer un modèle sans entraîner le réseau de neurones
    create_model("model1", None)

    # Entraîner le modèle avec des vecteurs non vides
    train_model("model1", [[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Vérifier que la recherche renvoie une erreur
    with pytest.raises(HTTPException) as excinfo:
        search_model("model1", [1, 2, 3])
    
    assert excinfo.value.status_code == 400
    assert str(excinfo.value.detail) == "No vectors available in the model"

# Test 3: Recherche dans un modèle inexistant
def test_search_model_not_found():
    # Essayer de faire une recherche sur un modèle qui n'existe pas
    with pytest.raises(HTTPException) as excinfo:
        search_model("model_not_exist", [1, 2, 3])
    
    assert excinfo.value.status_code == 404
    assert str(excinfo.value.detail) == "Model not found"

# Test 4: Recherche avec un réseau de neurones non entraîné
def test_search_model_no_nn_model():
    # Créer un modèle sans entraîner le réseau de neurones
    create_model("model1", ["token1", "token2", "token3"])
    
    # Enregistrer un dictionnaire sans réseau de neurones
    models["model1"]["dictionary"] = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    # Vérifier que la recherche renvoie une erreur
    with pytest.raises(HTTPException) as excinfo:
        search_model("model1", [1, 2, 3])
    
    assert excinfo.value.status_code == 400
    assert str(excinfo.value.detail) == "No neural network model found in the model"