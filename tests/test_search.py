import pytest
from app.usecases.create_model import create_model
from app.usecases.train_model import train_model
from app.usecases.search import search_model
from app.repositories.memory import models
from fastapi import HTTPException # type: ignore

# Fonction de configuration pour réinitialiser la mémoire avant chaque test
def setup_function():
    models.clear()

#@pytest.mark.focus
# Test 1: Recherche réussie avec un réseau de neurones
def test_search_model_success():
    # Créer et entraîner un modèle
    create_model("model1", [["token1", "token2", "token3"], ["token1", "token2", "token4"], ["token1", "token2", "token5"]], ["token1", "token2", "token3", "token4", "token5"])

    # Entraîner le modèle avec des vecteurs non vides
    train_model("model1", [
        (["token1", "token2", "token3"], ["token1", "token2", "token3"]), 
        (["token1", "token2", "token4"], ["token1", "token2", "token3"]), 
        (["token1", "token2", "token5"], ["token1", "token2", "token3"])
    ])

    # Recherche avec un vecteur valide
    result = search_model("model1", ["token1", "token2", "token3"])

    # Assertions
    assert result["search"] == ["token1", "token2", "token3"]
    assert result["indexed_search"] == [2, 3, 4]
    assert result["find"] == ['token1', 'token2', 'token3']  # Le vecteur recherché devrait correspondre exactement
    assert result["indexed_find"] == [2, 3, 4]  # Le vecteur recherché devrait correspondre exactement
    assert result["stats"]["accuracy"] > 0  # La précision devrait être supérieure à 0

# Test 2: Recherche réussie avec des vecteurs à taille multiple
def test_search_pad_success():
    # Créer et entraîner un modèle
    create_model(
        "model1", 
        [
            ["token1", "token2", "token3"], 
            ["token1", "token2", "token4"], 
            ["token1", "token4"], 
            ["token1", "token2", "token5"],
            ["token1", "token2", "token3", "token4"],
        ], 
        ["token1", "token2", "token3", "token4", "token5"]
    )

    # Entraîner le modèle avec des vecteurs non vides
    train_model(
        "model1", 
        [
            (["token1", "token2", "token3"], ["token2", "token3", "token4"]),  # Input -> Target
            (["token1", "token2", "token4"], ["token2", "token4", "token5"]),
            (["token1", "token4"], ["token4", "token5"]),  # Cible plus courte que l'input
            (["token1", "token2", "token5"], ["token2", "token5", "token1"]),
            (["token1", "token2", "token3", "token4"], ["token2", "token3", "token4", "token5"])
        ]
    )

    # Recherche avec un vecteur valide
    result = search_model("model1", ["token1", "token2"])

    # Assertions
    assert result["search"] == ["token1", "token2"]
    assert result["indexed_search"] == [2, 3]
    assert result["find"] == ['token1', 'token4', '', '']  # Le vecteur recherché devrait correspondre exactement
    assert result["indexed_find"] == [2, 5, 0, 0]  # Le vecteur recherché devrait correspondre exactement
    assert result["stats"]["accuracy"] > 0  # La précision devrait être supérieure à 0

# Test: Recherche réussie avec mot inconnu
# @pytest.mark.focus
def test_search_unkown_success():
    # Créer et entraîner un modèle
    create_model(
        "model1", 
        [
            ["token1", "token2", "token3"], 
            ["token1", "token2", "token4"], 
            ["token1", "token4"], 
            ["token1", "token2", "token5"],
            ["token1", "token2", "token3", "token4"],
        ], 
        ["token1", "token2", "token3", "token4", "token5"]
    )

    # Entraîner le modèle avec des vecteurs non vides
    train_model(
        "model1", 
        [
            (["token1", "token2", "token3"], ["token2", "token3", "token4"]),  # Input -> Target
            (["token1", "token2", "token4"], ["token2", "token4", "token5"]),
            (["token1", "token4"], ["token4", "token5"]),  # Cible plus courte que l'input
            (["token1", "token2", "token5"], ["token2", "token5", "token1"]),
            (["token1", "token2", "token3", "token4"], ["token2", "token3", "token4", "token5"])
        ]
    )

    # Recherche avec un vecteur whith token unkown 
    result = search_model("model1", ["token1", "token6"])

    # Assertions
    assert result["search"] == ["token1", "token6"]
    assert result["indexed_search"] == [2, 1]
    assert result["find"] == ['token1', 'token4', '', '']  # Le vecteur recherché devrait correspondre exactement
    assert result["indexed_find"] == [2, 5, 0, 0]  # Le vecteur recherché devrait correspondre exactement
    assert result["stats"]["accuracy"] > 0  # La précision devrait être supérieure à 0

# Test 3: Recherche dans un modèle inexistant
def test_search_model_not_found():
    # Essayer de faire une recherche sur un modèle qui n'existe pas
    with pytest.raises(HTTPException) as excinfo:
        search_model("model_not_exist", [["token1", "token2"]])
    
    assert excinfo.value.status_code == 404
    assert str(excinfo.value.detail) == "Model not found"

# Test 4: Recherche avec un réseau de neurones non entraîné
def test_search_model_no_nn_model():
    # Créer un modèle sans entraîner le réseau de neurones
    create_model("model1", [["token1", "token2"], ["token1", "token3"], ["token2", "token3"]], ["token1", "token2", "token3"])
    
    # Enregistrer indexed_dictionary sans réseau de neurones
    models["model1"]["indexed_dictionary"] = [[0, 1],[0, 2],[1, 2]]

    # Vérifier que la recherche renvoie une erreur
    with pytest.raises(HTTPException) as excinfo:
        search_model("model1", [["token1", "token2"]])
    
    assert excinfo.value.status_code == 400
    assert str(excinfo.value.detail) == "No neural network model found in the model"