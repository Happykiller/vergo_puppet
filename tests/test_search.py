import pytest
from app.usecases.create_model import create_model
from app.usecases.train_model import train_model
from app.usecases.search import search_model
from app.repositories.memory import models
from fastapi import HTTPException # type: ignore

# Fonction de configuration pour réinitialiser la mémoire avant chaque test
def setup_function():
    models.clear()

# Test 1: Recherche réussie avec un réseau de neurones et des données variées
def test_search_model_success():
    # Créer et entraîner un modèle avec des données variées
    create_model("model1", [["chat", "chien", "oiseau"], ["voiture", "vélo", "train"], ["ordinateur", "table", "stylo"]], 
                 ["chat", "chien", "oiseau", "voiture", "vélo", "train", "ordinateur", "table", "stylo"])

    # Entraîner le modèle avec des vecteurs non vides
    train_model("model1", [
        (["chat", "chien", "oiseau"], ["chat", "chien", "oiseau"]), 
        (["voiture", "vélo", "train"], ["voiture", "vélo", "train"]), 
        (["ordinateur", "table", "stylo"], ["ordinateur", "table", "stylo"])
    ])

    # Recherche avec un vecteur valide
    result = search_model("model1", ["chat", "chien", "oiseau"])

    # Assertions
    assert result["search"] == ["chat", "chien", "oiseau"]
    assert result["indexed_search"] == [2, 3, 4]
    assert result["find"] == ["chat", "chien", "oiseau"]
    assert result["indexed_find"] == [2, 3, 4]
    assert result["stats"]["accuracy"] > 0

# Test 2: Recherche réussie avec des vecteurs à taille multiple
def test_search_pad_success():
    # Créer et entraîner un modèle avec des chaînes variées
    create_model(
        "model2", 
        [
            ["livre", "stylo", "sac"], 
            ["bouteille", "tasse", "verre"], 
            ["chaise", "table", "lampe"], 
            ["ordinateur", "clavier", "souris"]
        ], 
        ["livre", "stylo", "sac", "bouteille", "tasse", "verre", "chaise", "table", "lampe", "ordinateur", "clavier", "souris"]
    )

    # Entraîner le modèle avec des vecteurs non vides
    train_model(
        "model2", 
        [
            (["livre", "stylo", "sac"], ["livre", "stylo", "sac"]),
            (["bouteille", "tasse", "verre"], ["bouteille", "tasse", "verre"]),
            (["chaise", "table", "lampe"], ["chaise", "table", "lampe"]),
            (["ordinateur", "clavier", "souris"], ["ordinateur", "clavier", "souris"])
        ]
    )

    # Recherche avec un vecteur valide
    result = search_model("model2", ["livre", "stylo"])

    # Assertions
    assert result["search"] == ["livre", "stylo"]
    assert result["indexed_search"] == [2, 3]
    assert result["find"] == ["livre", "stylo", "sac"]
    assert result["indexed_find"] == [2, 3, 4]
    assert result["stats"]["accuracy"] > 0

# Test 3: Recherche réussie avec un mot inconnu
def test_search_unknown_success():
    # Créer et entraîner un modèle avec des chaînes variées
    create_model(
        "model3", 
        [
            ["chat", "chien", "oiseau"], 
            ["voiture", "vélo", "train"], 
            ["ordinateur", "table", "stylo"]
        ], 
        ["chat", "chien", "oiseau", "voiture", "vélo", "train", "ordinateur", "table", "stylo"]
    )

    # Entraîner le modèle avec des vecteurs non vides
    train_model(
        "model3", 
        [
            (["chat", "chien", "oiseau"], ["chat", "chien", "oiseau"]),
            (["voiture", "vélo", "train"], ["voiture", "vélo", "train"]),
            (["ordinateur", "table", "stylo"], ["ordinateur", "table", "stylo"])
        ]
    )

    # Recherche avec un vecteur contenant un token inconnu
    result = search_model("model3", ["chat", "lion"])

    # Assertions
    assert result["search"] == ["chat", "lion"]
    assert result["indexed_search"] == [2, 1]  # "lion" est traité comme inconnu
    assert result["find"] == ["chat", "chien", "oiseau"]
    assert result["indexed_find"] == [2, 3, 4]
    assert result["stats"]["accuracy"] > 0

# Test 4: Recherche dans un modèle inexistant
def test_search_model_not_found():
    # Essayer de faire une recherche sur un modèle qui n'existe pas
    with pytest.raises(HTTPException) as excinfo:
        search_model("model_not_exist", ["chat", "chien"])
    
    assert excinfo.value.status_code == 404
    assert str(excinfo.value.detail) == "Model not found"

# Test 5: Recherche avec un réseau de neurones non entraîné
def test_search_model_no_nn_model():
    # Créer un modèle sans entraîner le réseau de neurones
    create_model("model4", [["chat", "chien"], ["oiseau", "poisson"], ["voiture", "train"]], ["chat", "chien", "oiseau", "poisson", "voiture", "train"])
    
    # Enregistrer indexed_dictionary sans réseau de neurones
    models["model4"]["indexed_dictionary"] = [[0, 1], [2, 3], [4, 5]]

    # Vérifier que la recherche renvoie une erreur
    with pytest.raises(HTTPException) as excinfo:
        search_model("model4", ["chat", "chien"])
    
    assert excinfo.value.status_code == 400
    assert str(excinfo.value.detail) == "No neural network model found in the model"
