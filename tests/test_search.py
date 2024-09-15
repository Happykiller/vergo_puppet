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
    create_model("model1", [[1, 2, 3]])
    
    # Enregistrer vector_dict sans réseau de neurones
    models["model1"]["vector_dict"] = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    # Vérifier que la recherche renvoie une erreur
    with pytest.raises(HTTPException) as excinfo:
        search_model("model1", [1, 2, 3])
    
    assert excinfo.value.status_code == 400
    assert str(excinfo.value.detail) == "No neural network model found in the model"

#@pytest.mark.focus
# Test 6: Recherche réussie avec un réseau de neurones
def test_search_model_success():
    # Créer un modèle avec des vecteurs de référence
    create_model("model1", [[1, 2, 0], [0, 2, 3]])

    # Entraîner le modèle avec des vecteurs non vides
    train_model("model1", [[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Recherche avec un vecteur valide
    result = search_model("model1", [1, 2, 3])

    # Assertions
    # Ajouter des messages explicatifs aux assertions
    assert result["search"] == [1, 2, 3], "#6.1 Le vecteur de recherche ne correspond pas au vecteur attendu."
    assert result["find"] == [1, 2, 0], "#6.2 Le vecteur trouvé n'est pas celui attendu."
    assert result["stats"]["accuracy"], "#6.3 La précision n'est pas présente dans les résultats."