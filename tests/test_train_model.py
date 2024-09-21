import pytest
from app.usecases.train_model import train_model
from app.repositories.memory import models, save_model

# Réinitialiser la mémoire avant chaque test
def setup_function():
    models.clear()

# Test 1: Vérifier qu'une erreur 404 est levée si le modèle n'existe pas
def test_train_model_not_found():
    with pytest.raises(Exception) as excinfo:
        train_model("model1", [["token1", "token2", "token3"]]) # Modèle non existant
    
    assert excinfo.value.status_code == 404 # Vérifie que l'erreur est 404
    assert str(excinfo.value.detail) == "Model not found"

# Test 2: Vérifier qu'une erreur 400 est levée si aucun dictionnaire n'est fourni
def test_train_model_no_dictionary():
    # Enregistrer un modèle vide
    save_model("model1", {
        "dictionary": [["token1", "token2", "token3"]],
        "indexed_dictionary": [[0,1,2]],
        "glossary": ["token1", "token2", "token3"]
    })

    with pytest.raises(Exception) as excinfo:
        train_model("model1", None)
    
    assert excinfo.value.status_code == 400
    assert str(excinfo.value.detail) == "No vectors provided for training"

# Test 3: Vérifier qu'une erreur 400 est levée si le dictionnaire est vide
def test_train_model_empty_dictionary():
    # Enregistrer un modèle vide
    save_model("model1", {
        "dictionary": [["token1", "token2", "token3"]],
        "indexed_dictionary": [[0,1,2]],
        "glossary": ["token1", "token2", "token3"]
    })

    with pytest.raises(Exception) as excinfo:
        train_model("model1", [])
    
    assert excinfo.value.status_code == 400
    assert str(excinfo.value.detail) == "Dictionary of vectors is empty"

#@pytest.mark.focus
# Test 4: Vérifier que tous les vecteurs ont la même longueur
def test_train_model_inconsistent_vector_size():
    # Enregistrer un modèle
    save_model("model1", {
        "dictionary": [["token1", "token2", "token3"]],
        "indexed_dictionary": [[0,1,2]],
        "glossary": ["token1", "token2", "token3"]
    })

    # Fournir un dictionnaire avec des vecteurs de taille différente
    with pytest.raises(Exception) as excinfo:
        train_model("model1", [["token1", "token2", "token3"], ["token1", "token2"]])

    assert excinfo.value.status_code == 400
    assert str(excinfo.value.detail) == "All vectors must have the same length"

# Test 5: Entraînement réussi
def test_train_model_success():
    # Enregistrer un modèle
    save_model("model1", {
        "dictionary": [["token1", "token2", "token3"], ["token1", "token2", "token4"], ["token1", "token2", "token5"]],
        "indexed_dictionary": [[0,1,2], [0,1,3], [0,1,4]],
        "glossary": ["token1", "token2", "token3", "token4", "token5"]
    })

    # Appeler la fonction avec des données d'entrainement valide
    response = train_model("model1", [["token1", "token2", "token3"], ["token1", "token2", "token4"], ["token1", "token2", "token5"]])

    # Vérifier que la réponse est correcte
    assert response["status"] == "training completed"
    assert response["model_name"] == "model1"
    
    # Vérifier que le modèle de réseau de neurones est bien enregistré
    model = models.get("model1")
    assert "nn_model" in model
