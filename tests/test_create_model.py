import pytest
from app.usecases.create_model import create_model
from app.repositories.memory import models
from fastapi import HTTPException

def setup_function():
    """Réinitialise la mémoire avant chaque test"""
    models.clear()

# Test de création d'un modèle avec succès
def test_create_model_success():
    # Teste la création d'un modèle avec un dictionnaire et un glossaire valides
    result = create_model("model1", [["token1", "token2"], ["token1", "token3"]], ["token1", "token2", "token3"])
    
    # Vérifie que le résultat est comme attendu
    assert result == {"status": "model created", "model_name": "model1"}
    
    # Vérifie que le modèle a bien été enregistré dans la mémoire
    assert "model1" in models
    
    # Vérifie que le dictionnaire original est enregistré correctement
    assert models["model1"]["dictionary"] == [["token1", "token2"], ["token1", "token3"]]
    
    # Vérifie que le dictionnaire indexé a été correctement généré
    assert models["model1"]["indexed_dictionary"] == [[0, 1], [0, 2]]

# Test de tentative de création d'un modèle déjà existant
def test_create_model_already_exists():
    # Crée un premier modèle
    create_model("model1", [["token1", "token2"], ["token1", "token3"]], ["token1", "token2", "token3"])
    
    # Tente de créer un modèle avec le même nom, ce qui devrait lever une exception
    with pytest.raises(HTTPException) as excinfo:
        create_model("model1", [["token1", "token2"]], ["token1", "token2", "token3"])
    
    # Vérifie que l'exception levée contient le bon message
    assert str(excinfo.value.detail) == "Model already exists"

# Test de création d'un modèle avec un dictionnaire None
def test_create_model_dictionary_none():
    # Vérifie que la fonction lève une exception si le dictionnaire est None
    with pytest.raises(HTTPException) as excinfo:
        create_model("model1", None, ["token1", "token2", "token3"])
    
    # Vérifie que le message d'erreur est correct
    assert excinfo.value.status_code == 400
    assert str(excinfo.value.detail) == "Dictionary cannot be None"

# Test de création d'un modèle avec un glossaire None
def test_create_model_glossary_none():
    # Vérifie que la fonction lève une exception si le glossaire est None
    with pytest.raises(HTTPException) as excinfo:
        create_model("model1", [["token1", "token2"]], None)
    
    # Vérifie que le message d'erreur est correct
    assert excinfo.value.status_code == 400
    assert str(excinfo.value.detail) == "Glossary cannot be None"

# Test de création d'un modèle avec un dictionnaire vide
def test_create_model_dictionary_empty():
    # Vérifie que la fonction lève une exception si le dictionnaire est vide
    with pytest.raises(HTTPException) as excinfo:
        create_model("model1", [], ["token1", "token2", "token3"])
    
    # Vérifie que le message d'erreur est correct
    assert excinfo.value.status_code == 400
    assert str(excinfo.value.detail) == "Dictionary cannot be empty"

# Test de création d'un modèle avec un glossaire vide
def test_create_model_glossary_empty():
    # Vérifie que la fonction lève une exception si le glossaire est vide
    with pytest.raises(HTTPException) as excinfo:
        create_model("model1", [["token1", "token2"]], [])
    
    # Vérifie que le message d'erreur est correct
    assert excinfo.value.status_code == 400
    assert str(excinfo.value.detail) == "Glossary cannot be empty"

# Test de création d'un modèle avec des tokens non reconnus
def test_create_model_with_unknown_tokens():
    # Vérifie que la fonction traite correctement les tokens non trouvés dans le glossaire
    result = create_model("model1", [["token1", "token2", "tokenX"]], ["token1", "token2", "token3"])
    
    # Vérifie que le modèle est bien créé
    assert result == {"status": "model created", "model_name": "model1"}
    
    # Vérifie que le dictionnaire indexé contient None pour le token "tokenX"
    assert models["model1"]["indexed_dictionary"] == [[0, 1, None]]

# Test de création d'un modèle avec un glossaire contenant des doublons
def test_create_model_no_duplicates_in_glossary():
    # Vérifie que la première occurrence d'un token est utilisée pour l'indexation
    result = create_model("model1", [["token1", "token2"]], ["token1", "token2", "token1", "token3"])
    
    # Vérifie que le modèle est bien créé
    assert result == {"status": "model created", "model_name": "model1"}
    
    # Vérifie que le dictionnaire indexé n'utilise que la première occurrence de "token1"
    assert models["model1"]["indexed_dictionary"] == [[0, 1]]

# Test de création d'un modèle avec des sous-listes vides dans le dictionnaire
def test_create_model_empty_token_lists():
    # Vérifie que les sous-listes vides dans le dictionnaire sont correctement gérées
    result = create_model("model1", [[], ["token1", "token2"], []], ["token1", "token2", "token3"])
    
    # Vérifie que le modèle est bien créé
    assert result == {"status": "model created", "model_name": "model1"}
    
    # Vérifie que les sous-listes vides restent vides dans le dictionnaire indexé
    assert models["model1"]["indexed_dictionary"] == [[], [0, 1], []]
