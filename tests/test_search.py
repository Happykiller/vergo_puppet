import pytest
from app.usecases.search import search_model
from app.usecases.create_model import create_model
from app.usecases.train_model import train_model
from app.repositories.memory import models

def setup_function():
    """Réinitialise la mémoire avant chaque test"""
    models.clear()

def test_search_model_success():
    create_model("model1", ["token1", "token2", "token3"])
    train_model("model1", [["token1", "token3"], ["token2"]])
    
    result = search_model("model1", ["token1", "token4", "token3"])
    assert result["search"] == ["token1", "token4", "token3"]
    assert result["find"] == ["token1", "token3"]
    assert result["stats"]["accuracy"] == 2/3

def test_search_model_no_dictionary():
    create_model("model1", ["token1", "token2", "token3"])
    
    with pytest.raises(Exception) as excinfo:
        search_model("model1", ["token1", "token3"])
    assert str(excinfo.value.detail) == "No vectors available in the model"

def test_search_model_not_found():
    with pytest.raises(Exception) as excinfo:
        search_model("model_not_exist", ["token1", "token3"])
    assert str(excinfo.value.detail) == "Model not found"
