import pytest
from app.usecases.train_model import train_model
from app.usecases.create_model import create_model
from app.repositories.memory import models

def setup_function():
    """Réinitialise la mémoire avant chaque test"""
    models.clear()

def test_train_model_success():
    create_model("model1", ["token1", "token2", "token3"])
    result = train_model("model1", [["token1", "token3"], ["token2"]])
    assert result == {"status": "training completed", "model_name": "model1"}
    assert "dictionary" in models["model1"]
    assert models["model1"]["dictionary"] == [["token1", "token3"], ["token2"]]

def test_train_model_not_found():
    with pytest.raises(Exception) as excinfo:
        train_model("model_not_exist", [["token1", "token3"]])
    assert str(excinfo.value.detail) == "Model not found"
