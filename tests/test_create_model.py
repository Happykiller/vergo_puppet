import pytest
from app.usecases.create_model import create_model
from app.repositories.memory import models

def setup_function():
    """Réinitialise la mémoire avant chaque test"""
    models.clear()

def test_create_model_success():
    result = create_model("model1", ["token1", "token2", "token3"])
    assert result == {"status": "model created", "model_name": "model1"}
    assert "model1" in models
    assert models["model1"]["vector_dict"] == ["token1", "token2", "token3"]

def test_create_model_already_exists():
    create_model("model1", ["token1", "token2", "token3"])
    with pytest.raises(Exception) as excinfo:
        create_model("model1", ["token4", "token5"])
    assert str(excinfo.value.detail) == "Model already exists"
