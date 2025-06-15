# app\conftest.py
import pytest # type: ignore
from unittest.mock import patch, MagicMock

from app.common import _reset_env_cache

"""Helper to configure mock Inversify and its dependencies."""
@pytest.fixture
def patch_inversify():
  with patch("app.usecases.simple.usecase_search_simple.Inversify"), \
    patch("app.usecases.simple.usecase_train_simple.Inversify"), \
    patch("app.usecases.simple.usecase_search_simple.Inversify") as mock_inversify_class:
    # Création du mock "Inversify"
    mock_inversify_instance = MagicMock()
    # Création du mock "bdd"
    mock_bdd = MagicMock()
    
    # On simule l’appel get_bdd() pour qu’il renvoie mock_bdd
    mock_inversify_instance.get_bdd.return_value = mock_bdd
    
    # On configure la classe retournée pour renvoyer notre instance mockée
    mock_inversify_class.return_value = mock_inversify_instance
    
    # On "yield" un tuple (mock_inversify_instance, mock_bdd)
    # afin que le test (ou d'autres fixtures) puissent les utiliser
    yield mock_inversify_instance, mock_bdd

@pytest.fixture(autouse=True, scope="function")
def mock_env_vars(monkeypatch):
    """
    Automatically apply mocked env vars for every test.
    Resets the singleton cache so each test gets a fresh environment.
    """
    _reset_env_cache()

    monkeypatch.setenv("SECRET_KEY", "mocked-secret-key")
    monkeypatch.setenv("MODE", "test")
    monkeypatch.setenv("BDD", "fake")
    monkeypatch.setenv("MONGO_URI", "mongodb://mock")
    monkeypatch.setenv("MONGO_DB_NAME", "mock-db")
    monkeypatch.setenv("DEBUG", "true")