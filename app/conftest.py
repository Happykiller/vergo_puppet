# app\conftest.py
import pytest # type: ignore
from unittest.mock import patch, MagicMock

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
