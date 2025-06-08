# app/usecases/thing/usecase_store_thing_test.py
import torch
import pytest
from unittest.mock import MagicMock, patch

from app.usecases.thing.usecase_store_thing import store_thing_usecase, flatten_for_embedding, encode_text_with_model

# --- Fixtures & helpers ---

@pytest.fixture
def mock_model():
    """
    Mock of a model used for embedding generation.
    """
    mock = MagicMock()
    mock.word2idx = {"vélo": 2, "rouge": 3, "pro": 4}
    mock_tensor = MagicMock()
    # Ensure .squeeze(0).cpu().tolist() returns a real list
    mock_tensor.squeeze.return_value = mock_tensor
    mock_tensor.cpu.return_value = mock_tensor
    mock_tensor.tolist.return_value = [0.1, 0.2, 0.3]
    mock.nn_model = MagicMock()
    mock.nn_model.forward_once.return_value = mock_tensor
    return mock

@pytest.fixture
def valid_item():
    """
    Example of a complete Thing object for indexing.
    """
    return {
        "id": "thing_123",
        "label": "Vélo rouge",
        "description": "Vélo de course ultra léger",
        "type": "cb",
        "cb": {"label": "Visa Pro"},
        "credential": {"id": "abc", "address": "demo@ex.com"},
        "code": None,
        "note": None,
        "totp": None
    }

# --- Main tests ---

@patch("app.usecases.thing.usecase_store_thing.encode_text_with_model", return_value=[0.1, 0.2, 0.3])
@patch("app.usecases.thing.usecase_store_thing.get_model_usecase")
def test_store_thing_success(mock_get_model_usecase, mock_encode, mock_model, valid_item):
    """
    Nominal case: the object is indexed, the vector is generated and persisted.
    """
    mock_get_model_usecase.return_value = mock_model
    mock_bdd = MagicMock()
    inversify = MagicMock()
    inversify.get_bdd.return_value = mock_bdd

    result = store_thing_usecase(valid_item, "mock_model", inversify)

    mock_bdd.store_thing_embedding.assert_called()
    assert result["status"] == "stored"
    assert result["id"] == valid_item["id"]
    assert isinstance(result["text"], str)
    assert result["vector_dim"] == 3  # The mocked vector has 3 dimensions

@patch("app.usecases.thing.usecase_store_thing.get_model_usecase")
def test_store_thing_missing_label(mock_get_model_usecase, mock_model):
    """
    Error case: missing label field.
    """
    item = {"id": "thing_123"}
    mock_get_model_usecase.return_value = mock_model
    inversify = MagicMock()
    with pytest.raises(Exception) as exc:
        store_thing_usecase(item, "mock_model", inversify)
    assert "label" in str(exc.value)

@patch("app.usecases.thing.usecase_store_thing.get_model_usecase")
def test_store_thing_missing_id(mock_get_model_usecase, mock_model):
    """
    Error case: missing id field.
    """
    item = {"label": "Vélo rouge"}
    mock_get_model_usecase.return_value = mock_model
    inversify = MagicMock()
    with pytest.raises(Exception) as exc:
        store_thing_usecase(item, "mock_model", inversify)
    assert "id" in str(exc.value)

@patch("app.usecases.thing.usecase_store_thing.get_model_usecase")
def test_store_thing_model_not_found(mock_get_model_usecase):
    """
    Error case: unknown model.
    """
    item = {"id": "thing_123", "label": "Vélo rouge"}
    mock_get_model_usecase.return_value = None
    inversify = MagicMock()
    with pytest.raises(Exception) as exc:
        store_thing_usecase(item, "model_does_not_exist", inversify)
    assert "not found" in str(exc.value)

def test_flatten_for_embedding_flat_and_nested():
    """
    flatten_for_embedding should concatenate all string fields, including nested dicts.
    """
    obj = {
        "label": "Red bike",
        "description": "Ultra light racing bike",
        "cb": {"label": "Visa Pro"},
        "credential": {"id": "abc", "address": "demo@ex.com"},
        "code": None,
        "note": None,
        "totp": None
    }
    result = flatten_for_embedding(obj)
    # Order can vary depending on Python version, but all values must be present
    assert "Red bike" in result
    assert "Ultra light racing bike" in result
    assert "Visa Pro" in result
    assert "abc" in result
    assert "demo@ex.com" in result
    assert "None" not in result  # Nulls should not be present

def test_flatten_for_embedding_empty():
    """
    flatten_for_embedding should return an empty string if all fields are empty or null.
    """
    obj = {"code": None, "cb": {}, "credential": None}
    assert flatten_for_embedding(obj) == ""

def test_flatten_for_embedding_only_flat():
    """
    flatten_for_embedding should work with only flat (non-nested) string fields.
    """
    obj = {"label": "X", "description": "Y"}
    assert flatten_for_embedding(obj) == "X Y"

def test_flatten_for_embedding_only_nested():
    """
    flatten_for_embedding should work with only nested dict fields.
    """
    obj = {"cb": {"label": "Visa"}, "credential": {"address": "z@t.com"}}
    result = flatten_for_embedding(obj)
    assert "Visa" in result
    assert "z@t.com" in result

def test_encode_text_with_model_basic():
    """
    encode_text_with_model should produce a vector as a list of floats from the model.
    """
    import torch
    mock_model = MagicMock()
    mock_model.word2idx = {"bike": 2, "red": 3}
    mock_model.nn_model = MagicMock()
    mock_model.nn_model.to.return_value = mock_model.nn_model
    mock_model.nn_model.forward_once.return_value = torch.tensor([[1.1, 2.2, 3.3]])

    result = encode_text_with_model(mock_model, "bike red")
    assert result == pytest.approx([1.1, 2.2, 3.3], abs=1e-6)

def test_encode_text_with_model_handles_unknown_words():
    """
    encode_text_with_model should use 1 for unknown tokens (UNK).
    """
    import torch
    mock_model = MagicMock()
    mock_model.word2idx = {"bike": 2}
    mock_model.nn_model = MagicMock()
    mock_model.nn_model.to.return_value = mock_model.nn_model
    mock_model.nn_model.forward_once.return_value = torch.tensor([[4.4, 5.5]])

    result = encode_text_with_model(mock_model, "bike unknownword")
    assert result == pytest.approx([4.4, 5.5], abs=1e-6)
