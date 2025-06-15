# app/usecases/usecase_get_things_test.py
import pytest # type: ignore
from unittest.mock import MagicMock

from app.services.bdd.models.model_thing import ThingModel
from app.usecases.thing.usecase_get_things import get_things_usecase

@pytest.fixture
def fake_thing_1():
    return ThingModel(
        id="thing_1",
        vector=[0.1, 0.2, 0.3],
        text="Red bike pro",
        metadata={"label": "Red bike"},
        collection_name="colleciton"
    )

@pytest.fixture
def fake_thing_2():
    return ThingModel(
        id="thing_2",
        vector=[0.4, 0.5, 0.6],
        text="Blue helmet",
        metadata={"label": "Blue helmet"},
        collection_name="colleciton"
    )

def test_get_things_all(fake_thing_1, fake_thing_2):
    """
    Test retrieving all things when ids=None.
    """
    # Mock the BDD and Inversify
    mock_bdd = MagicMock()
    mock_bdd.get_things.return_value = [fake_thing_1, fake_thing_2]
    inversify = MagicMock()
    inversify.get_bdd.return_value = mock_bdd

    result = get_things_usecase('colleciton', None, inversify)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0].id == "thing_1"
    assert result[1].id == "thing_2"

def test_get_things_by_ids(fake_thing_1, fake_thing_2):
    """
    Test retrieving by a list of IDs.
    """
    mock_bdd = MagicMock()
    mock_bdd.get_things.return_value = [fake_thing_2]
    inversify = MagicMock()
    inversify.get_bdd.return_value = mock_bdd

    result = get_things_usecase('colleciton', ["thing_2"], inversify)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].id == "thing_2"

def test_get_things_empty_result():
    """
    Test when no objects are found.
    """
    mock_bdd = MagicMock()
    mock_bdd.get_things.return_value = []
    inversify = MagicMock()
    inversify.get_bdd.return_value = mock_bdd

    result = get_things_usecase('colleciton', ["unknown_id"], inversify)
    assert isinstance(result, list)
    assert len(result) == 0

def test_get_things_exception():
    """
    Test exception propagation from the BDD layer.
    """
    mock_bdd = MagicMock()
    mock_bdd.get_things.side_effect = Exception("BDD error")
    inversify = MagicMock()
    inversify.get_bdd.return_value = mock_bdd

    with pytest.raises(Exception) as exc:
        get_things_usecase('colleciton', None, inversify)
    assert "[get_things_usecase]" in str(exc.value)
