# app\usecases\siamese\usecase_create_siamese_test.py
import pytest

from app.services.bdd.models.model_data import ModelData
from app.usecases.siamese.usecase_create_siamese import CreateSiameseUsecaseDto, create_model_siamese


# Test for successful model creation
def test_create_model_siamese_success(patch_inversify):
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify

    mock_bdd.model_exists.return_value = False

    # Call create_model_siamese with valid inputs
    result = create_model_siamese(CreateSiameseUsecaseDto(
        name="model1",
        dictionary=[["token1", "token2"], ["token1", "token3"]],
        glossary=["token1", "token2", "token3"],
        inversify=mock_inversify
    ))

    # Verify that the model is created successfully
    assert result == {
        "status": "model created",
        "model_name": "model1",
        "missing_tokens": []
    }

    # Verify save_model was called with correct ModelData
    expected_model_data = ModelData(
        name="model1",
        neural_network_type="SIAMESE",
        dictionary=[["token1", "token2"], ["token1", "token3"]],
        indexed_dictionary=[[2, 3], [2, 4]],
        glossary=["", "UNK", "token1", "token2", "token3"]
    )
    mock_bdd.save_model.assert_called_once()
    actual_model_data = mock_bdd.save_model.call_args[0][0]

    # Validate ModelData content
    assert isinstance(actual_model_data, ModelData)
    assert actual_model_data.name == expected_model_data.name
    assert actual_model_data.dictionary == expected_model_data.dictionary
    assert actual_model_data.indexed_dictionary == expected_model_data.indexed_dictionary
    assert actual_model_data.glossary == expected_model_data.glossary


# Test for attempting to create a model that already exists
def test_create_model_siamese_already_exists(patch_inversify):
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify

    mock_bdd.model_exists.return_value = True

    with pytest.raises(Exception) as excinfo:
        create_model_siamese(CreateSiameseUsecaseDto(
            name="model1",
            dictionary=[["token1", "token2"]],
            glossary=["token1", "token2"],
            inversify=mock_inversify
        ))

    # Validate exception message
    assert str(excinfo.value) == "[#create_model_siamese]Model already exists"


# Test for missing dictionary or glossary
@pytest.mark.parametrize("dictionary, glossary, expected_message", [
    (None, ["token1"], "[#create_model_siamese]Dictionary cannot be None"),
    ([["token1"]], None, "[#create_model_siamese]Glossary cannot be None"),
    ([], ["token1"], "[#create_model_siamese]Dictionary cannot be empty"),
    ([["token1"]], [], "[#create_model_siamese]Glossary cannot be empty"),
])
def test_create_model_siamese_invalid_inputs(patch_inversify, dictionary, glossary, expected_message):
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify

    mock_bdd.model_exists.return_value = False

    with pytest.raises(Exception) as excinfo:
        create_model_siamese(CreateSiameseUsecaseDto(
            name="model1",
            dictionary=dictionary,
            glossary=glossary,
            inversify=mock_inversify
        ))

    # Validate exception message
    assert str(excinfo.value) == expected_message


# Test for creating a model with unknown tokens
def test_create_model_with_unknown_tokens(patch_inversify):
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify

    mock_bdd.model_exists.return_value = False

    result = create_model_siamese(CreateSiameseUsecaseDto(
        name="model1",
        dictionary=[["token1", "tokenX"]],
        glossary=["token1", "token2"],
        inversify=mock_inversify
    ))

    # Verify the response and missing tokens
    assert result == {
        "status": "model created",
        "model_name": "model1",
        "missing_tokens": ["tokenX"]
    }


# Test for creating a model with empty sublists in the dictionary
def test_create_model_empty_token_lists(patch_inversify):
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify

    mock_bdd.model_exists.return_value = False

    result = create_model_siamese(CreateSiameseUsecaseDto(
        name="model1",
        dictionary=[[], ["token1", "token2"], []],
        glossary=["token1", "token2", "token3"],
        inversify=mock_inversify
    ))

    # Verify the response
    assert result == {
        "status": "model created",
        "model_name": "model1",
        "missing_tokens": []
    }