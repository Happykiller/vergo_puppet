import pytest
from unittest.mock import patch, MagicMock

from app.usecases.embedding.usecase_train_embedding import train_embedding_usecase
from app.services.bdd.models.model_data import ModelStatus


@patch("app.usecases.embedding.usecase_train_embedding.train_embedding_model")
def test_train_embedding_usecase_success(mock_train_embedding, patch_inversify):
    mock_inversify, mock_bdd = patch_inversify
    mock_model = MagicMock(status=ModelStatus.TRAINED)
    mock_bdd.get_model.return_value = mock_model
    mock_nn_model = MagicMock()
    mock_train_embedding.return_value = (mock_nn_model, {"final_loss": 0.1})

    vocab = {"<PAD>": 0, "<UNK>": 1}
    trainset = [{"seq1": [0], "seq2": [1], "label": 1.0}]

    result = train_embedding_usecase("model", vocab, trainset, mock_inversify)

    assert result["model"] == "model"
    assert mock_bdd.update_model.call_count == 2
    assert mock_bdd.update_model.call_args_list[0][0][0].status == ModelStatus.TRAINING
    assert mock_bdd.update_model.call_args_list[1][0][0].status == ModelStatus.TRAINED


@patch("app.usecases.embedding.usecase_train_embedding.train_embedding_model")
def test_train_embedding_usecase_already_training(mock_train_embedding, patch_inversify):
    mock_inversify, mock_bdd = patch_inversify
    mock_model = MagicMock(status=ModelStatus.TRAINING)
    mock_bdd.get_model.return_value = mock_model

    vocab = {"<PAD>": 0}
    trainset = [{"seq1": [0], "seq2": [0], "label": 1.0}]

    with pytest.raises(Exception, match="Model is training"):
        train_embedding_usecase("model", vocab, trainset, mock_inversify)
    mock_train_embedding.assert_not_called()
