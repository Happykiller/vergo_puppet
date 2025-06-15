# app\usecases\embedding\usecase_mesure_embedding_test.py
import pytest  # type: ignore
from unittest.mock import patch

from app.usecases.embedding.usecase_mesure_embedding import (
    MesureEmbeddingUsecaseDto,
    mesure_embedding,
)


@patch("app.usecases.embedding.usecase_mesure_embedding.encode_embedding_usecase")
def test_mesure_embedding_success(mock_encode, patch_inversify):
    mock_inversify, _ = patch_inversify

    # Mock embeddings so cosine similarity returns 1 for first pair and 0 for second
    mock_encode.side_effect = [
        [1.0, 0.0], [1.0, 0.0],  # similarity 1.0
        [1.0, 0.0], [0.0, 1.0],  # similarity 0.0
    ]

    test_data = [
        {"seq1": "hello", "seq2": "bonjour", "similarity": 1.0},
        {"seq1": "hello", "seq2": "au revoir", "similarity": 0.0},
    ]

    result = mesure_embedding(
        MesureEmbeddingUsecaseDto(
            name="puppet-o5",
            test_data=test_data,
            inversify=mock_inversify,
        )
    )

    assert result["total_tests"] == 2
    assert result["correct_predictions"] == 1
    assert result["prediction_accuracy_percentage"] == pytest.approx(50.0)


@patch("app.usecases.embedding.usecase_mesure_embedding.encode_embedding_usecase", side_effect=Exception("fail"))
def test_mesure_embedding_failure(mock_encode, patch_inversify):
    mock_inversify, _ = patch_inversify
    test_data = [{"seq1": "a", "seq2": "b", "similarity": 1.0}]
    with pytest.raises(Exception):
        mesure_embedding(
            MesureEmbeddingUsecaseDto(
                name="puppet-o5",
                test_data=test_data,
                inversify=mock_inversify,
            )
        )

