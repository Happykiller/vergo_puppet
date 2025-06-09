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
        {"sentence1": "hello", "sentence2": "bonjour", "label": 1.0},
        {"sentence1": "hello", "sentence2": "au revoir", "label": 0.0},
    ]

    result = mesure_embedding(
        MesureEmbeddingUsecaseDto(
            name="puppet-o5",
            test_data=test_data,
            inversify=mock_inversify,
        )
    )

    assert result["total_tests"] == 2
    assert result["correct_predictions"] == 2
    assert result["prediction_accuracy_percentage"] == pytest.approx(100.0)


@patch("app.usecases.embedding.usecase_mesure_embedding.encode_embedding_usecase", side_effect=Exception("fail"))
def test_mesure_embedding_failure(mock_encode, patch_inversify):
    mock_inversify, _ = patch_inversify
    test_data = [{"sentence1": "a", "sentence2": "b", "label": 1.0}]
    with pytest.raises(Exception):
        mesure_embedding(
            MesureEmbeddingUsecaseDto(
                name="puppet-o5",
                test_data=test_data,
                inversify=mock_inversify,
            )
        )

