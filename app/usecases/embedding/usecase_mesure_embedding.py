# app/usecases/embedding/usecase_mesure_embedding.py
import traceback
import numpy as np  # type: ignore
from collections import Counter
from typing import Any, List, NamedTuple, Dict

from app.services.logger import logger
from app.services.bdd.models.model_metrics import MetricsModel
from app.usecases.embedding.usecase_similarity_embedding import cosine_similarity
from app.usecases.embedding.usecase_encode_embedding import encode_embedding_usecase


class MesureEmbeddingUsecaseDto(NamedTuple):
    name: str
    test_data: List[Dict[str, Any]]
    inversify: Any


def mesure_embedding(dto: MesureEmbeddingUsecaseDto) -> Dict[str, Any]:
    """Measure performance of an embedding model on a test dataset."""
    try:
        bdd = dto.inversify.get_bdd()
        total_tests = len(dto.test_data)
        correct_predictions = 0
        similarity_precision: List[float] = []
        details: List[Dict[str, Any]] = []

        for item in dto.test_data:
            sent1 = item["seq1"]
            sent2 = item["seq2"]
            expected = float(item.get("similarity", 0.0))

            emb1 = encode_embedding_usecase(dto.name, sent1, dto.inversify)
            emb2 = encode_embedding_usecase(dto.name, sent2, dto.inversify)
            predicted = cosine_similarity(emb1, emb2)
            error = abs(predicted - expected)
            precision = (1 - error) * 100
            similarity_precision.append(precision)
            is_correct = error <= 0.1
            if is_correct:
                correct_predictions += 1
            details.append({
                "seq1": sent1,
                "seq2": sent2,
                "expected_similarity": expected,
                "predicted_similarity": predicted,
                "error": error,
                "precision_percentage": precision,
                "is_correct": is_correct,
            })
            logger.debug(
                f"Expected: {expected*100:.1f}%, Predicted: {predicted*100:.1f}%, Error: {error*100:.1f}%"
            )

        prediction_accuracy = (correct_predictions / total_tests) * 100 if total_tests else 0.0
        if similarity_precision:
            avg_precision = float(np.mean(similarity_precision))
            median_precision = float(np.median(similarity_precision))
            mode_precision = Counter(similarity_precision).most_common(1)[0][0]
            range_precision = max(similarity_precision) - min(similarity_precision)
            variance_precision = float(np.var(similarity_precision))
            std_dev_precision = float(np.std(similarity_precision))
            q1 = float(np.percentile(similarity_precision, 25))
            q3 = float(np.percentile(similarity_precision, 75))
            coeff_variation = (
                (std_dev_precision / avg_precision) * 100 if avg_precision != 0 else 0
            )
        else:
            avg_precision = median_precision = mode_precision = 0.0
            range_precision = variance_precision = std_dev_precision = 0.0
            q1 = q3 = coeff_variation = 0.0

        report = {
            "model_name": dto.name,
            "total_tests": total_tests,
            "correct_predictions": correct_predictions,
            "prediction_accuracy_percentage": prediction_accuracy,
            "avg_similarity_precision_percentage": avg_precision,
            "median_similarity_precision_percentage": median_precision,
            "mode_similarity_precision_percentage": mode_precision,
            "range_similarity_percentage": range_precision,
            "variance_similarity": variance_precision,
            "std_dev_similarity": std_dev_precision,
            "quartile_1": q1,
            "quartile_3": q3,
            "coefficient_of_variation_percentage": coeff_variation,
            "details": details,
        }

        bdd.save_metrics(
            MetricsModel(
                model_name=dto.name,
                metrics={
                    "type": "mesure",
                    "total_tests": total_tests,
                    "correct_predictions": correct_predictions,
                    "prediction_accuracy_percentage": prediction_accuracy,
                    "avg_similarity_precision_percentage": avg_precision,
                    "median_similarity_precision_percentage": median_precision,
                    "mode_similarity_precision_percentage": mode_precision,
                    "range_similarity_percentage": range_precision,
                    "variance_similarity": variance_precision,
                    "std_dev_similarity": std_dev_precision,
                    "quartile_1": q1,
                    "quartile_3": q3,
                    "coefficient_of_variation_percentage": coeff_variation,
                },
            )
        )

        logger.info(
            f"Prediction accuracy: {prediction_accuracy:.2f}% ({correct_predictions}/{total_tests})"
        )
        logger.info(f"Average similarity precision: {avg_precision:.2f}%")
        return report

    except Exception as e:
        logger.error(
            f"[mesure_embedding] Error: {str(e)}\n{traceback.format_exc()}"
        )
        raise Exception(f"[#mesure_embedding]{str(e)}")
