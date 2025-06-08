# app\usecases\siamese\usecase_mesure_siamese.py
import traceback
import numpy as np
from collections import Counter
from typing import List, NamedTuple

from app.inversify import Inversify
from app.services.logger import logger
from app.services.bdd.models.model_metrics import MetricsModel
from app.neural_network.nn_siamese import SiameseLSTM, evaluate_similarity
from app.usecases.siamese.usecase_commons_siamese import create_indexed_glossary, tokens_to_indices

class MesureSiameseUsecaseDto(NamedTuple):
    name: str
    test_data: List[List[str]]
    inversify: Inversify

def mesure_siamese(dto: MesureSiameseUsecaseDto):
    try:
        # Fetch Bdd
        bdd = dto.inversify.get_bdd()

        correct_predictions = 0
        total_tests = len(dto.test_data)
        similarity_precision = []
        details = []
        
        # Retrieve the model
        model = bdd.get_model(dto.name, SiameseLSTM)
        if not model:
            raise Exception("Model not found")

        glossary = model.glossary
        nn_model = model.nn_model
        if not nn_model:
            raise Exception("Model not completed")

        # Create index mapping for glossary terms
        word2idx = create_indexed_glossary(glossary)
        
        # Iterate through test data
        for vector1, vector2, expected_similarity in dto.test_data:
            # Convert token lists to indices
            vector1_indices = tokens_to_indices(vector1, word2idx)
            vector2_indices = tokens_to_indices(vector2, word2idx)
            
            # Evaluate similarity
            predicted_similarity = evaluate_similarity(nn_model, vector1_indices, vector2_indices)
            error = abs(predicted_similarity - expected_similarity)

            # Record similarity precision as a percentage
            precision = (1 - error) * 100  # Precision as 100% when no error
            similarity_precision.append(precision)
            
            # Check if prediction is correct
            is_correct = error <= 0.1
            if is_correct:
                correct_predictions += 1
            
            # Store detailed results for each test case
            details.append({
                "vector1": vector1,
                "vector2": vector2,
                "expected_similarity": expected_similarity,
                "predicted_similarity": predicted_similarity,
                "error": error,
                "precision_percentage": precision,
                "is_correct": is_correct
            })
            
            # Log the output details
            logger.debug(f"Query: {vector1}, Image: {vector2}")
            logger.debug(f"Expected similarity: {expected_similarity*100}%, Model similarity: {predicted_similarity*100:.2f}%, Error: {error*100:.2f}%")
        
        # Calculate metrics
        prediction_accuracy = (correct_predictions / total_tests) * 100
        avg_similarity_precision = np.mean(similarity_precision)
        median_similarity = np.median(similarity_precision)
        mode_similarity = Counter(similarity_precision).most_common(1)[0][0]
        range_similarity = max(similarity_precision) - min(similarity_precision)
        variance_similarity = np.var(similarity_precision)
        std_dev_similarity = np.std(similarity_precision)
        q1 = np.percentile(similarity_precision, 25)
        q3 = np.percentile(similarity_precision, 75)
        coeff_variation = (std_dev_similarity / avg_similarity_precision) * 100 if avg_similarity_precision != 0 else 0

        # Compile the report
        report = {
            "model_name": dto.name,
            "total_tests": total_tests,
            "correct_predictions": correct_predictions,
            "prediction_accuracy_percentage": prediction_accuracy,
            "avg_similarity_precision_percentage": avg_similarity_precision,
            "median_similarity_precision_percentage": median_similarity,
            "mode_similarity_precision_percentage": mode_similarity,
            "range_similarity_percentage": range_similarity,
            "variance_similarity": variance_similarity,
            "std_dev_similarity": std_dev_similarity,
            "quartile_1": q1,
            "quartile_3": q3,
            "coefficient_of_variation_percentage": coeff_variation,
            "details": details
        }
        
        # Log summary details
        logger.info(f"Prediction accuracy: {prediction_accuracy:.2f}% ({correct_predictions}/{total_tests})")
        logger.info(f"Average similarity precision: {avg_similarity_precision:.2f}%")
        
        # Median: the middle value that splits an ordered distribution in two.
        # Useful for understanding the central tendency, especially when outliers are present.
        logger.info(f"Median similarity precision: {median_similarity:.2f}%")

        # Mode: the value or values that occur most frequently.
        # Helps identify common values or peaks in the distribution.
        logger.info(f"Mode similarity precision: {mode_similarity:.2f}%")

        # Range: difference between the maximum and minimum values.
        # Provides an indication of data dispersion.
        logger.info(f"Range similarity precision: {range_similarity:.2f}%")

        # Variance: measures the dispersion of data around the mean.
        # Calculated as the average of squared deviations from the mean.
        logger.info(f"Variance similarity: {variance_similarity:.2f}")

        # Standard deviation: square root of the variance providing dispersion in the same units as the data.
        logger.info(f"Standard deviation similarity: {std_dev_similarity:.2f}")

        # Quartiles: values that split an ordered dataset into four equal parts.
        # Q1 is the 25th percentile, the median the 50th, and Q3 the 75th percentile.
        # Useful to understand distribution and detect potential outliers.
        logger.info(f"Q1 (25th percentile): {q1:.2f}%")
        logger.info(f"Q3 (75th percentile): {q3:.2f}%")

        # Coefficient of variation: ratio of the standard deviation to the mean, usually expressed as a percentage.
        # Allows comparison of dispersion between distributions with different units or scales.
        logger.info(f"Coefficient of Variation: {coeff_variation:.2f}%")
        
        bdd.save_metrics(MetricsModel(
            model_name=dto.name,
            metrics={
                "type": "mesure",
                "total_tests": total_tests,
                "correct_predictions": correct_predictions,
                "prediction_accuracy_percentage": prediction_accuracy,
                "avg_similarity_precision_percentage": avg_similarity_precision,
                "median_similarity_precision_percentage": median_similarity,
                "mode_similarity_precision_percentage": mode_similarity,
                "range_similarity_percentage": range_similarity,
                "variance_similarity": variance_similarity,
                "std_dev_similarity": std_dev_similarity,
                "quartile_1": q1,
                "quartile_3": q3,
                "coefficient_of_variation_percentage": coeff_variation
            }
        ))

        return report
    
    except Exception as e:
        logger.error(f"Error message:{str(e)}\nStack trace:\n{traceback.format_exc()}")
        raise Exception(f"[#search_model_siamese]{str(e)}")
