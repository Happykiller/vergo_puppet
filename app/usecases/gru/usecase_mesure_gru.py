# app\usecases\gru\usecase_mesure_gru.py
import traceback
from typing import List, NamedTuple, Optional

from app.inversify import Inversify
from app.services.bdd.models.model_metrics import MetricsModel
from app.services.logger import logger
from app.neural_network.nn_gru import GRUClassifier, predict
from app.usecases.gru.usecase_commons_gru import process_input
from app.apis.models.gru_training_model_data import GRUTrainingModelData

class MesureGRUUsecaseDto(NamedTuple):
    name: str
    inversify: Inversify
    test_data: List[GRUTrainingModelData]
    iterate: Optional[int] = 10

def mesure_gru(dto: MesureGRUUsecaseDto):
    """
    Measures the performance of a GRU model on provided test data.
    :param name: The name of the model.
    :param test_data: A list of test data instances.
    """
    try:
        # Fetch Bdd
        bdd = dto.inversify.get_bdd()

        # Retrieve model data from the in-memory repository
        model = bdd.get_model(dto.name, GRUClassifier)

        # Check if the model data is found
        if model is None or not model:
            # Raise a error if the model is not found
            raise Exception("Model not found")

        # Extract the neural network model from the data
        nn_model = model.nn_model
        if nn_model is None:
            raise Exception("Model is not trained")
        
        # Retrieve dictionaries for token and category mapping
        word2idx = model.word2idx
        idx2category = model.idx2category
        category2idx = model.category2idx
        if word2idx is None or idx2category is None or category2idx is None:
            raise Exception("Model data is incomplete")
        
        def run_test():
            """Runs a single iteration of the test and returns accuracy and detailed results."""
            total_error = 0
            correct_predictions = 0
            total_tests = len(dto.test_data)
            detailed_results = []

            if total_tests == 0:
                return 0.0, []

            for data in dto.test_data:
                tokens = data.tokens
                expected_category = data.category
                input = process_input(tokens, word2idx)
                predicted_idx = predict(nn_model, input)
                predicted_category = idx2category[predicted_idx]

                try:
                    category2idx[expected_category]
                except KeyError:
                    logger.warning(f"Unknown category in test data: '{expected_category}'. Skipping sample.")
                    continue

                is_correct = predicted_category == expected_category
                if is_correct:
                    correct_predictions += 1
                else:
                    total_error += 1

                detailed_results.append({
                    "tokens": tokens,
                    "expected_category": expected_category,
                    "predicted_category": predicted_category,
                    "is_correct": is_correct
                })

            accuracy = correct_predictions / total_tests * 100
            return accuracy, detailed_results

        logger.info(f"Running tests {dto.iterate} times.")
        accuracies = []
        all_detailed_results = []

        for _ in range(dto.iterate):
            accuracy, detailed_results = run_test()
            accuracies.append(accuracy)
            all_detailed_results.append({
                "iteration": len(all_detailed_results) + 1,
                "accuracy": accuracy,
                "detailed_results": detailed_results
            })

        avg_accuracy = sum(accuracies) / len(accuracies)
        min_accuracy = min(accuracies)
        max_accuracy = max(accuracies)
        
        summary = {
            "type": "mesure",
            "iterations": dto.iterate,
            "average_accuracy": avg_accuracy,
            "min_accuracy": min_accuracy,
            "max_accuracy": max_accuracy
        }
        
        bdd.save_metrics(MetricsModel(
            model_name=dto.name,
            metrics=summary
        ))
        
        logger.info(f"Summary after {dto.iterate} iterations: {summary}")
        return {
            "summary": summary,
            "history": all_detailed_results
        }
    
    except Exception as e:
        logger.error(f"Error message:{str(e)}\nStack trace:\n{traceback.format_exc()}")
        raise Exception(f"[#mesure_gru]{str(e)}")
