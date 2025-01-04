# app\usecases\gru\usecase_mesure_gru.py
import traceback
from typing import List, NamedTuple

from app.inversify import Inversify
from app.services.logger import logger
from app.neural_network.nn_gru import GRUClassifier, predict
from app.usecases.gru.usecase_commons_gru import process_input
from app.apis.models.gru_training_model_data import GRUTrainingModelData

class MesureGRUUsecaseDto(NamedTuple):
    name: str
    inversify: Inversify
    test_data: List[GRUTrainingModelData]

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
        
        result = []
        
        # Retrieve dictionaries for token and category mapping
        word2idx = model.word2idx
        idx2category = model.idx2category
        category2idx = model.category2idx
        if word2idx is None or idx2category is None or category2idx is None:
            raise Exception("Model data is incomplete")
        
        # Initialize metrics
        total_error = 0
        correct_predictions = 0
        total_tests = len(dto.test_data)

        if total_tests == 0:
            logger.info("No test data provided. Returning zero accuracy.")
            return {
                "summary": {
                    "total_tests": 0,
                    "correct_predictions": 0,
                    "total_error": 0,
                    "accuracy": 0.0
                },
                "detailed_results": []
            }
        
        y_true = []  # Ground truth categories
        y_pred = []  # Predicted categories
        
        # Iterate over each data instance in the test set
        for data in dto.test_data:
            # Prepare input data for prediction
            tokens = data.tokens
            expected_category = data.category
            input = process_input(tokens, word2idx)
            
            # Make a prediction using the model
            predicted_idx = predict(nn_model, input)
            predicted_category = idx2category[predicted_idx]
            
            # Append actual and predicted categories for analysis
            try:
                y_true.append(category2idx[expected_category])
            except KeyError:
                logger.warning(f"Unknown category in test data: '{expected_category}'. It was not seen during training.")
                continue  # Skip this sample if category is unknown
            y_pred.append(predicted_idx)
            
            # Check if the prediction is correct
            is_correct = predicted_category == expected_category
            if is_correct:
                correct_predictions += 1
            else:
                total_error += 1
            
            # Log individual prediction results
            logger.info(f"Request: {tokens}")
            logger.info(f"Expected category: {expected_category}, Predicted category: {predicted_category}")

            # Add detailed result for this test case
            result.append({
                "tokens": tokens,
                "expected_category": expected_category,
                "predicted_category": predicted_category,
                "is_correct": is_correct
            })
        
        # Summarize performance results
        logger.info(f"Number of correct predictions: {correct_predictions}/{total_tests}")
        accuracy = correct_predictions / total_tests * 100
        logger.info(f"Model accuracy rate: {accuracy:.2f}%")

        # Append summary to the result
        summary = {
            "total_tests": total_tests,
            "correct_predictions": correct_predictions,
            "total_error": total_error,
            "accuracy": accuracy
        }
        
        return {
            "summary": summary,
            "detailed_results": result
        }
    
    except Exception as e:
        logger.error(f"Error message:{str(e)}\nStack trace:\n{traceback.format_exc()}")
        raise Exception(f"[#search_model_simple_nn]{str(e)}")
