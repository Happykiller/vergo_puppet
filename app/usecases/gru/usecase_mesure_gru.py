# app\usecases\gru\usecase_mesure_gru.py
from typing import List, NamedTuple
from fastapi import HTTPException  # type: ignore

from app.inversify import Inversify
from app.services.logger import logger
from app.neural_network.nn_gru import predict
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
        model_data = bdd.get_model(dto.name)

        # Check if the model data is found
        if model_data is None or not model_data:
            # Raise a error if the model is not found
            raise HTTPException(status_code=404, detail="Model not found")

        # Extract the neural network model from the data
        nn_model = model_data.get("nn_model", None)
        if nn_model is None:
            raise HTTPException(status_code=400, detail="Model is not trained")
        
        result = []
        
        # Retrieve dictionaries for token and category mapping
        word2idx = model_data.get("word2idx", None)
        idx2category = model_data.get("idx2category", None)
        category2idx = model_data.get("category2idx", None)
        if word2idx is None or idx2category is None or category2idx is None:
            raise Exception("Model data is incomplete")
        
        # Initialize metrics
        total_error = 0
        correct_predictions = 0
        total_tests = len(dto.test_data)
        
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
    
    except HTTPException as e:
        logger.error(f"An error occurred during measurement: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"An error occurred during measurement: {str(e)}")
        raise Exception(f"An error occurred during measurement: {str(e)}")
