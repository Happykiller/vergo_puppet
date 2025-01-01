# app\usecases\simple\usecase_mesure_simple.py
import joblib
from typing import Any, Dict, List, NamedTuple

from app.inversify import Inversify
from app.services.logger import logger
from app.neural_network.nn_simple import predict
from app.usecases.simple.usecase_commons_simple import process_input_data
from app.apis.models.simple_nn_training_model_data import SimpleNNTrainingModelData

class MesureSimpleUsecaseDto(NamedTuple):
    name: str
    test_data: List[SimpleNNTrainingModelData]
    inversify: Inversify

def mesure_simple_nn(dto: MesureSimpleUsecaseDto) -> Dict[str, Any]:
    try:
        # Fetch Bdd
        bdd = dto.inversify.get_bdd()

        results = []  # Store individual test results
        total_error = 0
        total_percentage_error = 0
        correct_predictions = 0
        total_tests = len(dto.test_data)
        
        # Retrieve model and associated parameters
        model = bdd.get_model(dto.name)
        nn_model = model.get("nn_model", None)
        if nn_model is None:
            raise Exception("Model not trained yet")
        
        encoder_filename = model.get("encoder_filename")
        scaler_filename = model.get("scaler_filename")
        indices_filename = model.get("indices_filename")
        if not encoder_filename or not scaler_filename or not indices_filename:
            raise Exception("Missing encoder, scaler, or indices in the model")
        
        encoder = joblib.load(encoder_filename)
        scaler = joblib.load(scaler_filename)
        indices_info = joblib.load(indices_filename)
        categorical_indices = indices_info["categorical_indices"]
        numerical_indices = indices_info["numerical_indices"]
        
        targets_mean = model.get("targets_mean")
        targets_std = model.get("targets_std")
        if targets_mean is None or targets_std is None:
            raise Exception("Missing normalization parameters in the model")
        
        for data in dto.test_data:
            input_data = [
                data.type, data.surface, data.pieces, data.floor, data.parking,
                data.balcon, data.ascenseur, data.orientation, data.transports, data.neighborhood
            ]
            expected = data.price
            
            # Process input
            input_processed = process_input_data(input_data, encoder, scaler, categorical_indices, numerical_indices)
            predicted = predict(nn_model, input_processed, targets_mean, targets_std)
            
            # Calculate error
            error = abs(predicted - expected)
            percentage_error = (error / expected) * 100
            total_error += error
            total_percentage_error += percentage_error
            
            # Determine if the prediction is correct
            is_correct = percentage_error <= 10
            if is_correct:
                correct_predictions += 1
            
            # Log details for each prediction
            results.append({
                "input": input_data,
                "expected_price": expected,
                "predicted_price": predicted,
                "error": error,
                "percentage_error": percentage_error,
                "is_correct": is_correct,
            })
        
        # Calculate global metrics
        avg_error = total_error / total_tests
        avg_percentage_error = total_percentage_error / total_tests
        
        # Global logs
        logger.info(f"Number of correct predictions: {correct_predictions}/{total_tests}")
        logger.info(f"Mean Absolute Error (MAE): {avg_error:.2f}€")
        logger.info(f"Mean Absolute Percentage Error (MAPE): {avg_percentage_error:.2f}%")
        
        # Return a summary of results and metrics
        return {
            "results": results,
            "metrics": {
                "total_tests": total_tests,
                "correct_predictions": correct_predictions,
                "mean_absolute_error": avg_error,
                "mean_absolute_percentage_error": avg_percentage_error,
            },
        }
    except Exception as e:
        logger.error(f"An error occurred during measurement: {str(e)}")
        raise Exception(f"An error occurred during measurement: {str(e)}")
