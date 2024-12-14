#app\usecases\simple\usecase_mesure_simple.py
import joblib
from app.services.logger import logger
from app.repositories.memory import get_model
from app.neural_network.nn_simple import predict
from app.usecases.simple.usecase_commons_simple import process_input_data

def mesure_simple_nn(name, test_data):
    try:
        total_error = 0
        total_percentage_error = 0
        correct_predictions = 0
        total_tests = len(test_data)
        model = get_model(name)
        nn_model = model.get("nn_model", None)
        if nn_model is None:
            raise Exception("Model not trained yet")
        
        # Load encoder, scaler, and indices
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
        
        # Retrieve normalization parameters for targets
        targets_mean = model.get("targets_mean")
        targets_std = model.get("targets_std")
        if targets_mean is None or targets_std is None:
            raise Exception("Missing normalization parameters in the model")
        
        for data in test_data:
            # Prepare input data
            input_data = [
                data.type,
                data.surface,
                data.pieces,
                data.floor,
                data.parking,
                data.balcon,
                data.ascenseur,
                data.orientation,
                data.transports,
                data.neighborhood
            ]
            expected = data.price
            
            # Transform input data
            input_processed = process_input_data(input_data, encoder, scaler, categorical_indices, numerical_indices)
            
            # Prediction
            predicted = predict(nn_model, input_processed, targets_mean, targets_std)
            
            # Calculate error
            error = abs(predicted - expected)
            percentage_error = (error / expected) * 100
            total_error += error
            total_percentage_error += percentage_error
            
            # Consider prediction correct if error is within 10% of the actual price
            if percentage_error <= 10:
                correct_predictions += 1
            
            # Log output
            logger.info(f"Request: {input_data}")
            logger.info(f"Expected price: {expected}€, Model price: {predicted:.2f}€, Error: {error:.2f}€, Percentage error: {percentage_error:.2f}%")
        
        avg_error = total_error / total_tests
        avg_percentage_error = total_percentage_error / total_tests
        # Log the number of correct predictions out of the total test cases
        logger.info(f"Number of correct predictions: {correct_predictions}/{total_tests}")
        logger.info(f"Mean Absolute Error (MAE) on the test set: {avg_error:.2f}€")
        logger.info(f"Mean Absolute Percentage Error (MAPE) on the test set: {avg_percentage_error:.2f}%")
    except Exception as e:
        logger.error(f"An error occurred during measurement: {str(e)}")
        raise Exception(f"An error occurred during measurement: {str(e)}")
