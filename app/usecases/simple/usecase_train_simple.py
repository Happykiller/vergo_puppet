# app\usecases\simple\usecase_train_simple.py
import io
import joblib
from typing import List, NamedTuple

from app.inversify import Inversify
from app.services.logger import logger
from app.neural_network.nn_simple import train_model_nn
from app.usecases.simple.usecase_commons_simple import transform_data
from app.apis.models.simple_nn_training_model_data import SimpleNNTrainingModelData


class TrainSimpleUsecaseDto(NamedTuple):
    name: str
    training_data: List[SimpleNNTrainingModelData]
    inversify: Inversify


def train_model_simple_nn(dto: TrainSimpleUsecaseDto):
    """
    Train the SimpleNN model with the provided training data.
    """
    # Fetch BDD
    bdd = dto.inversify.get_bdd()

    # Retrieve the model configuration
    model = bdd.get_model(dto.name)

    if model is None or not model:
        raise Exception(status_code=404, detail="Model not found")

    if dto.training_data is None or len(dto.training_data) == 0:
        raise Exception(status_code=400, detail="No training data provided or training data is empty")

    logger.info("Machine learning type used for training: SimpleNN")

    # Transform the data
    (
        features_processed,
        targets_standardized,
        encoder,
        scaler,
        targets_mean,
        targets_std,
        categorical_indices,
        numerical_indices,
    ) = transform_data(dto.training_data)

    # Determine the input size for the neural network
    input_size = features_processed.shape[1]

    # Train the neural network
    nn_model, _ = train_model_nn(features_processed, targets_standardized, input_size)

    # Serialize encoder, scaler, and indices to memory
    encoder_buffer = io.BytesIO()
    joblib.dump(encoder, encoder_buffer)
    encoder_serialized = encoder_buffer.getvalue()

    scaler_buffer = io.BytesIO()
    joblib.dump(scaler, scaler_buffer)
    scaler_serialized = scaler_buffer.getvalue()

    indices_info = {
        "categorical_indices": categorical_indices,
        "numerical_indices": numerical_indices,
    }
    indices_buffer = io.BytesIO()
    joblib.dump(indices_info, indices_buffer)
    indices_serialized = indices_buffer.getvalue()

    # Update the model with the trained neural network and additional metadata
    bdd.update_model(
        dto.name,
        {
            "nn_model": nn_model,  # Le modèle sera sérialisé séparément dans MongoBDDService
            "encoder": encoder_serialized,
            "scaler": scaler_serialized,
            "indices": indices_serialized,
            "targets_mean": targets_mean,
            "targets_std": targets_std,
        },
    )

    return {"status": "training completed", "model_name": dto.name}
