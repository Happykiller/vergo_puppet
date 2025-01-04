# app\usecases\simple\usecase_train_simple.py
import traceback
from typing import List, NamedTuple

from app.inversify import Inversify
from app.services.logger import logger
from app.services.bdd.models.model_data import ModelData
from app.neural_network.nn_simple import SimpleNN, train_model_nn
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
    try:
        # Fetch BDD
        bdd = dto.inversify.get_bdd()

        # Retrieve the model configuration
        model = bdd.get_model(dto.name, SimpleNN)

        if model is None or not model:
            raise Exception("Model not found")

        if dto.training_data is None or len(dto.training_data) == 0:
            raise Exception("No training data provided or training data is empty")

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

        # Update the model with the trained neural network and additional metadata
        bdd.update_model(ModelData(
            name=model.name,
            neural_network_type=model.neural_network_type,
            nn_model=nn_model,
            encoder=encoder,
            scaler=scaler,
            indices={"categorical_indices": categorical_indices, "numerical_indices": numerical_indices},
            targets_mean=targets_mean,
            targets_std=targets_std,
        ))

        return {"status": "training completed", "model_name": dto.name}
    
    except Exception as e:
        logger.error(f"Error message:{str(e)}\nStack trace:\n{traceback.format_exc()}")
        raise Exception(f"[#train_model_simple_nn]{str(e)}")