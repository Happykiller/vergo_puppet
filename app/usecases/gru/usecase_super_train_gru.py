# app/usecases/gru/usecase_super_train_gru.py
import copy
import traceback
from typing import List, NamedTuple, Dict

from app.inversify import Inversify
from app.services.logger import logger
from app.neural_network.nn_gru import GRUClassifier
from app.services.bdd.models.model_metrics import MetricsModel
from app.services.bdd.models.model_data import ModelData, ModelStatus
from app.apis.models.gru_training_model_data import GRUTrainingModelData
from app.usecases.gru.usecase_mesure_gru import mesure_gru, MesureGRUUsecaseDto
from app.usecases.gru.usecase_train_gru import train_model_gru, TrainGRUUsecaseDto

class SuperTrainGRUUsecaseDto(NamedTuple):
    name: str
    training_data: List[GRUTrainingModelData]
    test_data: List[GRUTrainingModelData]
    inversify: Inversify
    n_iterations: int = 10

def super_train_model_gru(dto: SuperTrainGRUUsecaseDto) -> Dict:
    """
    Runs the training process followed by the measurement phase.
    This usecase performs n_iterations of training (each with early stopping) and,
    for each cycle, evaluates the model on the test set.
    The best model (in terms of test prediction accuracy) is finally saved to the BDD.
    
    :param dto: SuperTrainGRUUsecaseDto containing model name, training data, test data, etc.
    :return: Dictionary with keys "best_test_accuracy", "training_report", and "measurement_report".
    """
    try:
        best_test_accuracy = -1.0
        best_model_state = None
        best_measurement_report = None

        bdd = dto.inversify.get_bdd()

        # Update the model in storage
        model = bdd.get_model(dto.name, GRUClassifier)

        if(model.status == ModelStatus.SUPER_TRAINING):
            raise Exception("Model is training")

        model.status = ModelStatus.SUPER_TRAINING
        bdd.update_model(model)
        
        for i in range(dto.n_iterations):
            logger.info(f"Starting training iteration {i+1}/{dto.n_iterations}")

            # --- Training Phase ---
            train_dto = TrainGRUUsecaseDto(
                name=dto.name,
                training_data=dto.training_data,
                inversify=dto.inversify
            )
            train_model_gru(train_dto)
            logger.info("Training completed for iteration %d.", i+1)

            # Retrieve current model state from BDD
            current_model = bdd.get_model(dto.name, GRUClassifier)
            current_model_state = copy.deepcopy(current_model.nn_model.state_dict())

            # --- Measurement Phase ---
            measure_dto = MesureGRUUsecaseDto(
                name=dto.name,
                test_data=dto.test_data,
                inversify=dto.inversify
            )
            measurement_result = mesure_gru(measure_dto)
            current_accuracy = measurement_result.get("summary", {}).get("average_accuracy", 0)
            logger.info(f"Iteration {i+1}: Test prediction accuracy: {current_accuracy:.2f}%")

            # If performance is better, save this model state
            if current_accuracy > best_test_accuracy:
                best_test_accuracy = current_accuracy
                best_model_state = current_model_state
                best_measurement_report = measurement_result

        # --- Update BDD with best model state ---
        if best_model_state is not None:
            best_model = bdd.get_model(dto.name, GRUClassifier)
            best_model.nn_model.load_state_dict(best_model_state)
            bdd.update_model(ModelData(
                name=best_model.name,
                neural_network_type=best_model.neural_network_type,
                status = ModelStatus.SUPER_TRAINED,
                nn_model=best_model.nn_model,
                word2idx=best_model.word2idx,
                idx2word=best_model.idx2word,
                category2idx=best_model.category2idx,
                idx2category=best_model.idx2category
            ))
            logger.info("Best model updated in BDD.")

        logger.info("Super training completed. Best test accuracy: %.2f%%", best_test_accuracy)
        
        bdd.save_metrics(MetricsModel(
            model_name=dto.name,
            metrics= {
                "type": "super_training",
                "iterations": dto.n_iterations,
                "best_test_accuracy": best_test_accuracy,
                "best_measurement_report": best_measurement_report["summary"]
            }
        ))
        
        return {
            "iterations": dto.n_iterations,
            "best_test_accuracy": best_test_accuracy,
            "best_measurement_report": best_measurement_report["summary"]
        }

    except Exception as e:
        logger.error(f"Error in super training usecase: {str(e)}\nStack trace:\n{traceback.format_exc()}")
        raise Exception(f"[#super_train_model_gru]{str(e)}")
