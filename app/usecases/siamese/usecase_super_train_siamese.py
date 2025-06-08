# app/usecases/siamese/usecase_super_train_siamese.py
import copy
import time
import traceback
from typing import List, NamedTuple, Dict, Tuple

from app.common import format_time
from app.inversify import Inversify
from app.services.logger import logger
from app.neural_network.nn_siamese import SiameseLSTM
from app.services.bdd.models.model_metrics import MetricsModel
from app.services.bdd.models.model_data import ModelData, ModelStatus
from app.usecases.siamese.usecase_mesure_siamese import mesure_siamese, MesureSiameseUsecaseDto
from app.usecases.siamese.usecase_train_siamese import train_model_siamese, TrainSiameseUsecaseDto

class SuperTrainSiameseUsecaseDto(NamedTuple):
    name: str
    training_data: List[Tuple[List[str], List[str], float]]
    test_data: List[List[str]]
    inversify: Inversify
    n_iterations: int = 30

def super_train_model_siamese(dto: SuperTrainSiameseUsecaseDto) -> Dict:
    """
    Runs the training process followed by the measurement phase.
    This usecase performs n_iterations of training (each with early stopping) and,
    for each cycle, evaluates the model on the test set.
    The best model (in terms of test prediction accuracy) is finally saved to the BDD.
    
    :param dto: SuperTrainSiameseUsecaseDto containing model name, training data, test data, etc.
    :return: Dictionary with keys "best_test_accuracy", "training_report", and "measurement_report".
    """
    try:
        best_test_accuracy = -1.0
        best_model_state = None
        best_training_report = None
        best_measurement_report = None
        start_time = time.time()

        bdd = dto.inversify.get_bdd()

        # Update the model in storage
        model = bdd.get_model(dto.name, SiameseLSTM)

        if(model.status == ModelStatus.SUPER_TRAINING):
            raise Exception("Model is training")

        model.status = ModelStatus.SUPER_TRAINING
        bdd.update_model(model)
        
        for i in range(dto.n_iterations):
            logger.info(f"Starting training iteration {i+1}/{dto.n_iterations}")

            # --- Training Phase ---
            train_dto = TrainSiameseUsecaseDto(
                name=dto.name,
                training_data=dto.training_data,
                inversify=dto.inversify
            )
            training_result = train_model_siamese(train_dto)
            logger.info("Training completed for iteration %d.", i+1)

            # Retrieve current model state from the database (the training usecase updated the model)
            current_model = bdd.get_model(dto.name, SiameseLSTM)
            current_model_state = copy.deepcopy(current_model.nn_model.state_dict())

            # --- Measurement Phase ---
            measure_dto = MesureSiameseUsecaseDto(
                name=dto.name,
                test_data=dto.test_data,
                inversify=dto.inversify
            )
            measurement_result = mesure_siamese(measure_dto)
            current_accuracy = measurement_result.get("prediction_accuracy_percentage", 0)
            logger.info(f"Iteration {i+1}: Test prediction accuracy: {current_accuracy:.2f}%")

            # If performance is better, save this state and the associated reports
            if current_accuracy > best_test_accuracy:
                best_test_accuracy = current_accuracy
                best_model_state = current_model_state
                best_training_report = training_result
                best_measurement_report = measurement_result

        # --- Update BDD with best model state ---
        if best_model_state is not None:
            best_model = bdd.get_model(dto.name, SiameseLSTM)
            best_model.nn_model.load_state_dict(best_model_state)
            bdd.update_model(ModelData(
                name=best_model.name,
                neural_network_type=best_model.neural_network_type,
                status = ModelStatus.SUPER_TRAINED,
                nn_model=best_model.nn_model,
                dictionary=best_model.dictionary,
                indexed_dictionary=best_model.indexed_dictionary,
                glossary=best_model.glossary
            ))
            logger.info("Best model updated in BDD.")
            
        end_time = time.time()
        total_time = end_time - start_time

        final_report = {
            "best_test_accuracy": best_test_accuracy,
            "training_report": best_training_report,
            "measurement_report": best_measurement_report,
            "total_time": total_time
        }
        
        logger.info("Super training completed. Best test accuracy: %.2f%%", best_test_accuracy)
        logger.info(f"Total super training time: {format_time(total_time)}")
        
        bdd.save_metrics(MetricsModel(
            model_name=dto.name,
            metrics= {
                "type": "super_training",
                "iterations": dto.n_iterations,
                "best_test_accuracy": best_test_accuracy,
                "best_measurement_report": best_measurement_report
            }
        ))
        
        return final_report

    except Exception as e:
        logger.error(f"Error in super training usecase: {str(e)}\nStack trace:\n{traceback.format_exc()}")
        raise Exception(f"[#super_train_model_siamese]{str(e)}")
