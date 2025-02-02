# app/usecases/siamese/usecase_super_train_siamese.py
import traceback
import copy
from typing import List, NamedTuple, Dict, Tuple

from app.inversify import Inversify
from app.services.logger import logger
from app.services.bdd.models.model_data import ModelData
from app.neural_network.nn_siamese import SiameseLSTM
from app.usecases.siamese.usecase_train_siamese import train_model_siamese, TrainSiameseUsecaseDto
from app.usecases.siamese.usecase_mesure_siamese import mesure_siamese, MesureSiameseUsecaseDto

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

        bdd = dto.inversify.get_bdd()
        
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

            # Retrieve current model state from BDD (the training usecase met à jour le modèle)
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

            # Si la performance est meilleure, sauvegarder cet état et les rapports associés
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
                nn_model=best_model.nn_model,
                dictionary=best_model.dictionary,
                indexed_dictionary=best_model.indexed_dictionary,
                glossary=best_model.glossary
            ))
            logger.info("Best model updated in BDD.")

        final_report = {
            "best_test_accuracy": best_test_accuracy,
            "training_report": best_training_report,
            "measurement_report": best_measurement_report
        }
        logger.info("Super training completed. Best test accuracy: %.2f%%", best_test_accuracy)
        return final_report

    except Exception as e:
        logger.error(f"Error in super training usecase: {str(e)}\nStack trace:\n{traceback.format_exc()}")
        raise Exception(f"[#super_train_model_siamese]{str(e)}")
