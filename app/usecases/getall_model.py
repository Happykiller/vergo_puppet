# app\usecases\getall_model.py
import traceback
from typing import List

from app.inversify import Inversify
from app.services.logger import logger
from app.services.bdd.models.model_data import ModelData

def get_all_models_usecase(inversify: Inversify) -> List[ModelData]:
    """
    Retrieves all models from the repository.
    - If no models are found, returns a message indicating this.
    - If models are found, returns a dictionary containing the list of models.
    
    :return: A dictionary with either a message or the list of models.
    """
    try:
        # Fetch BDD
        bdd = inversify.get_bdd()

        # Fetch all models from the database
        models = bdd.get_all_models()
        models_with_results = []

        for model in models:
            # Ensure that `model` is an instance of `ModelData`
            if not isinstance(model, ModelData):
                logger.warning(f"Invalid model format: expected ModelData but got {type(model)}")
                continue 

            # Fetch training results and sort them by timestamp (most recent first)
            metrics = sorted(
                bdd.get_metrics(model.name),
                key=lambda result: result.timestamp,
                reverse=True
            )[:50]

            # Convert training results to dictionaries
            metrics_dicts = [result.to_dict() for result in metrics]
            
            filtered_metrics = [
                {
                    "metrics": result["metrics"],
                    "timestamp": result["timestamp"]
                }
                for result in metrics_dicts
            ]

            # Append model with training results
            models_with_results.append({
                "name": model.name,
                "neural_network_type": model.neural_network_type,
                "status": model.status.value,
                "history": filtered_metrics
            })

        return models_with_results
    except Exception as e:
        logger.error(f"Error message:{str(e)}\nStack trace:\n{traceback.format_exc()}")
        raise Exception(f"[get_all_models_usecase]{str(e)}")