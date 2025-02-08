# app\usecases\getall_model.py
import traceback

from app.inversify import Inversify
from app.services.logger import logger
from app.services.bdd.models.model_data import ModelData

def get_model_usecase(name: str, inversify: Inversify) -> ModelData:
    """
    Retrieves model from the repository.
    - If no models are found, returns a message indicating this.
    - If models are found, returns a dictionary containing the list of models.
    
    :return: A dictionary with either a message or the list of models.
    """
    try :
        # Fetch Bdd
        bdd = inversify.get_bdd()

        # Fetch all models from the memory repository
        model = bdd.get_model(name)
        
        # Return the list of models
        return model
    except Exception as e:
        logger.error(f"Error message:{str(e)}\nStack trace:\n{traceback.format_exc()}")
        raise Exception(f"[get_model]{str(e)}")