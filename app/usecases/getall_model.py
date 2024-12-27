# app\usecases\getall_model.py
from app.inversify import Inversify

def get_all_models_usecase(inversify: Inversify):
    """
    Retrieves all models from the repository.
    - If no models are found, returns a message indicating this.
    - If models are found, returns a dictionary containing the list of models.
    
    :return: A dictionary with either a message or the list of models.
    """
    # Fetch Bdd
    bdd = inversify.get_bdd()

    # Fetch all models from the memory repository
    models = bdd.get_all_models()
    
    # Check if the models list is empty
    if not models:
        return {"message": "No models found"}
    
    # Return the list of models
    return {"models": models}
