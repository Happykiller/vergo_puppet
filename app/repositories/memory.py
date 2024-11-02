#app\repositories\memory.py
# Simulated in-memory database for storing models
models = {}

def save_model(name: str, data: dict):
    """
    Save a model with the given name and data to the in-memory storage.
    :param name: The name of the model to save.
    :param data: A dictionary containing the model's data.
    """
    models[name] = data

def get_model(name: str):
    """
    Retrieve a model by its name from the in-memory storage.
    :param name: The name of the model to retrieve.
    :return: The model data if found, otherwise None.
    """
    return models.get(name)

def model_exists(name: str):
    """
    Check if a model with the specified name exists in the in-memory storage.
    :param name: The name of the model to check.
    :return: True if the model exists, otherwise False.
    """
    return name in models

def update_model(name: str, data: dict):
    """
    Update an existing model's data with new information.
    :param name: The name of the model to update.
    :param data: A dictionary containing the updated model data.
    """
    models[name].update(data)

def get_all_models():
    """
    Retrieve all models from the in-memory storage.
    :return: A dictionary containing all models.
    """
    return models
