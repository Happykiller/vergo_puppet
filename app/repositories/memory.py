# Simulation de la base de données de modèles
models = {}

def save_model(name: str, data: dict):
    models[name] = data

def get_model(name: str):
    return models.get(name)

def model_exists(name: str):
    return name in models

def update_model(name: str, data: dict):
    models[name].update(data)

def get_all_models():
    return models