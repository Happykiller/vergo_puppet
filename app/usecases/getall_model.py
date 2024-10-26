from app.repositories.memory import get_all_models

def get_all_models_usecase():
    models = get_all_models()
    if not models:
        return {"message": "No models found"}
    
    return {"models": models}
