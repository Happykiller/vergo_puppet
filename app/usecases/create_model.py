from app.repositories.memory import model_exists, save_model
from fastapi import HTTPException # type: ignore

def create_model(name: str, vector_dict: list):
    if model_exists(name):
        raise HTTPException(status_code=400, detail="Model already exists")

    # Enregistrer le modèle avec le glossaire
    model_data = {"vector_dict": vector_dict}
    save_model(name, model_data)

    return {"status": "model created", "model_name": name}
