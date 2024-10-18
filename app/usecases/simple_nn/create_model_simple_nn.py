from app.repositories.memory import model_exists, save_model
from fastapi import HTTPException  # type: ignore
from typing import List
from app.services.logger import logger

def create_model_simpleNN(name: str, neural_network_type="SimpleNN"):
    logger.info(f"Type de machine learning utilisé pour la création du model {neural_network_type}")

    if model_exists(name):
        raise HTTPException(status_code=400, detail="Model already exists")

    # Enregistrer le modèle avec le glossaire et le dictionnaire d'indices
    model_data = {
        "neural_network_type": neural_network_type # Enregistrement du type de modèle
    }
    save_model(name, model_data)

    return {"status": "model created", "model_name": name}
