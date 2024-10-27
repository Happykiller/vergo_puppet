from app.services.logger import logger
from fastapi import HTTPException  # type: ignore
from app.repositories.memory import model_exists, save_model

def create_lstm(name: str):
    logger.info(f"Type de machine learning utilisé pour la création du model 'LSTM'")

    if model_exists(name):
        raise HTTPException(status_code=400, detail="Model already exists")

    # Enregistrer le modèle avec le glossaire et le dictionnaire d'indices
    model_data = {
        "neural_network_type": "LSTM" # Enregistrement du type de modèle
    }
    save_model(name, model_data)

    return {"status": "model created", "model_name": name}
