from app.repositories.memory import get_model, update_model
from fastapi import HTTPException # type: ignore

def train_model(name: str, dictionary: list):
    model = get_model(name)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Ajouter le dictionnaire de vecteurs au modèle existant
    update_model(name, {"dictionary": dictionary})

    return {"status": "training completed", "model_name": name}
