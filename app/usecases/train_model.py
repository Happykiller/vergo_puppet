from app.repositories.memory import get_model, update_model, get_all_models
from app.models.neural_network import train_model_nn
from fastapi import HTTPException # type: ignore

def train_model(name: str, training_data: list):
    
    model = get_model(name)

    if model is None or not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if training_data is None:
        raise HTTPException(status_code=400, detail="No vectors provided for training")
    
    # Assurez-vous que le dictionnaire n'est pas vide
    if len(training_data) == 0 or not training_data:
        raise HTTPException(status_code=400, detail="Dictionary of vectors is empty")
    
    # Vérifier que tous les vecteurs ont la même longueur
    vector_size = len(training_data[0])
    for vector in training_data:
        if len(vector) != vector_size:
            raise HTTPException(status_code=400, detail="All vectors must have the same length")
    
    # Entraîner le réseau de neurones avec le dictionnaire de vecteurs
    nn_model, _ = train_model_nn(training_data, vector_size)

    # Enregistrer le modèle de réseau de neurones entraîné
    update_model(name, {"nn_model": nn_model})

    return {"status": "training completed", "model_name": name}
