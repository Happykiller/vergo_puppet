from app.repositories.memory import get_model, update_model, get_all_models
from app.models.neural_network import train_model_nn
from fastapi import HTTPException # type: ignore

def train_model(name: str, dictionary: list):
    
    model = get_model(name)

    if model is None or not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if dictionary is None:
        raise HTTPException(status_code=400, detail="No vectors provided for training")
    
    # Assurez-vous que le dictionnaire n'est pas vide
    if len(dictionary) == 0 or not dictionary:
        raise HTTPException(status_code=400, detail="Dictionary of vectors is empty")
    
    # Vérifier que tous les vecteurs ont la même longueur
    vector_size = len(dictionary[0])
    for vector in dictionary:
        if len(vector) != vector_size:
            raise HTTPException(status_code=400, detail="All vectors must have the same length")
    
    # Entraîner le réseau de neurones avec le dictionnaire de vecteurs
    nn_model, _ = train_model_nn(dictionary, vector_size)

    # Enregistrer le modèle de réseau de neurones entraîné
    update_model(name, {"nn_model": nn_model})

    return {"status": "training completed", "model_name": name}
