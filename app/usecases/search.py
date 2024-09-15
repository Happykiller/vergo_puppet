from app.repositories.memory import get_model
from app.models.neural_network import predict
from fastapi import HTTPException # type: ignore
import torch

def search_model(name: str, vector: list):
    model = get_model(name)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    print(model)

    nn_model = model.get("nn_model", None)
    if not nn_model:
        raise HTTPException(status_code=400, detail="No neural network model found in the model")

    # Vérifiez si le dictionnaire de vecteurs est vide
    vector_dict = model.get("vector_dict", [])
    if not vector_dict:
        raise HTTPException(status_code=400, detail="No vectors available in the model")

    # Utiliser le réseau de neurones pour prédire le vecteur le plus proche
    predicted_vector = predict(nn_model, vector)

    # Recherche du meilleur vecteur en vérifiant que vector_dict n'est pas vide
    try:
        best_vector = max(vector_dict, key=lambda v: torch.dist(torch.Tensor(v), predicted_vector).item())
    except ValueError:
        raise HTTPException(status_code=400, detail="No valid vectors to compare")

    accuracy = 1 / torch.dist(torch.Tensor(best_vector), predicted_vector).item()

    return {
        "search": vector,
        "find": best_vector,
        "stats": {
            "accuracy": accuracy
        },
        "model_used": name
    }
