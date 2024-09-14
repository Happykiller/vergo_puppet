from app.repositories.memory import get_model
from fastapi import HTTPException # type: ignore

def search_model(name: str, vector: list):
    model = get_model(name)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Extraire le dictionnaire de vecteurs du modèle
    vector_dict = model.get("dictionary", [])
    if not vector_dict:
        raise HTTPException(status_code=400, detail="No vectors available in the model")

    # Recherche du meilleur vecteur
    best_vector = max(vector_dict, key=lambda v: len(set(vector) & set(v)))

    # Calcul de la précision
    accuracy = len(set(vector) & set(best_vector)) / len(set(vector))

    return {
        "search": vector,
        "find": best_vector,
        "stats": {
            "accuracy": accuracy
        },
        "model_used": name
    }
