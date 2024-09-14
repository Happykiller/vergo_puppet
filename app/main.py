from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

app = FastAPI()

# Simuler une base de données de modèles
models = {}

# Schéma pour la création de modèle
class CreateModelData(BaseModel):
    name: str
    glossary: List[str]  # Glossaire des tokens

# Schéma pour l'entraînement du modèle
class TrainModelData(BaseModel):
    name: str
    dictionary: List[List[str]]  # Dictionnaire de vecteurs

# Schéma pour la recherche
class SearchData(BaseModel):
    name: str
    vector: List[str]  # Vecteur à rechercher

# API pour créer un modèle
@app.post("/create_model")
async def create_model(data: CreateModelData):
    if data.name in models:
        raise HTTPException(status_code=400, detail="Model already exists")
    
    # Enregistrer le modèle avec le glossaire
    models[data.name] = {"glossary": data.glossary}
    
    return {"status": "model created", "model_name": data.name}

# API pour entraîner un modèle
@app.post("/train_model")
async def train_model(data: TrainModelData):
    if data.name not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Ajouter le dictionnaire de vecteurs au modèle existant
    models[data.name]["dictionary"] = data.dictionary
    
    return {"status": "training completed", "model_name": data.name}

# API pour rechercher un vecteur dans un modèle
@app.post("/search")
async def search(data: SearchData):
    if data.name not in models:
        raise HTTPException(status_code=404, detail="Model not found")

    # Extraire le dictionnaire de vecteurs du modèle
    model_data = models[data.name]
    vector_dict = model_data.get("dictionary", [])
    
    if not vector_dict:
        raise HTTPException(status_code=400, detail="No vectors available in the model")

    # Recherche du meilleur vecteur
    best_vector = max(vector_dict, key=lambda v: len(set(data.vector) & set(v)))
    
    # Calcul de la précision
    accuracy = len(set(data.vector) & set(best_vector)) / len(set(data.vector))
    
    return {
        "search": data.vector,
        "find": best_vector,
        "stats": {
            "accuracy": accuracy
        },
        "model_used": data.name
    }
