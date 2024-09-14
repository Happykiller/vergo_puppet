from fastapi import APIRouter # type: ignore
from pydantic import BaseModel
from typing import List
from app.usecases.create_model import create_model
from app.usecases.train_model import train_model
from app.usecases.search import search_model
from app.usecases.getall_model import get_all_models_usecase

router = APIRouter()

# Schéma pour la création de modèle
class CreateModelData(BaseModel):
    name: str
    glossary: List[str]

# Schéma pour l'entraînement du modèle
class TrainModelData(BaseModel):
    name: str
    dictionary: List[List[str]]

# Schéma pour la recherche
class SearchData(BaseModel):
    name: str
    vector: List[str]

# API pour créer un modèle
@router.post("/create_model")
async def create_model_api(data: CreateModelData):
    return create_model(data.name, data.glossary)

# API pour entraîner un modèle
@router.post("/train_model")
async def train_model_api(data: TrainModelData):
    return train_model(data.name, data.dictionary)

# API pour rechercher un vecteur dans un modèle
@router.post("/search")
async def search_model_api(data: SearchData):
    return search_model(data.name, data.vector)

# Nouvelle API pour récupérer tous les modèles
@router.get("/models")
async def get_all_models_api():
    return get_all_models_usecase()
