from fastapi import APIRouter, HTTPException  # type: ignore
from pydantic import BaseModel, Field
from typing import List
from app.usecases.create_model import create_model
from app.usecases.train_model import train_model
from app.usecases.search import search_model
from app.usecases.getall_model import get_all_models_usecase
from app.version import __version__

# Initialisation du routeur
router = APIRouter()

# Schéma pour la création de modèle
class CreateModelData(BaseModel):
    name: str = Field(..., description="Nom du modèle à créer")
    dictionary: List[List[str]] = Field(..., description="Liste de listes de tokens pour le modèle")
    glossary: List[str] = Field(..., description="Glossaire de référence pour le modèle")

# Schéma pour l'entraînement du modèle
class TrainModelData(BaseModel):
    name: str = Field(..., description="Nom du modèle à entraîner")
    dictionary: List[List[str]] = Field(..., description="Liste de listes de tokens pour entraîner le modèle")

# Schéma pour la recherche
class SearchData(BaseModel):
    name: str = Field(..., description="Nom du modèle dans lequel effectuer la recherche")
    vector: List[str] = Field(..., description="Liste de tokens représentant le vecteur à rechercher")

# API pour créer un modèle
@router.post("/create_model")
async def create_model_api(data: CreateModelData):
    """
    Crée un nouveau modèle avec un dictionnaire de tokens et un glossaire.
    """
    try:
        return create_model(data.name, data.dictionary, data.glossary)
    except HTTPException as e:
        raise e
    except Exception as e:
        # Gestion des erreurs générales
        raise HTTPException(status_code=500, detail=f"Une erreur s'est produite : {str(e)}")

# API pour entraîner un modèle
@router.post("/train_model")
async def train_model_api(data: TrainModelData):
    """
    Entraîne un modèle existant avec un dictionnaire de tokens.
    """
    try:
        return train_model(data.name, data.dictionary)
    except HTTPException as e:
        raise e
    except Exception as e:
        # Gestion des erreurs générales
        raise HTTPException(status_code=500, detail=f"Une erreur s'est produite : {str(e)}")

# API pour rechercher un vecteur dans un modèle
@router.post("/search")
async def search_model_api(data: SearchData):
    """
    Recherche un vecteur dans le modèle spécifié.
    """
    try:
        return search_model(data.name, data.vector)
    except HTTPException as e:
        raise e
    except Exception as e:
        # Gestion des erreurs générales
        raise HTTPException(status_code=500, detail=f"Une erreur s'est produite : {str(e)}")

# Nouvelle API pour récupérer tous les modèles
@router.get("/models")
async def get_all_models_api():
    """
    Retourne tous les modèles enregistrés dans la mémoire.
    """
    try:
        return get_all_models_usecase()
    except Exception as e:
        # Gestion des erreurs générales
        raise HTTPException(status_code=500, detail=f"Une erreur s'est produite : {str(e)}")
    
@router.get("/version")
async def get_version():
    """
    Retourne la version actuelle de l'application.
    """
    return {"version": __version__}
