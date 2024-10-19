from app.apis.models.simple_nn_search_data import SimpleNNSearchData
from app.usecases.mesure_siamese import mesure_siamese
from app.usecases.simple_nn.create_model_simple_nn import create_model_simpleNN
from app.usecases.simple_nn.mesure_simple_nn import mesure_simple_nn
from app.usecases.simple_nn.search_model_simple_nn import search_model_simple_nn
from app.usecases.simple_nn.train_model_simple_nn import train_model_simple_nn
from fastapi import APIRouter, HTTPException  # type: ignore
from pydantic import BaseModel, Field
from app.apis.models.simple_nn_training_data import SimpleNNTrainingData
from typing import List, Optional, Tuple, Union
from app.usecases.create_model import create_model
from app.usecases.train_model import train_model
from app.usecases.search import search_model
from app.usecases.getall_model import get_all_models_usecase
from app.services.logger import logger
from app.version import __version__

# Initialisation du routeur
router = APIRouter()

# Schéma pour la création de modèle
class CreateModelData(BaseModel):
    name: str = Field(..., description="Nom du modèle à créer")
    dictionary: Optional[List[List[str]]] = Field(None, description="Liste de listes de tokens pour le modèle (obligatoire sauf pour SimpleNN)")
    glossary: Optional[List[str]] = Field(None, description="Glossaire de référence pour le modèle (obligatoire sauf pour SimpleNN)")
    neural_network_type: str = Field(default="SimpleNN", description="Type de réseau de neurones ('SimpleNN' ou 'LSTMNN' ou 'SIAMESE')")

    def check_required_fields(cls, values):
        neural_network_type = values.get('neural_network_type')
        dictionary = values.get('dictionary')
        glossary = values.get('glossary')

        if neural_network_type != "SimpleNN":
            if dictionary is None or glossary is None:
                raise ValueError("Les champs 'dictionary' et 'glossary' sont obligatoires sauf pour 'SimpleNN'.")

        return values

# Schéma pour l'entraînement du modèle
class TrainModelData(BaseModel):
    name: str = Field(..., description="Nom du modèle à entraîner")
    neural_network_type: Optional[str] = Field(description="Type de réseau de neurones ('SimpleNN' ou 'LSTMNN' ou 'SIAMESE')")
    training_data: Union[
        List[ Tuple[List[str], List[str]] ],
        List[ Tuple[List[str], List[str], Optional[int]] ],
        List[ SimpleNNTrainingData ]
    ] = Field (
        ..., description="Liste de tuples (input, target) | (siamese1, siamese2, target) pour entraîner le modèle"
    )

    def __init__(self, **data):
        super().__init__(**data)
        if(self.neural_network_type == 'SIAMESE'):
            # Si le dernier élément du tuple n'est pas défini, on lui attribue une valeur par défaut
            for i, elem in enumerate(self.training_data):
                # Vérification de la longueur du tuple
                if len(elem) == 2:
                    siamese1, siamese2 = elem
                    self.training_data[i] = (siamese1, siamese2)
                elif len(elem) == 3:
                    siamese1, siamese2, target = elem
                    if target is None:
                        self.training_data[i] = (siamese1, siamese2, 0.0)

class TestModelData(BaseModel):
    name: str = Field(..., description="Nom du modèle à tester")
    neural_network_type: str = Field(..., description="Type de réseau de neurones ('SimpleNN', 'LSTMNN', 'SIAMESE')")
    test_data: Union[
        List[ Tuple[List[str], List[str]] ],
        List[ Tuple[List[str], List[str], Optional[int]] ],
        List[ SimpleNNTrainingData ]
    ] = Field(..., description="Données de test")

    def validate_test_data(cls, values):
        network_type = values.get('neural_network_type')
        test_data = values.get('test_data')

        if network_type == "SIAMESE":
            if not all(len(entry) == 3 for entry in test_data):
                raise ValueError("Pour un réseau SIAMESE, test_data doit contenir des tuples (siamese1, siamese2, target)")
        else:
            if not all(len(entry) == 2 for entry in test_data):
                raise ValueError("Pour les réseaux SimpleNN et LSTMNN, test_data doit contenir des tuples (input, target)")

        return values


# Schéma pour la recherche
class SearchData(BaseModel):
    name: str = Field(..., description="Nom du modèle dans lequel effectuer la recherche")
    neural_network_type: str = Field(..., description="Type de réseau de neurones ('SimpleNN', 'LSTMNN', 'SIAMESE')")
    vector: Union[
        List[str],
        SimpleNNSearchData
    ] = Field(..., description="Liste de tokens représentant le vecteur à rechercher")

# API pour créer un modèle
@router.post("/create_model")
async def create_model_api(data: CreateModelData):
    """
    Crée un nouveau modèle avec un dictionnaire de tokens et un glossaire.
    """
    try:
        if (data.neural_network_type == 'SimpleNN') :
            return create_model_simpleNN(data.name, data.neural_network_type)
        else:
            return create_model(data.name, data.dictionary, data.glossary, data.neural_network_type)
    except HTTPException as e:
        raise e
    except Exception as e:
        # Gestion des erreurs générales
        logger.error(f"Une erreur s'est produite pendant la création : {str(e)}")
        raise HTTPException(status_code=500, detail=f"Une erreur s'est produite pendant la création : {str(e)}")

# API pour entraîner un modèle
@router.post("/train_model")
async def train_model_api(data: TrainModelData):
    """
    Entraîne un modèle existant avec des tuples (input, target).
    """
    try:
        if (data.neural_network_type == 'SimpleNN') :
            return train_model_simple_nn(data.name, data.training_data)
        else:
            return train_model(data.name, data.training_data)
    except HTTPException as e:
        raise e
    except Exception as e:
        # Gestion des erreurs générales
        logger.error(f"Une erreur s'est produite pendant l'entrainement : {str(e)}")
        raise HTTPException(status_code=500, detail=f"Une erreur s'est produite pendant l'entrainement : {str(e)}")

# API pour rechercher un vecteur dans un modèle
@router.post("/search")
async def search_model_api(data: SearchData):
    """
    Recherche un vecteur dans le modèle spécifié.
    """
    try:
        if (data.neural_network_type == 'SimpleNN') :
            return search_model_simple_nn(data.name, data.vector)
        else:
            return search_model(data.name, data.vector)
    except HTTPException as e:
        raise e
    except Exception as e:
        # Gestion des erreurs générales
        logger.error(f"Une erreur s'est produite pendant la recherche : {str(e)}")
        raise HTTPException(status_code=500, detail=f"Une erreur s'est produite pendant la recherche : {str(e)}")
    
# API pour tester un modèle
@router.post("/test")
async def test(data: TestModelData):
    """
    Test un modèle spécifié.
    """
    try:
        if (data.neural_network_type == 'SIAMESE') :
            return mesure_siamese(data.name, data.test_data)
        elif (data.neural_network_type == 'SimpleNN') :
            return mesure_simple_nn(data.name, data.test_data)
        else :
            raise HTTPException(status_code=400, detail="Model type not supported yet")
    except HTTPException as e:
        raise e
    except Exception as e:
        # Gestion des erreurs générales
        logger.error(f"Une erreur s'est produite pendant le test : {str(e)}")
        raise HTTPException(status_code=500, detail=f"Une erreur s'est produite pendant le test : {str(e)}")

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
        raise HTTPException(status_code=500, detail=f"Une erreur s'est produite le listing : {str(e)}")
    
@router.get("/version")
async def get_version():
    """
    Retourne la version actuelle de l'application.
    """
    return {"version": __version__}