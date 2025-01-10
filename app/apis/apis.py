# app\apis\apis.py
import jwt
from fastapi.security import OAuth2PasswordBearer # type: ignore
from fastapi import Depends, APIRouter, HTTPException # type: ignore

from app.version import __version__
from app.common import load_env_vars
from app.services.logger import logger
from app.inversify import get_inversify
from app.apis.models.test_model_data import TestModelData
from app.usecases.usecase_tokenize import usecase_tokenize
from app.apis.models.train_model_data import TrainModelData
from app.usecases.getall_model import get_all_models_usecase
from app.apis.models.create_model_data import CreateModelData
from app.apis.models.update_model_data import UpdateModelData
from app.apis.models.search_model_data import SearchModelData
from app.apis.models.prepare_cache_data import PrepareCacheData
from app.apis.models.tokenize_model_data import TokenizeModelData
from app.usecases.gru.usecase_mesure_gru import MesureGRUUsecaseDto, mesure_gru
from app.usecases.lstm.usecase_train_lstm import TrainLSTMUsecaseDto, train_lstm
from app.apis.models.search_multi_brut_model_data import SearchBrutMultiModelData
from app.usecases.gru.usecase_train_gru import TrainGRUUsecaseDto, train_model_gru
from app.usecases.usecase_create_data_puppeto4 import usecase_create_data_puppeto4
from app.usecases.lstm.usecase_create_lstm import CreateLSTMUsecaseDto, create_lstm
from app.usecases.lstm.usecase_mesure_lstm import MesureLSTMUsecaseDto, mesure_lstm
from app.usecases.lstm.usecase_search_lstm import SearchLSTMUsecaseDto, search_lstm
from app.usecases.gru.usecase_search_gru import SearchGRUUsecaseDto, search_model_gru
from app.usecases.gru.usecase_create_gru import CreateGRUUsecaseDto, create_model_gru
from app.usecases.simple.usecase_mesure_simple import MesureSimpleUsecaseDto, mesure_simple_nn
from app.usecases.siamese.usecase_mesure_siamese import MesureSiameseUsecaseDto, mesure_siamese
from app.usecases.simple.usecase_train_simple import TrainSimpleUsecaseDto, train_model_simple_nn
from app.usecases.siamese.usecase_train_siamese import TrainSiameseUsecaseDto, train_model_siamese
from app.usecases.simple.usecase_search_simple import SearchSimpleUsecaseDto, search_model_simple_nn
from app.usecases.simple.usecase_create_simple import CreateSimpleUsecaseDto, create_model_simple_nn
from app.usecases.siamese.usecase_search_siamese import SearchSiameseUsecaseDto, search_model_siamese
from app.usecases.siamese.usecase_update_siamese import UpdateSiameseUsecaseDto, update_model_siamese
from app.usecases.siamese.usecase_create_siamese import CreateSiameseUsecaseDto, create_model_siamese
from app.usecases.siamese.prepare_cache_siamese import PrepareSiameseUsecaseDto, prepare_cache_siamese
from app.usecases.gru.usecase_search_multi_brut_gru import SearchMultiBrutGRUUsecaseDto, search_multi_brut_model_gru

# Initialisation du routeur
router = APIRouter()

# Instantiate OAuth2
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Token verification function (verify_access_token)
def verify_access_token(token: str = Depends(oauth2_scheme)):
    # Load the .env file
    envs = load_env_vars()
    try:
        # Decode the token
        payload = jwt.decode(token, envs["secret_key"], algorithms=["HS256"])
        logger.debug(f"payload: {payload}")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

    # Return the decoded payload
    return payload

# API to create a model
@router.post("/create_model")
async def create_model_api(data: CreateModelData, payload: dict = Depends(verify_access_token)):
    """
    Creates a new model with specified tokens and glossary.
    """
    try:
        if data.neural_network_type == 'SimpleNN':
            return create_model_simple_nn(CreateSimpleUsecaseDto(name=data.name, inversify=get_inversify()))
        elif data.neural_network_type == 'GRU':
            return create_model_gru(CreateGRUUsecaseDto(name=data.name, inversify=get_inversify()))
        elif data.neural_network_type == 'SIAMESE':
            return create_model_siamese(CreateSiameseUsecaseDto(name=data.name, dictionary=data.dictionary, glossary=data.glossary, inversify=get_inversify()))
        elif data.neural_network_type == 'LSTM':
            return create_lstm(CreateLSTMUsecaseDto(name=data.name, inversify=get_inversify()))
        else:
            raise HTTPException(status_code=500, detail=f"Unknown neural network type: {data.neural_network_type}")
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error occurred during model creation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred during model creation: {str(e)}")

# API to update a model
@router.patch("/update_model")
async def update_model_api(data: UpdateModelData, payload: dict = Depends(verify_access_token)):
    """
    Update existing model.
    """
    try:
        if data.neural_network_type == 'SIAMESE':
            if not data.dictionary or not data.glossary:
                raise HTTPException(status_code=400, detail="Both dictionary and glossary must be provided for SIAMESE model")
            return update_model_siamese(UpdateSiameseUsecaseDto(name=data.name, dictionary=data.dictionary, glossary=data.glossary, inversify=get_inversify()))
        else:
            raise HTTPException(status_code=400, detail=f"Model type '{data.neural_network_type}' not supported for update")
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error occurred during model update: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred during model update: {str(e)}")

# API to train a model
@router.post("/train_model")
async def train_model_api(data: TrainModelData, payload: dict = Depends(verify_access_token)):
    """
    Trains an existing model using input-target tuples.
    """
    try:
        if data.neural_network_type == 'SimpleNN':
            return train_model_simple_nn(TrainSimpleUsecaseDto(name=data.name, training_data=data.training_data, inversify=get_inversify()))
        elif data.neural_network_type == 'GRU':
            return train_model_gru(TrainGRUUsecaseDto(name=data.name, training_data=data.training_data, inversify=get_inversify()))
        elif data.neural_network_type == 'SIAMESE':
            return train_model_siamese(TrainSiameseUsecaseDto(name=data.name, training_data=data.training_data, inversify=get_inversify()))
        elif data.neural_network_type == 'LSTM':
            return train_lstm(TrainLSTMUsecaseDto(name=data.name, training_data=data.training_data, inversify=get_inversify()))
        else:
            raise HTTPException(status_code=500, detail=f"Unknown neural network type: {data.neural_network_type}")
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error occurred during training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred during training: {str(e)}")

# API to prepare seaching engine
@router.post("/prepare_cache")
async def prepare_cache_api(data: PrepareCacheData, payload: dict = Depends(verify_access_token)):
    """
    Prepares the search cache for a model by precomputing results for a collection of search vectors.
    """
    try:
        if data.neural_network_type == 'SIAMESE':
            return prepare_cache_siamese(PrepareSiameseUsecaseDto(name=data.name, search_vectors=data.search_vectors, inversify=get_inversify()))
        else:
            raise HTTPException(status_code=400, detail=f"Model type '{data.neural_network_type}' is not supported for cache preparation")

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error occurred during cache preparation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred during cache preparation: {str(e)}")

# API to search a vector within a model
@router.post("/search")
async def search_model_api(data: SearchModelData, payload: dict = Depends(verify_access_token)):
    """
    Searches for a vector within the specified model.
    """
    try:
        if data.neural_network_type == 'SimpleNN':
            return search_model_simple_nn(SearchSimpleUsecaseDto(name=data.name, search=data.vector, inversify=get_inversify()))
        elif data.neural_network_type == 'GRU':
            return search_model_gru(SearchGRUUsecaseDto(name=data.name, search=data.vector, inversify=get_inversify()))
        elif data.neural_network_type == 'SIAMESE':
            return search_model_siamese(SearchSiameseUsecaseDto(name=data.name, search=data.vector, inversify=get_inversify()))
        elif data.neural_network_type == 'LSTM':
            return search_lstm(SearchLSTMUsecaseDto(name=data.name, search=data.vector, inversify=get_inversify()))
        else:
            raise HTTPException(status_code=500, detail=f"Unknown neural network type: {data.neural_network_type}")
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error occurred during search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred during search: {str(e)}")
    
# API to search a vector within a model
@router.post("/search_brut_multi")
async def search_brut_multi_model_api(data: SearchBrutMultiModelData, payload: dict = Depends(verify_access_token)):
    """
    Searches for multi input within the specified model.
    """
    try:
        if data.neural_network_type == 'GRU':
            logger.info(f"search_brut_multi: {data}")
            return search_multi_brut_model_gru(SearchMultiBrutGRUUsecaseDto(name=data.name, documents=data.documents, inversify=get_inversify()))
        else:
            raise HTTPException(status_code=500, detail=f"Unknown neural network type: {data.neural_network_type}")
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error occurred during search multi brut: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred during search multi brut: {str(e)}")

# API to test a model
@router.post("/test")
async def test(data: TestModelData, payload: dict = Depends(verify_access_token)):
    """
    Tests the specified model with provided test data.
    """
    try:
        if data.neural_network_type == 'SimpleNN':
            return mesure_simple_nn(MesureSimpleUsecaseDto(name=data.name, test_data=data.test_data, inversify=get_inversify()))
        elif data.neural_network_type == 'GRU':
            return mesure_gru(MesureGRUUsecaseDto(
                    name=data.name,
                    test_data=data.test_data,
                    inversify=get_inversify(),
                    iterate=data.iterate
                ))
        elif data.neural_network_type == 'SIAMESE':
            return mesure_siamese(MesureSiameseUsecaseDto(name=data.name, test_data=data.test_data, inversify=get_inversify()))
        elif data.neural_network_type == 'LSTM':
            return mesure_lstm(MesureLSTMUsecaseDto(name=data.name, test_data=data.test_data, inversify=get_inversify()))
        else:
            raise HTTPException(status_code=400, detail="Model type not supported yet")
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error occurred during testing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred during testing: {str(e)}")

# API to retrieve all models
@router.get("/models")
async def get_all_models_api(payload: dict = Depends(verify_access_token)):
    """
    Returns all models stored in memory.
    """
    try:
        return get_all_models_usecase(get_inversify())
    except Exception as e:
        logger.error(f"Error occurred while retrieving models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while retrieving models: {str(e)}")

# API for data tokenization
@router.post("/tokenize")
async def tokenize(data: TokenizeModelData, payload: dict = Depends(verify_access_token)):
    """
    Tokenizes provided data.
    """
    try:
        return usecase_tokenize(data.data)
    except Exception as e:
        logger.error(f"Error occurred during tokenization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred during tokenization: {str(e)}")

# API to create data for Puppet-O4
@router.get("/create_data_puppet-o4")
async def create_data_puppet_o4(payload: dict = Depends(verify_access_token)):
    """
    Creates data for Puppet-O4 use case.
    """
    return usecase_create_data_puppeto4()

# API to get application version
@router.get("/version")
async def get_version():
    """
    Returns the current application version.
    """
    return {"version": __version__}

# Example implementation of a secure endpoint
@router.post("/secure_endpoint")
async def secure_endpoint(payload: dict = Depends(verify_access_token)):
    """
    Example of a secure endpoint using JWT
    """
    try:
        return {"message": "Secured endpoint accessed"}
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
