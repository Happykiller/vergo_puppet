#app\apis\apis.py
from fastapi import APIRouter, HTTPException  # type: ignore

from app.version import __version__
from app.services.logger import logger
from app.apis.models.train_model_data import TrainModelData
from app.apis.models.test_model_data import TestModelData
from app.usecases.usecase_tokenize import usecase_tokenize
from app.usecases.gru.usecase_mesure_gru import mesure_gru
from app.usecases.lstm.usecase_train_lstm import train_lstm
from app.usecases.getall_model import get_all_models_usecase
from app.apis.models.create_model_data import CreateModelData
from app.apis.models.search_model_data import SearchModelData
from app.usecases.lstm.usecase_create_lstm import create_lstm
from app.usecases.lstm.usecase_mesure_lstm import mesure_lstm
from app.usecases.lstm.usecase_search_lstm import search_lstm
from app.usecases.gru.usecase_train_gru import train_model_gru
from app.usecases.gru.usecase_create_gru import create_model_gru
from app.usecases.gru.usecase_search_gru import search_model_gru
from app.apis.models.tokenize_model_data import TokenizeModelData
from app.usecases.simple_nn.mesure_simple_nn import mesure_simple_nn
from app.usecases.siamese.usecase_mesure_siamese import mesure_siamese
from app.usecases.siamese.usecase_train_siamese import train_model_siamese
from app.usecases.siamese.usecase_create_siamese import create_model_siamese
from app.usecases.siamese.usecase_search_siamese import search_model_siamese
from app.usecases.simple_nn.train_model_simple_nn import train_model_simple_nn
from app.usecases.simple_nn.create_model_simple_nn import create_model_simpleNN
from app.usecases.simple_nn.search_model_simple_nn import search_model_simple_nn
from app.usecases.usecase_create_data_puppeto4 import usecase_create_data_puppeto4

# Initialisation du routeur
router = APIRouter()

# API to create a model
@router.post("/create_model")
async def create_model_api(data: CreateModelData):
    """
    Creates a new model with specified tokens and glossary.
    """
    try:
        if data.neural_network_type == 'SimpleNN':
            return create_model_simpleNN(data.name)
        elif data.neural_network_type == 'GRU':
            return create_model_gru(data.name)
        elif data.neural_network_type == 'SIAMESE':
            return create_model_siamese(data.name, data.dictionary, data.glossary, data.neural_network_type)
        elif data.neural_network_type == 'LSTM':
            return create_lstm(data.name)
        else:
            raise HTTPException(status_code=500, detail=f"Unknown neural network type: {data.neural_network_type}")
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error occurred during model creation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred during model creation: {str(e)}")

# API to train a model
@router.post("/train_model")
async def train_model_api(data: TrainModelData):
    """
    Trains an existing model using input-target tuples.
    """
    try:
        if data.neural_network_type == 'SimpleNN':
            return train_model_simple_nn(data.name, data.training_data)
        elif data.neural_network_type == 'GRU':
            return train_model_gru(data.name, data.training_data)
        elif data.neural_network_type == 'SIAMESE':
            return train_model_siamese(data.name, data.training_data)
        elif data.neural_network_type == 'LSTM':
            return train_lstm(data.name, data.training_data)
        else:
            raise HTTPException(status_code=500, detail=f"Unknown neural network type: {data.neural_network_type}")
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error occurred during training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred during training: {str(e)}")

# API to search a vector within a model
@router.post("/search")
async def search_model_api(data: SearchModelData):
    """
    Searches for a vector within the specified model.
    """
    try:
        if data.neural_network_type == 'SimpleNN':
            return search_model_simple_nn(data.name, data.vector)
        elif data.neural_network_type == 'GRU':
            return search_model_gru(data.name, data.vector)
        elif data.neural_network_type == 'SIAMESE':
            return search_model_siamese(data.name, data.vector)
        elif data.neural_network_type == 'LSTM':
            return search_lstm(data.name, data.vector)
        else:
            raise HTTPException(status_code=500, detail=f"Unknown neural network type: {data.neural_network_type}")
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error occurred during search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred during search: {str(e)}")

# API to test a model
@router.post("/test")
async def test(data: TestModelData):
    """
    Tests the specified model with provided test data.
    """
    try:
        if data.neural_network_type == 'SIAMESE':
            return mesure_siamese(data.name, data.test_data)
        elif data.neural_network_type == 'SimpleNN':
            return mesure_simple_nn(data.name, data.test_data)
        elif data.neural_network_type == 'GRU':
            return mesure_gru(data.name, data.test_data)
        elif data.neural_network_type == 'LSTM':
            return mesure_lstm(data.name, data.test_data)
        else:
            raise HTTPException(status_code=400, detail="Model type not supported yet")
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error occurred during testing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred during testing: {str(e)}")

# API to retrieve all models
@router.get("/models")
async def get_all_models_api():
    """
    Returns all models stored in memory.
    """
    try:
        return get_all_models_usecase()
    except Exception as e:
        logger.error(f"Error occurred while retrieving models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while retrieving models: {str(e)}")

# API for data tokenization
@router.post("/tokenize")
async def tokenize(data: TokenizeModelData):
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
async def create_data_puppet_o4():
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