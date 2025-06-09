# app\apis\apis.py
import jwt # type: ignore
import json
from typing import Any
from fastapi.security import OAuth2PasswordBearer # type: ignore
from fastapi import BackgroundTasks, Depends, APIRouter, HTTPException # type: ignore

from app.version import __version__
from app.apis.common import FILES_DIR
from app.services.logger import logger
from app.inversify import get_inversify
from app.apis.thing_api import thing_router
from app.apis.deps import verify_access_token
from app.apis.embedding_api import embedding_router
from app.usecases.get_model import get_model_usecase
from app.common import load_env_vars, parse_input_data
from app.apis.models.test_model_data import TestModelData
from app.services.bdd.models.model_data import ModelStatus
from app.usecases.usecase_tokenize import usecase_tokenize
from app.apis.models.train_model_data import TrainModelData
from app.usecases.getall_model import get_all_models_usecase
from app.apis.models.create_model_data import CreateModelData
from app.apis.models.update_model_data import UpdateModelData
from app.apis.models.search_model_data import SearchModelData
from app.apis.models.prepare_cache_data import PrepareCacheData
from app.apis.models.tokenize_model_data import TokenizeModelData
from app.apis.models.super_train_model_data import SuperTrainModelData
from app.apis.models.gru_training_model_data import GRUTrainingModelData
from app.apis.models.train_from_file_model_data import TrainFromFileModelData
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
from app.usecases.gru.usecase_super_train_gru import SuperTrainGRUUsecaseDto, super_train_model_gru
from app.usecases.simple.usecase_search_simple import SearchSimpleUsecaseDto, search_model_simple_nn
from app.usecases.simple.usecase_create_simple import CreateSimpleUsecaseDto, create_model_simple_nn
from app.usecases.siamese.usecase_search_siamese import SearchSiameseUsecaseDto, search_model_siamese
from app.usecases.siamese.usecase_update_siamese import UpdateSiameseUsecaseDto, update_model_siamese
from app.usecases.siamese.usecase_create_siamese import CreateSiameseUsecaseDto, create_model_siamese
from app.usecases.siamese.prepare_cache_siamese import PrepareSiameseUsecaseDto, prepare_cache_siamese
from app.usecases.siamese.usecase_super_train_siamese import SuperTrainSiameseUsecaseDto, super_train_model_siamese
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
            return create_model_siamese(CreateSiameseUsecaseDto(name=data.name, dictionary=data.dictionary, inversify=get_inversify()))
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
            return update_model_siamese(UpdateSiameseUsecaseDto(name=data.name, dictionary=data.dictionary, inversify=get_inversify()))
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
        train_data = parse_input_data(data.training_data, data.train_file)

        if data.neural_network_type == 'SimpleNN':
            return train_model_simple_nn(TrainSimpleUsecaseDto(name=data.name, training_data=train_data, inversify=get_inversify()))
        elif data.neural_network_type == 'GRU':
            return train_model_gru(TrainGRUUsecaseDto(name=data.name, training_data=train_data, inversify=get_inversify()))
        elif data.neural_network_type == 'SIAMESE':
            return train_model_siamese(TrainSiameseUsecaseDto(name=data.name, training_data=train_data, inversify=get_inversify()))
        elif data.neural_network_type == 'LSTM':
            return train_lstm(TrainLSTMUsecaseDto(name=data.name, training_data=train_data, inversify=get_inversify()))
        else:
            raise HTTPException(status_code=500, detail=f"Unknown neural network type: {data.neural_network_type}")
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error occurred during training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred during training: {str(e)}")
    
@router.post("/train_model_from_file")
async def train_model_from_file(
    background_tasks: BackgroundTasks,
    data: TrainFromFileModelData,
    payload: dict = Depends(verify_access_token)
):
    """
    Asynchronously trains a model using training data from an uploaded file.
    The file is stored in a directory and processed in the background.
    """
    try:
        model = get_model_usecase(data.name, inversify=get_inversify())

        if not model:
            raise Exception("Model not found")

        if(model.status == ModelStatus.TRAINING):
            raise Exception("Model is training")

        # Launch training asynchronously
        background_tasks.add_task(train_model_from_file_background, data.name, data.neural_network_type, data.file_name)

        return {"message": "Training initiated", "file": data.file_name}
    except Exception as e:
        logger.error(f"Error occurred while uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

def train_model_from_file_background(name: str, neural_network_type: str, file_name: str):
    """
    Background task to process and train the model using the uploaded file.
    """
    try:
        file_path = FILES_DIR / file_name

        # Check if the file exists
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File '{file_path}' not found.")

        # Load training data from the file
        with file_path.open("r", encoding="utf-8") as f:
            training_data = json.load(f)  # Assuming JSON format for training data

        # Call appropriate training function
        if neural_network_type == 'SimpleNN':
            train_model_simple_nn(TrainSimpleUsecaseDto(name=name, training_data=training_data, inversify=get_inversify()))
        elif neural_network_type == 'GRU':
            training_data = [GRUTrainingModelData(**data) for data in training_data]
            train_model_gru(TrainGRUUsecaseDto(name=name, training_data=training_data, inversify=get_inversify()))
        elif neural_network_type == 'SIAMESE':
            train_model_siamese(TrainSiameseUsecaseDto(name=name, training_data=training_data, inversify=get_inversify()))
        elif neural_network_type == 'LSTM':
            train_lstm(TrainLSTMUsecaseDto(name=name, training_data=training_data, inversify=get_inversify()))
        else:
            raise ValueError(f"Unknown neural network type: {neural_network_type}")

        logger.info(f"Training completed for {name} using {file_path}")
    except Exception as e:
        logger.error(f"Error during training from file {file_path}: {str(e)}")

@router.post("/super_train_model")
async def super_train_model(
    background_tasks: BackgroundTasks,
    data: SuperTrainModelData,
    payload: dict = Depends(verify_access_token)
):
    """
    Asynchronously trains a model using two files: one for training data and one for test data.
    """
    try:
        model = get_model_usecase(data.name, inversify=get_inversify())
        if not model:
            raise Exception("Model not found")

        if model.status == ModelStatus.SUPER_TRAINING:
            raise Exception("Model is already in SUPER_TRAINING state")
        
        train_data = parse_input_data(data.train_data, data.train_file)

        test_data = parse_input_data(data.test_data, data.test_file)

        # 3) Launch the background task with the final training & test data
        background_tasks.add_task(
            super_train_model_background,
            data,
            train_data,
            test_data,
            data.iterate
        )

        return {
            "message": "Super training initiated",
            "model": data.name,
        }

    except Exception as e:
        logger.error(f"Error occurred while initiating super training: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )

def super_train_model_background(
    data: SuperTrainModelData,
    train_data: Any,
    test_data: Any,
    iterate: int
):
    """
    Background task to process and super train the model using two separate files:
    one for training data and one for test data.
    """
    try:
        # For GRU, we must parse each item into GRUTrainingModelData
        if data.neural_network_type == 'GRU':
            train_data = [GRUTrainingModelData(**item) for item in train_data]

        # Depending on the neural network type, call the correct usecase
        if data.neural_network_type == 'SIAMESE':
            super_train_model_siamese(
                SuperTrainSiameseUsecaseDto(
                    name=data.name,
                    training_data=train_data,
                    test_data=test_data,
                    n_iterations=iterate,
                    inversify=get_inversify()
                )
            )
        elif data.neural_network_type == 'GRU':
            # For GRU, we must parse training data items into GRUTrainingModelData
            train_data = [GRUTrainingModelData(**item) for item in train_data]
            super_train_model_gru(
                SuperTrainGRUUsecaseDto(
                    name=data.name,
                    training_data=train_data,
                    test_data=test_data,
                    n_iterations=iterate,
                    inversify=get_inversify()
                )
            )
        else:
            raise ValueError(
                f"Unsupported neural network type for super training: {data.neural_network_type}"
            )

        logger.info(f"Super training completed for {data.name}")
    except Exception as e:
        logger.error(f"Error during super training from files: {str(e)}")

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
        test_data = parse_input_data(data.test_data, data.test_file)

        if data.neural_network_type == 'SimpleNN':
            return mesure_simple_nn(MesureSimpleUsecaseDto(name=data.name, test_data=test_data, inversify=get_inversify()))
        elif data.neural_network_type == 'GRU':
            return mesure_gru(MesureGRUUsecaseDto(
                    name=data.name,
                    test_data=test_data,
                    inversify=get_inversify(),
                    iterate=data.iterate
                ))
        elif data.neural_network_type == 'SIAMESE':
            return mesure_siamese(MesureSiameseUsecaseDto(name=data.name, test_data=test_data, inversify=get_inversify()))
        elif data.neural_network_type == 'LSTM':
            return mesure_lstm(MesureLSTMUsecaseDto(name=data.name, test_data=test_data, inversify=get_inversify()))
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
    Returns a list of all models stored in memory with their name and neural network type.
    """
    try:
        models = get_all_models_usecase(get_inversify())
        return models
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

router.include_router(thing_router)
router.include_router(embedding_router)
