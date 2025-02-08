# app\usecases\simple\usecase_create_simple.py
from typing import NamedTuple, Optional

from app.inversify import Inversify
from app.services.logger import logger
from app.services.bdd.models.model_data import ModelData, ModelStatus

class CreateSimpleUsecaseDto(NamedTuple):
    name: str
    inversify: Inversify
    param_optionnel: Optional[str] = None

def create_model_simple_nn(dto: CreateSimpleUsecaseDto):
    logger.info("Machine learning type used for creating 'SimpleNN' model")

    # Fetch Bdd
    bdd = dto.inversify.get_bdd()

    if bdd.model_exists(dto.name):
        raise Exception("Model already exists")

    # Save the model with its glossary and index dictionary
    bdd.save_model(ModelData(name=dto.name, neural_network_type="SimpleNN", status=ModelStatus.CREATED))

    return {"status": "model created", "model_name": dto.name}
