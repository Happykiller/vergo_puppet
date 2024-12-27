# app\usecases\simple\usecase_create_simple.py
from typing import NamedTuple, Optional
from fastapi import HTTPException  # type: ignore

from app.inversify import Inversify
from app.services.logger import logger

class CreateSimpleUsecaseDto(NamedTuple):
    name: str
    inversify: Inversify
    param_optionnel: Optional[str] = None

def create_model_simple_nn(dto: CreateSimpleUsecaseDto):
    logger.info("Machine learning type used for creating 'SimpleNN' model")

    # Fetch Bdd
    bdd = dto.inversify.get_bdd()

    if bdd.model_exists(dto.name):
        raise HTTPException(status_code=400, detail="Model already exists")

    # Save the model with its glossary and index dictionary
    model_data = {
        "neural_network_type": "SimpleNN"  # Save the model type
    }
    bdd.save_model(dto.name, model_data)

    return {"status": "model created", "model_name": dto.name}
