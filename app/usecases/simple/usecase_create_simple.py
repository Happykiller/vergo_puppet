# app\usecases\simple\usecase_create_simple.py
from app.services.logger import logger
from app.inversify import get_inversify
from fastapi import HTTPException  # type: ignore

def create_model_simple_nn(name: str):
    logger.info("Machine learning type used for creating 'SimpleNN' model")

    inversify = get_inversify()
    bdd = inversify.get_bdd()

    if bdd.model_exists(name):
        raise HTTPException(status_code=400, detail="Model already exists")

    # Save the model with its glossary and index dictionary
    model_data = {
        "neural_network_type": "SimpleNN"  # Save the model type
    }
    bdd.save_model(name, model_data)

    return {"status": "model created", "model_name": name}
