# app\usecases\siamese\usecase_create_siamese.py
import traceback
from typing import List, NamedTuple

from app.inversify import Inversify
from app.services.logger import logger
from app.services.bdd.models.model_data import ModelData
from app.usecases.siamese.usecase_commons_siamese import tokens_to_indices

class CreateSiameseUsecaseDto(NamedTuple):
    name: str
    dictionary: List[List[str]]
    glossary: List[str]
    inversify: Inversify

def create_model_siamese(dto: CreateSiameseUsecaseDto):
    try:
        logger.info(f"Machine learning type used for model creation: SIAMESE")

        # Fetch Bdd
        bdd = dto.inversify.get_bdd()

        if bdd.model_exists(dto.name):
            raise Exception("Model already exists")
        
        # Check if dictionary or glossary are None
        if dto.dictionary is None:
            raise Exception("Dictionary cannot be None")
        if dto.glossary is None:
            raise Exception("Glossary cannot be None")
        
        # Check if dictionary or glossary are empty
        if len(dto.dictionary) == 0:
            raise Exception("Dictionary cannot be empty")
        if len(dto.glossary) == 0:
            raise Exception("Glossary cannot be empty")

        # Add a blank entry at the beginning of the glossary
        glossary = [""] + ["UNK"] + dto.glossary

        # Track tokens not in glossary
        tokens_not_in_glossary = set()

        # Transform each list of tokens into a list of indices
        indexed_dictionary = []
        for tokens in dto.dictionary:
            indices = tokens_to_indices(tokens, glossary)
            indexed_dictionary.append(indices)

            # Identify tokens not mapped to known indices
            unknown_tokens = [token for token, index in zip(tokens, indices) if index == glossary.index("UNK")]
            tokens_not_in_glossary.update(unknown_tokens)

        # Save the model with the glossary and indexed dictionary
        bdd.save_model(ModelData(
            name=dto.name, 
            neural_network_type="SIAMESE",
            dictionary=dto.dictionary,
            indexed_dictionary=indexed_dictionary,
            glossary=glossary
        ))

        # Return response with missing tokens
        return {
            "status": "model created",
            "model_name": dto.name,
            "missing_tokens": list(tokens_not_in_glossary)  # List of tokens not in the glossary
        }
    
    except Exception as e:
        logger.error(f"Error message:{str(e)}\nStack trace:\n{traceback.format_exc()}")
        raise Exception(f"[#create_model_siamese]{str(e)}")
