# app\usecases\siamese\usecase_create_siamese.py
import traceback
from typing import List, NamedTuple

from app.inversify import Inversify
from app.services.logger import logger
from app.services.bdd.models.model_data import ModelData, ModelStatus
from app.usecases.siamese.usecase_commons_siamese import calculate_word_representation, create_glossary_from_dictionary, tokens_to_indices

class CreateSiameseUsecaseDto(NamedTuple):
    name: str
    dictionary: List[List[str]]
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
        
        # Check if dictionary or glossary are empty
        if len(dto.dictionary) == 0:
            raise Exception("Dictionary cannot be empty")

        # Création automatique du glossaire depuis le dictionnaire
        glossary = create_glossary_from_dictionary(dto.dictionary)

        # Transform each list of tokens into a list of indices
        indexed_dictionary = []
        for tokens in dto.dictionary:
            indices = tokens_to_indices(tokens, glossary)
            indexed_dictionary.append(indices)

        # Save the model with the glossary and indexed dictionary
        bdd.save_model(ModelData(
            name=dto.name, 
            neural_network_type="SIAMESE",
            dictionary=dto.dictionary,
            indexed_dictionary=indexed_dictionary,
            glossary=glossary,
            status=ModelStatus.CREATED
        ))

        # Compute word representation rate
        word_representation = calculate_word_representation(dto.dictionary)

        # Return response with missing tokens
        return {
            "status": "model created",
            "model_name": dto.name,
            "word_representation": word_representation
        }
    
    except Exception as e:
        logger.error(f"Error message:{str(e)}\nStack trace:\n{traceback.format_exc()}")
        raise Exception(f"[#create_model_siamese]{str(e)}")
