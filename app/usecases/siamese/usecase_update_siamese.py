# app/usecases/siamese/usecase_update_siamese.py
import traceback
from typing import List, NamedTuple

from app.inversify import Inversify
from app.services.logger import logger
from app.neural_network.nn_siamese import SiameseLSTM
from app.services.bdd.models.model_data import ModelData
from app.usecases.siamese.usecase_commons_siamese import calculate_word_representation, create_glossary_from_dictionary, tokens_to_indices

class UpdateSiameseUsecaseDto(NamedTuple):
    name: str
    dictionary: List[List[str]]
    inversify: Inversify

def update_model_siamese(dto: UpdateSiameseUsecaseDto):
    """
    Updates the SIAMESE model with a new dictionary and glossary.
    :param name: Name of the model
    :param dictionary: New dictionary for the model
    :return: Status of the update
    """
    try:
        # Fetch Bdd
        bdd = dto.inversify.get_bdd()

        # Retrieve the model from memory
        model = bdd.get_model(dto.name, SiameseLSTM)

        if model is None or not model:
            raise Exception("Model not found")
        
        if not dto.dictionary or len(dto.dictionary) == 0:
            raise Exception("Dictionary cannot be empty")
        
        logger.info(f"Updating SIAMESE model '{dto.name}' with new dictionary and glossary")
        
        # Création automatique du glossaire depuis le dictionnaire
        glossary = create_glossary_from_dictionary(dto.dictionary)

        # Transform each list of tokens into a list of indices
        indexed_dictionary = []
        for tokens in dto.dictionary:
            indices = tokens_to_indices(tokens, glossary)
            indexed_dictionary.append(indices)

        # Save the updated model
        bdd.update_model(ModelData(
            name=dto.name, 
            neural_network_type="SIAMESE",
            dictionary=dto.dictionary,
            indexed_dictionary=indexed_dictionary,
            glossary=glossary
        ))
        
        logger.info(f"SIAMESE model '{dto.name}' successfully updated")

        # Clear the search buffer for the model
        bdd.clear_search_buffer(dto.name)

        # Compute word representation rate
        word_representation = calculate_word_representation(dto.dictionary)
        
        # Return response with missing tokens
        return {
            "status": "model updated",
            "model_name": dto.name,
            "word_representation": word_representation
        }
    
    except Exception as e:
        logger.error(f"Error message:{str(e)}\nStack trace:\n{traceback.format_exc()}")
        raise Exception(f"[#update_model_siamese]{str(e)}")
