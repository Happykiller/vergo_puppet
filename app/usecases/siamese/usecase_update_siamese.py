# app/usecases/siamese/usecase_update_siamese.py
from typing import List, NamedTuple
from app.inversify import Inversify
from fastapi import HTTPException  # type: ignore

from app.services.logger import logger
from app.usecases.siamese.usecase_commons_siamese import tokens_to_indices

class UpdateSiameseUsecaseDto(NamedTuple):
    name: str
    dictionary: List[List[str]]
    glossary: List[str]
    inversify: Inversify

def update_model_siamese(dto: UpdateSiameseUsecaseDto):
    """
    Updates the SIAMESE model with a new dictionary and glossary.
    :param name: Name of the model
    :param dictionary: New dictionary for the model
    :param glossary: New glossary for the model
    :return: Status of the update
    """
    # Fetch Bdd
    bdd = dto.inversify.get_bdd()

    # Retrieve the model from memory
    model = bdd.get_model(dto.name)

    if model is None or not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if not dto.dictionary or len(dto.dictionary) == 0:
        raise HTTPException(status_code=400, detail="Dictionary cannot be empty")
    
    if not dto.glossary or len(dto.glossary) == 0:
        raise HTTPException(status_code=400, detail="Glossary cannot be empty")
    
    logger.info(f"Updating SIAMESE model '{dto.name}' with new dictionary and glossary")
    
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
    
    # Update the model
    model["dictionary"] = dto.dictionary
    model["indexed_dictionary"] = indexed_dictionary
    model["glossary"] = glossary
    
    # Save the updated model back to memory
    bdd.update_model(dto.name, model)
    
    logger.info(f"SIAMESE model '{dto.name}' successfully updated")

    # Clear the search buffer for the model
    bdd.clear_search_buffer(dto.name)
    
    # Return response with missing tokens
    return {
        "status": "model updated",
        "model_name": dto.name,
        "missing_tokens": list(tokens_not_in_glossary)  # List of tokens not in the glossary
    }
