# app/usecases/siamese/usecase_update_siamese.py
from typing import List
from fastapi import HTTPException  # type: ignore

from app.services.logger import logger
from app.repositories.memory import clear_search_buffer, get_model, update_model

def update_model_siamese(name: str, dictionary: List[List[str]], glossary: List[str]):
    """
    Updates the SIAMESE model with a new dictionary and glossary.
    :param name: Name of the model
    :param dictionary: New dictionary for the model
    :param glossary: New glossary for the model
    :return: Status of the update
    """
    # Retrieve the model from memory
    model = get_model(name)

    if model is None or not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if not dictionary or len(dictionary) == 0:
        raise HTTPException(status_code=400, detail="Dictionary cannot be empty")
    
    if not glossary or len(glossary) == 0:
        raise HTTPException(status_code=400, detail="Glossary cannot be empty")
    
    logger.info(f"Updating SIAMESE model '{name}' with new dictionary and glossary")
    
    # Validate consistency of the glossary and dictionary
    flat_dictionary = {token for tokens in dictionary for token in tokens}
    if not flat_dictionary.issubset(set(glossary)):
        raise HTTPException(
            status_code=400,
            detail="All tokens in the dictionary must be present in the glossary"
        )
    
    # Update the model
    model["dictionary"] = dictionary
    model["glossary"] = glossary
    
    # Save the updated model back to memory
    update_model(name, model)
    
    logger.info(f"SIAMESE model '{name}' successfully updated")

    # Clear the search buffer for the model
    clear_search_buffer(name)
    
    return {"status": "model updated", "model_name": name}
