#app\usecases\siamese\usecase_create_siamese.py
from typing import List
from fastapi import HTTPException  # type: ignore
from app.services.logger import logger
from app.repositories.memory import model_exists, save_model
from app.usecases.siamese.usecase_commons_siamese import tokens_to_indices

def create_model_siamese(name: str, dictionary: List[List[str]], glossary: List[str]):
    logger.info(f"Machine learning type used for model creation: SIAMESE")

    if model_exists(name):
        raise HTTPException(status_code=400, detail="Model already exists")
    
    # Check if dictionary or glossary are None
    if dictionary is None:
        raise HTTPException(status_code=400, detail="Dictionary cannot be None")
    if glossary is None:
        raise HTTPException(status_code=400, detail="Glossary cannot be None")
    
    # Check if dictionary or glossary are empty
    if len(dictionary) == 0:
        raise HTTPException(status_code=400, detail="Dictionary cannot be empty")
    if len(glossary) == 0:
        raise HTTPException(status_code=400, detail="Glossary cannot be empty")

    # Add a blank entry at the beginning of the glossary
    glossary = [""] + ["UNK"] + glossary

    # Transform each list of tokens into a list of indices
    indexed_dictionary = [
        tokens_to_indices(tokens, glossary) for tokens in dictionary
    ]

    # Save the model with the glossary and indexed dictionary
    model_data = {
        "dictionary": dictionary,
        "indexed_dictionary": indexed_dictionary,  # Dictionary transformed with indices
        "glossary": glossary,  # Original glossary
    }
    save_model(name, model_data)

    return {"status": "model created", "model_name": name}
