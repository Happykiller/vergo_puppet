#app\usecases\siamese\usecase_train_siamese.py
from typing import List, Tuple
from fastapi import HTTPException  # type: ignore

from app.services.logger import logger
from app.neural_network.nn_siamese import train_siamese_model_nn
from app.repositories.memory import clear_search_buffer, get_model, update_model
from app.usecases.siamese.usecase_commons_siamese import create_glossary_from_training_data, tokens_to_indices

def train_model_siamese(name: str, training_data: List[Tuple[List[str], List[str], float]]):
    """
    Trains the model with pairs (input, target).
    :param name: Name of the model
    :param training_data: List of tuples (input, target) where input and target are lists of tokens
    """
    model = get_model(name)

    if model is None or not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if training_data is None:
        raise HTTPException(status_code=400, detail="No training data provided")
    
    if len(training_data) == 0:
        raise HTTPException(status_code=400, detail="Training data is empty")
    
    indexed_dictionary = model.get("indexed_dictionary", [])
    if not indexed_dictionary:
        raise HTTPException(status_code=400, detail="No vectors available in the model")
    
    # Check the neural network type to use
    logger.info(f"Machine learning type used for training: SIAMESE")

    # Train the neural network according to the specified model type
    training_glossary = create_glossary_from_training_data(training_data)
    training_word2idx = {word: idx for idx, word in enumerate(training_glossary)}
    
    transformed_data = []
    for source_tokens, target_tokens, score in training_data:
        source_indices = tokens_to_indices(source_tokens, training_word2idx)
        target_indices = tokens_to_indices(target_tokens, training_word2idx)
        transformed_data.append((source_indices, target_indices, score))

    vocab_size = len(model["glossary"]) + 1
    
    # Train the model
    nn_model, report = train_siamese_model_nn(transformed_data, vocab_size)

    # Save the trained neural network model
    update_model(name, {"nn_model": nn_model})

    # Clear the search buffer for the model
    clear_search_buffer(name)

    return {
        "status": "training completed",
        "model_name": name,
        "training_report": report
    }
