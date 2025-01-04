# app\usecases\siamese\usecase_train_siamese.py
import traceback
from typing import List, NamedTuple, Tuple

from app.inversify import Inversify
from app.services.logger import logger
from app.services.bdd.models.model_data import ModelData
from app.neural_network.nn_siamese import SiameseLSTM, train_siamese_model_nn
from app.usecases.siamese.usecase_commons_siamese import create_glossary_from_training_data, tokens_to_indices

class TrainSiameseUsecaseDto(NamedTuple):
    name: str
    training_data: List[Tuple[List[str], List[str], float]]
    inversify: Inversify

def train_model_siamese(dto: TrainSiameseUsecaseDto):
    """
    Trains the model with pairs (input, target).
    :param name: Name of the model
    :param training_data: List of tuples (input, target) where input and target are lists of tokens
    """
    try:
        # Fetch Bdd
        bdd = dto.inversify.get_bdd()

        model = bdd.get_model(dto.name, SiameseLSTM)

        if model is None or not model:
            raise Exception("Model not found")
        
        if dto.training_data is None:
            raise Exception("No training data provided")
        
        if len(dto.training_data) == 0:
            raise Exception("Training data is empty")
        
        indexed_dictionary = model.indexed_dictionary
        if not indexed_dictionary:
            raise Exception("No vectors available in the model")
        
        # Check the neural network type to use
        logger.info(f"Machine learning type used for training: SIAMESE")

        # Train the neural network according to the specified model type
        training_glossary = create_glossary_from_training_data(dto.training_data)
        training_word2idx = {word: idx for idx, word in enumerate(training_glossary)}
        
        transformed_data = []
        for source_tokens, target_tokens, score in dto.training_data:
            source_indices = tokens_to_indices(source_tokens, training_word2idx)
            target_indices = tokens_to_indices(target_tokens, training_word2idx)
            transformed_data.append((source_indices, target_indices, score))

        vocab_size = len(model.glossary) + 1
        
        # Train the model
        nn_model, report = train_siamese_model_nn(transformed_data, vocab_size)

        # Save the trained neural network model
        bdd.update_model(ModelData(
            name=model.name, 
            neural_network_type=model.neural_network_type,
            nn_model=nn_model,
            dictionary=model.dictionary,
            indexed_dictionary=model.indexed_dictionary,
            glossary=model.glossary
        ))

        # Clear the search buffer for the model
        bdd.clear_search_buffer(dto.name)

        return {
            "status": "training completed",
            "model_name": dto.name,
            "training_report": report
        }
    
    except Exception as e:
        logger.error(f"Error message:{str(e)}\nStack trace:\n{traceback.format_exc()}")
        raise Exception(f"[#train_model_siamese]{str(e)}")