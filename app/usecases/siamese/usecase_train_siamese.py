# app\usecases\siamese\usecase_train_siamese.py
import traceback
import numpy as np
from collections import Counter
from typing import Dict, List, NamedTuple, Tuple

from app.inversify import Inversify
from app.services.logger import logger
from app.services.bdd.models.model_metrics import MetricsModel
from app.services.bdd.models.model_data import ModelData, ModelStatus
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

        # Update the model in storage
        if(model.status != ModelStatus.SUPER_TRAINING):
            model.status = ModelStatus.TRAINING
            bdd.update_model(model)

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
            status=ModelStatus.TRAINED if model.status != ModelStatus.SUPER_TRAINING else ModelStatus.SUPER_TRAINING,
            nn_model=nn_model,
            dictionary=model.dictionary,
            indexed_dictionary=model.indexed_dictionary,
            glossary=model.glossary
        ))
        
        data_training_stats = compute_training_statistics(dto.training_data)
        
        bdd.save_metrics(MetricsModel(
            model_name=dto.name,
            metrics={
                "type": "training",
                "data_training_stats": data_training_stats,
                "training_stats": report
            }
        ))

        # Clear the search buffer for the model
        bdd.clear_search_buffer(dto.name)

        return {
            "status": "training completed",
            "model_name": dto.name,
            "data_training_stats": data_training_stats,
            "training_report": report
        }
    
    except Exception as e:
        logger.error(f"Error message:{str(e)}\nStack trace:\n{traceback.format_exc()}")
        raise Exception(f"[#train_model_siamese]{str(e)}")
    
def compute_training_statistics(training_data: List[Tuple[List[str], List[str], float]]) -> Dict:
    """
    Compute statistics on training data for a Siamese model.
    
    :param training_data: List of (source_tokens, target_tokens, similarity_score)
    :return: Dictionary with computed statistics
    """
    word_counts = Counter()
    source_lengths = []
    target_lengths = []
    similarity_buckets = Counter({
        "0.0-0.0": 0,
        "0.0-0.1": 0, "0.1-0.2": 0, "0.2-0.3": 0, "0.3-0.4": 0, "0.4-0.5": 0, 
        "0.5-0.6": 0, "0.6-0.7": 0, "0.7-0.8": 0, "0.8-0.9": 0, "0.9-1.0": 0,
        "1.0-1.0": 0
    })

    for source_tokens, target_tokens, similarity in training_data:
        word_counts.update(source_tokens + target_tokens)
        source_lengths.append(len(source_tokens))
        target_lengths.append(len(target_tokens))

        if similarity == 0.0:
            similarity_buckets["0.0-0.0"] += 1
        elif similarity == 1.0:
            similarity_buckets["1.0-1.0"] += 1
        else:
            lower_bound = int(similarity * 10) / 10
            upper_bound = lower_bound + 0.1
            bucket_key = f"{lower_bound:.1f}-{upper_bound:.1f}"
            similarity_buckets[bucket_key] += 1

    total_words = sum(word_counts.values())
    word_frequencies = {str(word): count / total_words * 100 for word, count in word_counts.items()}
    total_pairs = len(training_data)

    source_length_distribution = {str(length): count / total_pairs * 100 for length, count in Counter(source_lengths).items()}
    target_length_distribution = {str(length): count / total_pairs * 100 for length, count in Counter(target_lengths).items()}
    similarity_distribution = {str(k): v / total_pairs * 100 for k, v in similarity_buckets.items()}
    
    # Sorting in descending order
    word_frequencies = dict(sorted(word_frequencies.items(), key=lambda item: item[1], reverse=True))
    source_length_distribution = dict(sorted(source_length_distribution.items(), key=lambda item: item[1], reverse=True))
    target_length_distribution = dict(sorted(target_length_distribution.items(), key=lambda item: item[1], reverse=True))

    return {
        "total_training_pairs": total_pairs,
        "average_source_length": round(np.mean(source_lengths), 2) if source_lengths else 0,
        "average_target_length": round(np.mean(target_lengths), 2) if target_lengths else 0,
        "word_frequencies": word_frequencies,
        "source_length_distribution": source_length_distribution,
        "target_length_distribution": target_length_distribution,
        "similarity_distribution": similarity_distribution
    }