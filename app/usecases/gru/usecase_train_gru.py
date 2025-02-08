# app\usecases\gru\usecase_train_gru.py
import traceback
from collections import Counter
from typing import List, Dict, NamedTuple

from app.inversify import Inversify
from app.services.bdd.models.training_result import TrainingResult
from app.services.logger import logger
from app.neural_network.nn_gru import GRUClassifier, train_gru
from app.services.bdd.models.model_data import ModelData, ModelStatus
from app.apis.models.gru_training_model_data import GRUTrainingModelData

class TrainGRUUsecaseDto(NamedTuple):
    name: str
    inversify: Inversify
    training_data: List[GRUTrainingModelData]

def train_model_gru(dto: TrainGRUUsecaseDto):
    """
    Trains the GRU model with the provided training data.
    :param name: Model name.
    :param training_data: List of training data.
    """
    try:
        # Fetch Bdd
        bdd = dto.inversify.get_bdd()
        
        model = bdd.get_model(dto.name, GRUClassifier)
        
        # Check if model exists
        if model is None or not model:
            raise Exception("Model not found")
        
        # Check if training data is valid
        if dto.training_data is None or len(dto.training_data) == 0:
            raise Exception("No training data provided or data is empty")
        
        logger.info("Machine learning type for training: GRU")
        
        # Update the model in storage
        if(model.status != ModelStatus.SUPER_TRAINING):
            model.status = ModelStatus.TRAINING
            bdd.update_model(model)
        
        # Display training data statistics
        num_documents = len(dto.training_data)
        categories = [data.category for data in dto.training_data]
        category_counts = Counter(categories)
        num_categories = len(category_counts)
        
        logger.info(f"Number of documents: {num_documents}")
        logger.info(f"Number of categories: {num_categories}")
        logger.info("Category distribution:")
        for category, count in category_counts.items():
            percentage = (count / num_documents) * 100
            logger.info(f" - {category}: {count} documents ({percentage:.2f}%)")
        
        # Analyze sequence lengths
        sequence_lengths = [len(data.tokens) for data in dto.training_data]
        max_seq_length = max(sequence_lengths)
        min_seq_length = min(sequence_lengths)
        avg_seq_length = sum(sequence_lengths) / num_documents
        
        logger.info(f"Max sequence length: {max_seq_length}")
        logger.info(f"Min sequence length: {min_seq_length}")
        logger.info(f"Average sequence length: {avg_seq_length:.2f}")
        
        # Build vocabulary for the model
        word2idx, idx2word = build_vocab(dto.training_data)
        vocab_size = len(word2idx)
        logger.info(f"Vocabulary size: {vocab_size}")
        
        # Map categories to indices
        category2idx, idx2category = build_category_mapping(dto.training_data)
        
        # Prepare sequences and labels for training
        sequences, labels = prepare_sequences(dto.training_data, word2idx, category2idx, max_seq_length)
        
        # Set model parameters
        num_classes = len(category2idx)

        # Train the model
        logger.info("Training model...")
        # Train the model and retrieve statistics
        nn_model, training_stats = train_gru(vocab_size, num_classes, sequences, labels)
        
        # Save trained model, mappings, and hyperparameters
        bdd.update_model(ModelData(
            name=model.name,
            neural_network_type=model.neural_network_type,
            status=ModelStatus.TRAINED if model.status != ModelStatus.SUPER_TRAINING else ModelStatus.SUPER_TRAINING,
            nn_model=nn_model,
            word2idx=word2idx,
            idx2word=idx2word,
            category2idx=category2idx,
            idx2category=idx2category
        ))

        bdd.save_training_result(TrainingResult(
            model_name=dto.name,
            metrics=training_stats
        ))
        
        return {
            "status": "Training complete",
            "model_name": dto.name,
            "training_stats": training_stats
        }
    except Exception as e:
        logger.error(f"Error message:{str(e)}\nStack trace:\n{traceback.format_exc()}")
        raise Exception(f"[#search_model_simple_nn]{str(e)}")

def build_vocab(training_data: List[GRUTrainingModelData]):
    """
    Builds a vocabulary from training data.
    :param training_data: List of training data.
    :return: Word-to-index and index-to-word dictionaries.
    """
    # Gather all tokens from training data
    all_tokens = [token for data in training_data for token in data.tokens]
    token_counts = Counter(all_tokens)
    
    # Map words to indices, starting at 1 (0 reserved for padding)
    word2idx = {word: idx+1 for idx, (word, _) in enumerate(token_counts.most_common())}
    word2idx['<PAD>'] = 0  # Padding token
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    return word2idx, idx2word

def build_category_mapping(training_data: List[GRUTrainingModelData]):
    """
    Builds mappings between categories and indices.
    :param training_data: List of training data.
    :return: Category-to-index and index-to-category dictionaries.
    """
    categories = set(data.category for data in training_data)
    category2idx = {category: idx for idx, category in enumerate(categories)}
    idx2category = {idx: category for category, idx in category2idx.items()}
    
    return category2idx, idx2category

def prepare_sequences(training_data: List[GRUTrainingModelData], word2idx: Dict[str, int], category2idx: Dict[str, int], max_seq_length: int):
    """
    Prepares sequences and labels for training.
    :param training_data: List of training data.
    :param word2idx: Word-to-index dictionary.
    :param category2idx: Category-to-index dictionary.
    :param max_seq_length: Maximum sequence length for padding.
    :return: Tensor of sequences and labels.
    """
    sequences = []
    labels = []
    
    for data in training_data:
        # Convert tokens to indices
        seq = [word2idx.get(token, word2idx['<PAD>']) for token in data.tokens]
        
        # Pad sequences to ensure uniform length
        seq += [word2idx['<PAD>']] * (max_seq_length - len(seq))
        
        sequences.append(seq)
        labels.append(category2idx[data.category])
    
    return sequences, labels
