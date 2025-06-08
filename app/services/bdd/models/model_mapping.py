# app\services\bdd\models\model_mapping.py
import torch
from typing import Dict, Type

# Import neural network models
from app.neural_network.nn_lstm import LSTMNN
from app.neural_network.nn_simple import SimpleNN
from app.neural_network.nn_gru import GRUClassifier
from app.neural_network.nn_siamese import SiameseLSTM
from app.neural_network.nn_embedding import UniversalEmbeddingModel

# Map neural network types to their respective classes
MODEL_MAPPING: Dict[str, Type[torch.nn.Module]] = {
    "GRU": GRUClassifier,
    "LSTM": LSTMNN,
    "SIAMESE": SiameseLSTM,
    "SimpleNN": SimpleNN,
    "EMBEDDING": UniversalEmbeddingModel
}