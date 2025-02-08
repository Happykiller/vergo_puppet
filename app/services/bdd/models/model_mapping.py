# app\services\bdd\models\model_mapping.py
import torch
from typing import Dict, Type

# Importation des modèles neuronaux
from app.neural_network.nn_lstm import LSTMNN
from app.neural_network.nn_simple import SimpleNN
from app.neural_network.nn_gru import GRUClassifier
from app.neural_network.nn_siamese import SiameseLSTM

# Mapping des types de réseaux neuronaux vers leurs classes respectives
MODEL_MAPPING: Dict[str, Type[torch.nn.Module]] = {
    "GRU": GRUClassifier,
    "LSTM": LSTMNN,
    "SIAMESE": SiameseLSTM,
    "SimpleNN": SimpleNN,
}