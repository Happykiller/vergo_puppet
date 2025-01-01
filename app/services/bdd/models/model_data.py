# app\services\bdd\models\model_data.py
import io
import torch
import base64
import joblib
from typing import List, Optional, Dict, Any
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class ModelData:
    """
    A class to represent and manage model-related data, including serialization/deserialization
    """
    def __init__(
        self,
        name: str,
        neural_network_type: str,
        dictionary: Optional[List[List[str]]] = None,
        indexed_dictionary: Optional[List[List[int]]] = None,
        glossary: Optional[List[str]] = None,
        nn_model: Optional[torch.nn.Module] = None,
        word2idx: Optional[Dict[str, int]] = None,
        idx2word: Optional[Dict[int, str]] = None,
        category2idx: Optional[Dict[str, int]] = None,
        idx2category: Optional[Dict[int, str]] = None,
        encoder: Optional[OneHotEncoder] = None,
        scaler: Optional[StandardScaler] = None,
        indices: Optional[Dict[str, List[int]]] = None,
        targets_mean: Optional[float] = None,
        targets_std: Optional[float] = None,
    ):
        self.name = name
        self.neural_network_type = neural_network_type
        self.dictionary = dictionary
        self.indexed_dictionary = indexed_dictionary
        self.glossary = glossary
        self.nn_model = nn_model
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.category2idx = category2idx
        self.idx2category = idx2category
        self.encoder = encoder
        self.scaler = scaler
        self.indices = indices
        self.targets_mean = targets_mean
        self.targets_std = targets_std

    def serialize(self) -> Dict[str, Any]:
        """
        Serialize the ModelData instance for storage in MongoDB.
        """
        serialized_data = {
            "name": self.name,
            "neural_network_type": self.neural_network_type,
            "dictionary": self.dictionary,
            "indexed_dictionary": self.indexed_dictionary,
            "glossary": self.glossary,
            "word2idx": self.word2idx,
            "idx2word": self.idx2word,
            "category2idx": self.category2idx,
            "idx2category": self.idx2category,
            "encoder": joblib.dumps(self.encoder) if self.encoder else None,
            "scaler": joblib.dumps(self.scaler) if self.scaler else None,
            "indices": joblib.dumps(self.indices) if self.indices else None,
            "targets_mean": self.targets_mean,
            "targets_std": self.targets_std,
        }

        # Serialize the PyTorch model if it exists
        if self.nn_model:
            buffer = io.BytesIO()
            torch.save(self.nn_model.state_dict(), buffer)
            buffer.seek(0)
            model_binary = buffer.read()
            serialized_data["nn_model"] = base64.b64encode(model_binary).decode("utf-8")

        return serialized_data

    @classmethod
    def deserialize(cls, data: Dict[str, Any], model_class: Optional[torch.nn.Module] = None) -> "ModelData":
        """
        Deserialize a MongoDB document into a ModelData instance.
        """
        nn_model = None
        if "nn_model" in data and data["nn_model"] and model_class:
            model_binary = base64.b64decode(data["nn_model"])
            buffer = io.BytesIO(model_binary)
            nn_model = model_class()
            nn_model.load_state_dict(torch.load(buffer))

        return cls(
            name=data["name"],
            neural_network_type=data["neural_network_type"],
            dictionary=data.get("dictionary"),
            indexed_dictionary=data.get("indexed_dictionary"),
            glossary=data.get("glossary"),
            word2idx=data.get("word2idx"),
            idx2word=data.get("idx2word"),
            category2idx=data.get("category2idx"),
            idx2category=data.get("idx2category"),
            encoder=joblib.loads(data["encoder"]) if data.get("encoder") else None,
            scaler=joblib.loads(data["scaler"]) if data.get("scaler") else None,
            indices=joblib.loads(data["indices"]) if data.get("indices") else None,
            targets_mean=data.get("targets_mean"),
            targets_std=data.get("targets_std"),
            nn_model=nn_model,
        )
