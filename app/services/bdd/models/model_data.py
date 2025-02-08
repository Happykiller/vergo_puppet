# app\services\bdd\models\model_data.py
import io
import torch
import base64
import joblib
import traceback
from enum import Enum
from typing import List, Optional, Dict, Any
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.services.logger import logger

class ModelStatus(Enum):
    INIT = "init"
    CREATED = "created"
    UPDATED = "updated"
    TRAINING = "training"
    TRAINED = "trained"
    SUPER_TRAINING = "super_training"
    SUPER_TRAINED = "super_trained"

class ModelData:
    """
    A class to represent and manage model-related data, including serialization/deserialization
    """
    def __init__(
        self,
        name: str,
        neural_network_type: str,
        status: Optional[ModelStatus] = ModelStatus.INIT,
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
        target_scaler: Optional[StandardScaler] = None,
        indices: Optional[Dict[str, List[int]]] = None,
        targets_mean: Optional[float] = None,
        targets_std: Optional[float] = None,
    ):
        self.name = name
        self.neural_network_type = neural_network_type
        self.status = status
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
        self.target_scaler = target_scaler
        self.indices = indices
        self.targets_mean = targets_mean
        self.targets_std = targets_std

    def serialize(self) -> Dict[str, Any]:
        """
        Serialize the ModelData instance for storage in MongoDB.
        """
        try:
            def convert_keys_to_strings(obj):
                if isinstance(obj, dict):
                    return {str(k): convert_keys_to_strings(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_keys_to_strings(item) for item in obj]
                else:
                    return obj

            def serialize_with_joblib(obj):
                if obj is None:
                    return None
                buffer = io.BytesIO()
                joblib.dump(obj, buffer)
                buffer.seek(0)
                return base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            serialized_data = {
                "name": self.name,
                "neural_network_type": self.neural_network_type,
                "status": self.status.value,
                "dictionary": self.dictionary,
                "indexed_dictionary": self.indexed_dictionary,
                "glossary": self.glossary,
                "word2idx": convert_keys_to_strings(self.word2idx),
                "idx2word": convert_keys_to_strings(self.idx2word),
                "category2idx": convert_keys_to_strings(self.category2idx),
                "idx2category": convert_keys_to_strings(self.idx2category),
                "encoder": serialize_with_joblib(self.encoder),
                "scaler": serialize_with_joblib(self.scaler),
                "target_scaler": serialize_with_joblib(self.target_scaler),
                "indices": serialize_with_joblib(self.indices),
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
                serialized_data["args"] = self.nn_model.args  # Include model arguments
            else:
                serialized_data["nn_model"] = None

            return serialized_data
        except Exception as e:
            logger.error(f"Error message:{str(e)}\nStack trace:\n{traceback.format_exc()}")
            raise Exception(f"[ModelData][#serialize]{str(e)}")

    @classmethod
    def deserialize(cls, data: Dict[str, Any], model_class: Optional[torch.nn.Module] = None) -> "ModelData":
        """
        Deserialize a MongoDB document into a ModelData instance.
        """
        try:
            def convert_keys_from_strings(obj):
                if isinstance(obj, dict):
                    # Convert string keys that represent integers back to integers
                    return {
                        int(k) if k.isdigit() else k: convert_keys_from_strings(v)
                        for k, v in obj.items()
                    }
                elif isinstance(obj, list):
                    return [convert_keys_from_strings(item) for item in obj]
                else:
                    return obj

            def deserialize_with_joblib(encoded_data: Optional[str]):
                if not encoded_data:
                    return None
                buffer = io.BytesIO(base64.b64decode(encoded_data))
                return joblib.load(buffer)

            nn_model = None
            if "nn_model" in data and data["nn_model"]:
                if not callable(model_class):
                    raise TypeError(
                        f"[ModelData][deserialize]: model_class must be callable, got {type(model_class).__name__}"
                    )
                model_binary = base64.b64decode(data["nn_model"])
                buffer = io.BytesIO(model_binary)
                args = data.get("args", {})
                nn_model = model_class(**args)
                nn_model.load_state_dict(torch.load(buffer))

            return cls(
                name=data["name"],
                neural_network_type=data["neural_network_type"],
                status=ModelStatus(data["status"]),
                dictionary=data.get("dictionary"),
                indexed_dictionary=data.get("indexed_dictionary"),
                glossary=data.get("glossary"),
                word2idx=convert_keys_from_strings(data.get("word2idx")),
                idx2word=convert_keys_from_strings(data.get("idx2word")),
                category2idx=convert_keys_from_strings(data.get("category2idx")),
                idx2category=convert_keys_from_strings(data.get("idx2category")),
                encoder=deserialize_with_joblib(data.get("encoder")),
                scaler=deserialize_with_joblib(data.get("scaler")),
                target_scaler=deserialize_with_joblib(data.get("target_scaler")),
                indices=deserialize_with_joblib(data.get("indices")),
                targets_mean=data.get("targets_mean"),
                targets_std=data.get("targets_std"),
                nn_model=nn_model,
            )
        except Exception as e:
            logger.error(f"Error message:{str(e)}\nStack trace:\n{traceback.format_exc()}")
            raise Exception(f"[ModelData][#deserialize]{str(e)}")
