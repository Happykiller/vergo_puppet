# app\services\bdd\bdd_fake.py
import torch
import traceback
from typing import List, Optional

from app.services.logger import logger
from app.services.bdd.bdd import BDDService
from app.services.bdd.models.model_data import ModelData
from app.services.bdd.models.model_mapping import MODEL_MAPPING
from app.services.bdd.models.model_metrics import MetricsModel

class FakeBDDService(BDDService):
    """Fake implementation of the database service for development/testing."""

    def __init__(self):
        # Simulated in-memory database for storing models
        self.models = {}
        # Buffer for caching search results
        self.search_buffer = {}
        self.metrics = []
        logger.debug("Bdd: Fake Bdd initialized")

    def save_model(self, data: ModelData):
        """
        Save a model with the given name and data to the in-memory storage.
        :param name: The name of the model to save.
        :param data: An instance of ModelData.
        """
        if not isinstance(data, ModelData):
            raise ValueError("Data must be an instance of ModelData.")
        self.models[data.name] = data.serialize()

    def get_model(self, name: str, model_class: Optional[torch.nn.Module] = None) -> Optional[ModelData]:
        """
        Retrieve a model by its name from the in-memory storage.
        :param name: The name of the model to retrieve.
        :param model_class: The class of the PyTorch model (optional).
        :return: An instance of ModelData if found, otherwise None.
        """
        model = self.models.get(name)
        
        if not model:
            return None
        
        model_class = model_class if model_class is not None else MODEL_MAPPING.get(model.get("neural_network_type"))

        if model_class is None:
            raise ValueError(f"Unknown neural network type: {model.get('neural_network_type')}")
        
        return ModelData.deserialize(model, model_class=model_class)

    def model_exists(self, name: str) -> bool:
        """
        Check if a model with the specified name exists in the in-memory storage.
        :param name: The name of the model to check.
        :return: True if the model exists, otherwise False.
        """
        return name in self.models

    def update_model(self, data: ModelData):
        """
        Update an existing model's data with new information.
        :param name: The name of the model to update.
        :param data: An instance of ModelData.
        """
        if data.name not in self.models:
            raise ValueError(f"Model '{data.name}' does not exist.")
        if not isinstance(data, ModelData):
            raise ValueError("Data must be an instance of ModelData.")
        self.models[data.name] = data.serialize()

    def get_all_models(self):
        try :
            """
            Retrieve all models from the in-memory storage.
            :return: A dictionary containing all serialized models.
            """
            models = self.models.items()
            deserialized_models = []

            for _, model in models:
                model_class = MODEL_MAPPING.get(model.get("neural_network_type"))
                deserialized_model = ModelData.deserialize(model, model_class=model_class)
                deserialized_models.append(deserialized_model)

            return deserialized_models
        except Exception as e:
            logger.error(f"Error message:{str(e)}\nStack trace:\n{traceback.format_exc()}")
            raise Exception(f"[FakeBDDService#get_all_models]{str(e)}")

    def save_search_result(self, model_name: str, search_query: str, result: dict):
        """
        Save a search result to the buffer.
        :param model_name: Name of the model used for the search.
        :param search_query: Search query as a stringified representation of the input.
        :param result: Result of the search.
        """
        if model_name not in self.search_buffer:
            self.search_buffer[model_name] = {}
        self.search_buffer[model_name][search_query] = result

    def get_search_result(self, model_name: str, search_query: str):
        """
        Retrieve a search result from the buffer.
        :param model_name: Name of the model used for the search.
        :param search_query: Search query as a stringified representation of the input.
        :return: Cached result if available, otherwise None.
        """
        return self.search_buffer.get(model_name, {}).get(search_query)

    def clear_search_buffer(self, model_name: str):
        """
        Clear the search buffer for a specific model.
        :param model_name: Name of the model whose buffer needs clearing.
        """
        if model_name in self.search_buffer:
            del self.search_buffer[model_name]

    def save_metrics(self, result: MetricsModel):
        """Save a training result to the in-memory database."""
        self.metrics.append(result)

    def get_metrics(self, model_name: Optional[str] = None) -> List[MetricsModel]:
        """Retrieve all training results, optionally filtered by model name."""
        if model_name:
            return [res for res in self.metrics if res.model_name == model_name]
        return self.metrics