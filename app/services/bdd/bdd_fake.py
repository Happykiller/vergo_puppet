# app\services\bdd\bdd_fake.py
from typing import Optional

import torch
from app.services.bdd.models.model_data import ModelData
from app.services.logger import logger
from app.services.bdd.bdd import BDDService

class FakeBDDService(BDDService):
    """Fake implementation of the database service for development/testing."""

    def __init__(self):
        # Simulated in-memory database for storing models
        self.models = {}
        # Buffer for caching search results
        self.search_buffer = {}
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
        serialized_data = self.models.get(name)
        if not serialized_data:
            return None
        return ModelData.deserialize(serialized_data, model_class=model_class)

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
        """
        Retrieve all models from the in-memory storage.
        :return: A dictionary containing all serialized models.
        """
        logger.debug(f"self.models: {self.models}")
        return {name: ModelData.deserialize(data) for name, data in self.models.items()}

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
