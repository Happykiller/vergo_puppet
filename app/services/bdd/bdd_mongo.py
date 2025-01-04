# app\services\bdd\bdd_mongo.py
import torch
from typing import Optional
from pymongo import MongoClient # type: ignore

from app.services.bdd.models.model_data import ModelData

from app.services.logger import logger
from app.services.bdd.bdd import BDDService

class MongoBDDService(BDDService):
    """MongoDB implementation of BDDService."""

    def __init__(self, uri: str, database_name: str):
        self.client = MongoClient(uri)
        self.database = self.client[database_name]
        logger.info(f"Connected to MongoDB at {uri}")

    def save_model(self, data: ModelData):
        """Save a ModelData instance to MongoDB."""
        serialized_data = data.serialize()
        self.database.models.update_one(
            {"name": data.name}, {"$set": serialized_data}, upsert=True
        )

    def get_model(self, name: str, model_class: Optional[torch.nn.Module] = None) -> Optional[ModelData]:
        """Retrieve a ModelData instance from MongoDB."""
        record = self.database.models.find_one({"name": name})
        if not record:
            return None
        return ModelData.deserialize(record, model_class=model_class)

    def update_model(self, data: ModelData):
        """
        Update an existing model in MongoDB.

        :param name: The name of the model to update.
        :param data: An instance of ModelData containing updated model information.
        """
        if not isinstance(data, ModelData):
            raise ValueError("The 'data' parameter must be an instance of ModelData.")

        # Sérialiser les données du modèle
        serialized_data = data.serialize()

        # Mettre à jour les données dans MongoDB
        self.database.models.update_one({"name": data.name}, {"$set": serialized_data}, upsert=True)

    def model_exists(self, name: str):
        """Check if a model exists in MongoDB."""
        return self.database.models.count_documents({"name": name}) > 0

    def get_all_models(self):
        """Retrieve all models from MongoDB."""
        return list(self.database.models.find())

    def save_search_result(self, model_name: str, search_query: str, result: dict):
        """Save a search result to MongoDB."""
        self.database.search_results.update_one(
            {"model_name": model_name, "search_query": search_query},
            {"$set": {"result": result}},
            upsert=True
        )

    def get_search_result(self, model_name: str, search_query: str):
        """Retrieve a search result from MongoDB."""
        record = self.database.search_results.find_one(
            {"model_name": model_name, "search_query": search_query}
        )
        return record["result"] if record else None

    def clear_search_buffer(self, model_name: str):
        """Clear the search buffer for a specific model."""
        self.database.search_results.delete_many({"model_name": model_name})
