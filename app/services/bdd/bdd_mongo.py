# app\services\bdd\bdd_mongo.py
import torch
from typing import List, Optional

from app.services.bdd.bdd import BDDService
from app.services.bdd.models.model_data import ModelData
from app.services.bdd.models.model_mapping import MODEL_MAPPING
from app.services.bdd.models.training_result import TrainingResult

class MongoBDDService(BDDService):
    """MongoDB implementation of BDDService."""

    def __init__(self, mongo_client: any, database_name: str):
        self.client = mongo_client
        self.database = self.client[database_name]

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
        
        model_class = model_class if model_class is not None else MODEL_MAPPING.get(record.get("neural_network_type"))

        if model_class is None:
            raise ValueError(f"Unknown neural network type: {record.get('neural_network_type')}")
    
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
        """Retrieve all models from MongoDB and deserialize them with the appropriate model class."""
        models = self.database.models.find()
        deserialized_models = []

        for model in models:
            model_class = MODEL_MAPPING.get(model.get("neural_network_type"))
            deserialized_model = ModelData.deserialize(model, model_class=model_class)
            deserialized_models.append(deserialized_model)

        return deserialized_models

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

    def save_training_result(self, result: TrainingResult):
        """Save a training result to MongoDB."""
        self.database.training_results.insert_one(result.to_dict())

    def get_training_results(self, model_name: Optional[str] = None) -> List[TrainingResult]:
        """Retrieve all training results, optionally filtered by model name."""
        query = {"model_name": model_name} if model_name else {}
        results = self.database.training_results.find(query)

        return [TrainingResult.from_dict(res) for res in results]
