# services/bdd_service.py
from abc import ABC, abstractmethod

from app.services.bdd.models.model_data import ModelData

class BDDService(ABC):
    """Abstract Base Class for database service."""

    @abstractmethod
    def save_model(name: str, data: ModelData):
        pass

    @abstractmethod
    def get_model(name: str):
        pass

    @abstractmethod
    def model_exists(name: str):
        pass

    @abstractmethod
    def update_model(name: str, data: ModelData):
        pass

    @abstractmethod
    def get_all_models():
        pass
    
    @abstractmethod
    def save_search_result(model_name: str, search_query: str, result: dict):
        pass

    @abstractmethod
    def get_search_result(model_name: str, search_query: str):
        pass

    @abstractmethod
    def clear_search_buffer(model_name: str):
        pass