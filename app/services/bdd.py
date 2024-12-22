# services/bdd_service.py
from abc import ABC, abstractmethod
from typing import Any

class AbstractBDDService(ABC):
    """Abstract Base Class for database service."""

    @abstractmethod
    def get_data(self, key: str) -> Any:
        """Retrieve data from the database."""
        pass

    @abstractmethod
    def save_data(self, key: str, value: Any) -> None:
        """Save data to the database."""
        pass
    
    @abstractmethod
    def who_is(self) -> str:
        """Retrieve who is."""
        pass

class FakeBDDService(AbstractBDDService):
    """Fake implementation of the database service for development/testing."""

    def __init__(self):
        self.storage = {}  # Simple in-memory storage for fake data

    def get_data(self, key: str) -> Any:
        return self.storage.get(key, None)

    def save_data(self, key: str, value: Any) -> None:
        self.storage[key] = value

    def who_is(self) -> str:
        return "Fake Bdd"
