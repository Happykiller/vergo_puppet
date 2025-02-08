# app\services\bdd\models\training_result.py
from typing import Dict, Any
from datetime import datetime, timezone

class TrainingResult:
    """Represents the result of a model training session."""
    
    def __init__(self, model_name: str, metrics: Dict[str, Any], timestamp: datetime = None):
        """
        Initialize a training result.

        :param model_name: The name of the trained model.
        :param metrics: Dictionary containing the training metrics (e.g., accuracy, loss).
        :param timestamp: Timestamp of the training session.
        """
        self.model_name = model_name
        self.metrics = metrics
        self.timestamp = timestamp or datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the training result to a dictionary format."""
        return {
            "model_name": self.model_name,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat()
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TrainingResult":
        """Reconstruct a TrainingResult object from a dictionary."""
        return TrainingResult(
            model_name=data["model_name"],
            metrics=data["metrics"],
            timestamp=datetime.fromisoformat(data["timestamp"]).replace(tzinfo=timezone.utc)
        )
