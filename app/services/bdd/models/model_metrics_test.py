# app\services\bdd\models\training_result_test.py
import pytest
from datetime import datetime, timezone
from app.services.bdd.models.model_metrics import MetricsModel

def test_training_result_initialization():
    """Test the initialization of a MetricsModel instance."""
    model_name = "test_model"
    metrics = {"accuracy": 0.95, "loss": 0.02}

    # Create an instance without a timestamp (it should default to current UTC time)
    result = MetricsModel(model_name=model_name, metrics=metrics)

    assert result.model_name == model_name
    assert result.metrics == metrics
    assert isinstance(result.timestamp, datetime)
    assert result.timestamp.tzinfo == timezone.utc  # Ensure it's timezone-aware

def test_training_result_with_custom_timestamp():
    """Test initialization with a specific timestamp."""
    model_name = "test_model"
    metrics = {"accuracy": 0.98, "loss": 0.01}
    custom_timestamp = datetime(2024, 2, 6, 12, 0, 0, tzinfo=timezone.utc)

    result = MetricsModel(model_name=model_name, metrics=metrics, timestamp=custom_timestamp)

    assert result.timestamp == custom_timestamp

def test_training_result_to_dict():
    """Test conversion of MetricsModel to dictionary."""
    model_name = "test_model"
    metrics = {"accuracy": 0.90, "loss": 0.05}
    timestamp = datetime(2024, 2, 6, 14, 30, tzinfo=timezone.utc)

    result = MetricsModel(model_name=model_name, metrics=metrics, timestamp=timestamp)
    result_dict = result.to_dict()

    assert result_dict["model_name"] == model_name
    assert result_dict["metrics"] == metrics
    assert result_dict["timestamp"] == timestamp.isoformat()

def test_training_result_from_dict():
    """Test reconstruction of MetricsModel from dictionary."""
    data = {
        "model_name": "test_model",
        "metrics": {"accuracy": 0.92, "loss": 0.03},
        "timestamp": "2024-02-06T14:30:00+00:00"
    }

    result = MetricsModel.from_dict(data)

    assert result.model_name == data["model_name"]
    assert result.metrics == data["metrics"]
    assert result.timestamp == datetime.fromisoformat(data["timestamp"])
    assert result.timestamp.tzinfo == timezone.utc  # Ensure it's timezone-aware
