# app\apis\models\test_model_data.py
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple, Union

from app.apis.models.weather_model_data import WeatherModelData
from app.apis.models.gru_training_model_data import GRUTrainingModelData
from app.apis.models.simple_nn_training_model_data import SimpleNNTrainingModelData

# Schema for model testing
class TestModelData(BaseModel):
    name: str = Field(..., description="Name of the model to test")
    neural_network_type: str = Field(..., description="Type of neural network ('SimpleNN', 'LSTMNN', 'SIAMESE', 'GRU')")
    test_data: Optional[Union[
        List[Tuple[List[str], List[str], float]],  # Siamese
        List[SimpleNNTrainingModelData],  # SimpleNN
        List[GRUTrainingModelData],  # GRU
        List[WeatherModelData],  # LSTM
    ]] = Field(None, description="Test data")
    iterate: Optional[int] = Field(1, description="Optional parameter for the number of iterations to run during testing")
    test_file: Optional[str] = Field(None, description="Name of the file for test")

    def validate_test_data(cls, values):
        """
        Validates the format of test data according to the specified neural network type.
        """
        network_type = values.get('neural_network_type')
        test_data = values.get('test_data')

        if network_type == "SIAMESE" and not all(len(entry) == 3 for entry in test_data):
            raise ValueError("For SIAMESE network, test_data must contain tuples (siamese1, siamese2, target).")
        elif network_type != "SIAMESE" and not all(len(entry) == 2 for entry in test_data):
            raise ValueError("For SimpleNN and LSTMNN, test_data must contain tuples (input, target).")

        return values