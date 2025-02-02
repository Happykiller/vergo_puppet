# app\apis\models\super_train_from_file_model_data.py
from typing import List, Tuple, Union
from pydantic import BaseModel, Field

from app.apis.models.gru_training_model_data import GRUTrainingModelData
from app.apis.models.simple_nn_training_model_data import SimpleNNTrainingModelData
from app.apis.models.weather_model_data import WeatherModelData

class SuperTrainFromFileModelData(BaseModel):
    name: str  # Name of the model to train
    neural_network_type: str  # Neural network type (e.g., SIAMESE)
    file_name: str  # file name
    test_data: Union[
        List[Tuple[List[str], List[str], float]],  # Siamese
        List[SimpleNNTrainingModelData],  # SimpleNN
        List[GRUTrainingModelData],  # GRU
        List[WeatherModelData],  # LSTM
    ] = Field(..., description="Test data")
