# app\apis\models\super_train_model_data.py
from pydantic import BaseModel, Field
from typing import List, Optional, Union
from app.apis.models.weather_model_data import WeatherModelData
from app.apis.models.gru_training_model_data import GRUTrainingModelData
from app.apis.models.siamse_training_model_data import SiameseTrainingModelData
from app.apis.models.simple_nn_training_model_data import SimpleNNTrainingModelData

class SuperTrainModelData(BaseModel):
    name: str = Field(..., description="Name of the model")
    neural_network_type: str = Field(..., description="Type of the model")
    iterate: Optional[int] = Field(1, description="Optional parameter for the number of iterations to run during testing")
    train_data: Optional[Union[
        List[SiameseTrainingModelData], # Siamese
        List[SimpleNNTrainingModelData], # SimpleNN
        List[GRUTrainingModelData], # GRU
        List[WeatherModelData], # LSTM
    ]] = Field(None, description="Train data")
    test_data: Optional[Union[
        List[SiameseTrainingModelData], # Siamese
        List[SimpleNNTrainingModelData], # SimpleNN
        List[GRUTrainingModelData], # GRU
        List[WeatherModelData], # LSTM
    ]] = Field(None, description="Test data")
    train_file: Optional[str] = Field(None, description="Name of the file for training")
    test_file: Optional[str] = Field(None, description="Name of the file for tests")