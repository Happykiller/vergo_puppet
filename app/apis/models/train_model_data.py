#app\apis\models\create_model_data.py
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple, Union

from app.apis.models.weather_model_data import WeatherModelData
from app.apis.models.gru_training_model_data import GRUTrainingModelData
from app.apis.models.simple_nn_training_model_data import SimpleNNTrainingModelData

# Schema for model training
class TrainModelData(BaseModel):
    name: str = Field(..., description="Name of the model to train")
    neural_network_type: Optional[str] = Field(description="Type of neural network ('SimpleNN', 'LSTMNN', 'GRU', or 'SIAMESE')")
    training_data: Optional[Union[
        List[WeatherModelData],  # LSTM
        List[Tuple[List[str], List[str], float]],  # Siamese
        List[SimpleNNTrainingModelData],  # SimpleNN
        List[GRUTrainingModelData]  # GRU
    ]] = Field(None, description="Training data")
    train_file: Optional[str] = Field(None, description="Name of the file for training")

    def __init__(self, **data):
        """
        Initializes the TrainModelData and assigns default values for missing elements in siamese training data.
        """
        super().__init__(**data)
        if self.neural_network_type == 'SIAMESE' and self.training_data:
            for i, elem in enumerate(self.training_data):
                # Ensures each tuple is complete with target initialized to 0.0 if missing
                if len(elem) == 2:
                    siamese1, siamese2 = elem
                    self.training_data[i] = (siamese1, siamese2, 0.0)
                elif len(elem) == 3:
                    siamese1, siamese2, target = elem
                    self.training_data[i] = (siamese1, siamese2, target or 0.0)