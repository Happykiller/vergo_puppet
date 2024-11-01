#app\apis\models\search_model_data.py
from typing import List, Union
from pydantic import BaseModel, Field

from app.apis.models.weather_model_data import WeatherSearchModelData
from app.apis.models.simple_nn_search_model_data import SimpleNNSearchModelData

# Schema for search operation
class SearchModelData(BaseModel):
    name: str = Field(..., description="Name of the model to search within")
    neural_network_type: str = Field(..., description="Type of neural network ('SimpleNN', 'LSTMNN', 'SIAMESE', 'GRU')")
    vector: Union[
        List[str],  # SIAMESE, GRU
        SimpleNNSearchModelData,  # SimpleNN
        WeatherSearchModelData  # LSTM
    ] = Field(..., description="Token list representing the vector to search")