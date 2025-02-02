# app\apis\models\train_from_file_model_data.py
from pydantic import BaseModel

class TrainFromFileModelData(BaseModel):
    name: str  # Name of the model to train
    neural_network_type: str  # Neural network type (e.g., SIAMESE)
    file_name: str  # file name
