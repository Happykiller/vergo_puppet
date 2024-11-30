# app/apis/models/update_model_data.py
from pydantic import BaseModel
from typing import List, Optional

class UpdateModelData(BaseModel):
    name: str  # Name of the model to update
    neural_network_type: str  # Neural network type (e.g., SIAMESE)
    dictionary: Optional[List[List[str]]] = None  # New dictionary, if applicable
    glossary: Optional[List[str]] = None  # New glossary, if applicable
