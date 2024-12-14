# app/apis/models/gru_training_model_data.py
from typing import List
from pydantic import BaseModel, Field

class GRUTrainingModelData(BaseModel):
    """
    GRUTrainingModelData is a data model for representing tokenized support tickets 
    and their assigned categories for training purposes with GRU-based neural networks.

    Attributes:
        category (str): The assigned category for a support ticket.
        tokens (List[str]): A tokenized description of a support ticket.
    """
    
    category: str = Field(..., description="The assigned category for a support ticket")
    tokens: List[str] = Field(..., description="A tokenized description of a support ticket")
