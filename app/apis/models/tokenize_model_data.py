# app/apis/models/tokenize_model_data.py
from typing import List
from pydantic import BaseModel, Field

class ModelTokenizeData(BaseModel):
    """
    ModelTokenizeData represents an individual incident with an identifier
    and a description, used as input for tokenization processes.

    Attributes:
        incidentId (str): The unique identifier of the incident.
        description (str): The description text of the incident.
    """
    
    incidentId: str = Field(..., description="The unique identifier of the incident")
    description: str = Field(..., description="The description text of the incident")

# Schema for tokenization
class TokenizeModelData(BaseModel):
    """
    TokenizeModelData represents a schema containing multiple incidents for
    batch processing in tokenization.

    Attributes:
        data (List[ModelTokenizeData]): List of incidents to be tokenized.
    """
    
    data: List[ModelTokenizeData] = Field(..., description="List of incidents to be tokenized")
