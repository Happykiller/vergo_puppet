#app\apis\models\tokenize_model_data.py
from typing import List
from pydantic import BaseModel, Field

class ModelTokenizeData(BaseModel):
    incidentId: str = Field(..., description="Type de la propriété")
    description: str = Field(..., description="Type de la propriété")

# Schema for tokenization
class TokenizeModelData(BaseModel):
    data: List[ModelTokenizeData] = Field(..., description="Data for tokenization")
