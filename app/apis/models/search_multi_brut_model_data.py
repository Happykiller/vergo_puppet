# app\apis\models\search_multi_brut_model_data.py
from typing import List, Union
from pydantic import BaseModel, Field
from app.apis.models.tokenize_model_data import ModelTokenizeData

# Schema for search operation
class SearchBrutMultiModelData(BaseModel):
    name: str = Field(..., description="Name of the model to search multi brut within")
    neural_network_type: str = Field(..., description="Type of neural network ('GRU')")
    documents: Union[
        List[ModelTokenizeData]  # GRU
    ] = Field(..., description="Token list representing documents to search")