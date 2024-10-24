from typing import List
from pydantic import BaseModel, Field

class GRUTrainingData(BaseModel):
    category: str = Field(..., description="La catégorie")
    tokens: List[str] = Field(..., description="La description d'un ticket au support tokenifier")