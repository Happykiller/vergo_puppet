from pydantic import BaseModel
from typing import List

class PrepareCacheData(BaseModel):
    name: str  # Name of the model
    neural_network_type: str  # Neural network type (e.g., SIAMESE)
    search_vectors: List[List[str]]  # Collection of search vectors to precompute
