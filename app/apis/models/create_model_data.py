#app\apis\models\create_model_data.py
from typing import List, Optional
from pydantic import BaseModel, Field

# Schema for model creation
class CreateModelData(BaseModel):
    name: str = Field(..., description="Name of the model to create")
    dictionary: Optional[List[List[str]]] = Field(None, description="List of token lists for the model for SIAMESE")
    neural_network_type: str = Field(default="SimpleNN", description="Type of neural network ('SimpleNN', 'LSTMNN', 'GRU', or 'SIAMESE')")

    def check_required_fields(cls, values):
        """
        Checks if required fields are provided based on the neural network type.
        Raises a ValueError if 'dictionary' are missing for network types SIAMESE.
        """
        neural_network_type = values.get('neural_network_type')
        dictionary = values.get('dictionary')

        if neural_network_type == "SIAMESE" and (dictionary is None):
            raise ValueError("Fields 'dictionary' are required for 'SIAMESE'.")

        return values