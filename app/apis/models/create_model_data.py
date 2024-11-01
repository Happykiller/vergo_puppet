#app\apis\models\create_model_data.py
from typing import List, Optional
from pydantic import BaseModel, Field

# Schema for model creation
class CreateModelData(BaseModel):
    name: str = Field(..., description="Name of the model to create")
    dictionary: Optional[List[List[str]]] = Field(None, description="List of token lists for the model for SIAMESE")
    glossary: Optional[List[str]] = Field(None, description="Reference glossary for the model required for SIAMESE)")
    neural_network_type: str = Field(default="SimpleNN", description="Type of neural network ('SimpleNN', 'LSTMNN', 'GRU', or 'SIAMESE')")

    def check_required_fields(cls, values):
        """
        Checks if required fields are provided based on the neural network type.
        Raises a ValueError if 'dictionary' and 'glossary' are missing for network types SIAMESE.
        """
        neural_network_type = values.get('neural_network_type')
        dictionary = values.get('dictionary')
        glossary = values.get('glossary')

        if neural_network_type == "SIAMESE" and (dictionary is None or glossary is None):
            raise ValueError("Fields 'dictionary' and 'glossary' are required for 'SIAMESE'.")

        return values