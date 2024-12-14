# app/apis/models/simple_nn_search_model_data.py
from pydantic import BaseModel, Field

class SimpleNNSearchModelData(BaseModel):
    """
    SimpleNNSearchModelData is a data model representing the features of a property
    used for neural network search operations.

    Attributes:
        type (int): The type of the property.
        surface (int): The surface area of the property in square meters.
        pieces (int): The number of rooms.
        floor (int): The floor number.
        parking (int): The number of garages.
        balcon (int): The number of balconies.
        ascenseur (int): Presence of an elevator (0 or 1).
        orientation (int): The orientation of the property.
        transports (int): Proximity to public transport (0 or 1).
        neighborhood (int): Quality of the neighborhood (1 to 10).
    """

    type: int = Field(..., description="The type of the property")
    surface: int = Field(..., description="The surface area of the property in square meters")
    pieces: int = Field(..., description="The number of rooms")
    floor: int = Field(..., description="The floor number")
    parking: int = Field(..., description="The number of garages")
    balcon: int = Field(..., description="The number of balconies")
    ascenseur: int = Field(..., description="Presence of an elevator (0 or 1)")
    orientation: int = Field(..., description="The orientation of the property")
    transports: int = Field(..., description="Proximity to public transport (0 or 1)")
    neighborhood: int = Field(..., description="Quality of the neighborhood (1 to 10)")
