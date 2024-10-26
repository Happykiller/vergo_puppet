from pydantic import BaseModel, Field

class SimpleNNSearchData(BaseModel):
    type: int = Field(..., description="Type de la propriété")
    surface: int = Field(..., description="Surface de la propriété en m²")
    pieces: int = Field(..., description="Nombre de pièces")
    floor: int = Field(..., description="Étage")
    parking: int = Field(..., description="Nombre de garages")
    balcon: int = Field(..., description="Nombre de balcons")
    ascenseur: int = Field(..., description="Présence d'un ascenseur (0 ou 1)")
    orientation: int = Field(..., description="Orientation de la propriété")
    transports: int = Field(..., description="Proximité des transports (0 ou 1)")
    neighborhood: int = Field(..., description="Qualité du quartier (1 à 10)")