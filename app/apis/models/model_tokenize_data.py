from pydantic import BaseModel, Field

class ModelTokenizeData(BaseModel):
    incidentId: str = Field(..., description="Type de la propriété")
    description: str = Field(..., description="Type de la propriété")