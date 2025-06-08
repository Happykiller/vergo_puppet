# app/services/bdd/models/model_thing.py
from typing import Dict, Any
from pydantic import BaseModel, Field

class ThingModel(BaseModel):
    id: str = Field(..., description="Unique identifier for the indexed item")
    vector: list[float] = Field(..., description="Vector embedding of the item")
    text: str = Field(..., description="Flattened text used for generating the embedding")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional metadata about the item")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "vector": self.vector,
            "text": self.text,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ThingModel":
        return ThingModel(
            id=data["id"],
            vector=data["vector"],
            text=data["text"],
            metadata=data.get("metadata", {})
        )
