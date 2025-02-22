# app\apis\models\siamese_train_model_data.py
from pydantic import BaseModel
from typing import List, Tuple

class SiameseTrainingModelData(BaseModel):
  __root__: Tuple[List[str], List[str], float]  