# app/usecases/usecase_get_things.py
import traceback
from typing import Optional

from app.inversify import Inversify
from app.services.logger import logger
from app.services.bdd.models.model_thing import ThingModel

def get_things_usecase(collection_name: str, ids: Optional[list[str]], inversify: Inversify) -> list[ThingModel]:
    try:
        bdd = inversify.get_bdd()
        return bdd.get_things(collection=collection_name, ids=ids)
    except Exception as e:
        logger.error(f"[get_things_usecase] Error: {str(e)}\n{traceback.format_exc()}")
        raise Exception(f"[get_things_usecase] {str(e)}")
