# app/usecases/usecase_get_things.py
import traceback
from typing import Optional

from app.inversify import Inversify
from app.services.logger import logger
from app.services.bdd.models.model_thing import ThingModel

def get_things_usecase(ids: Optional[list[str]], inversify: Inversify) -> list[ThingModel]:
    """
    Retrieves one or more ThingModel entries by their ID(s), or all if None.

    :param ids: Optional list of Thing IDs to fetch
    :param inversify: Dependency injection container
    :return: List of ThingModel instances
    """
    try:
        bdd = inversify.get_bdd()
        return bdd.get_things(ids)
    except Exception as e:
        logger.error(f"Error message: {str(e)}\nStack trace:\n{traceback.format_exc()}")
        raise Exception(f"[get_things_usecase] {str(e)}")
