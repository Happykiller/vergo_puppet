# inversify.py
from pymongo import MongoClient # type: ignore

from app.common import load_env_vars
from app.services.logger import logger
from app.services.bdd.bdd import BDDService
from app.services.bdd.bdd_fake import FakeBDDService
from app.services.bdd.bdd_mongo import MongoBDDService

class Inversify:
  """Dependency injector based on the mode."""

  def __init__(self):
    envs = load_env_vars()
    self._mode = envs["mode"]
    self._bdd = envs["bdd"]
    self._mongo_uri = envs["mongo_uri"]
    self._mongo_db_name = envs["mongo_db_name"]
    
    # Log initial environment configuration
    logger.info(f"[Inversify] Mode: {self._mode}")
    debug = envs["debug"]
    logger.info(f"[Inversify] Debug: {debug}")
    logger.info(f"[Inversify] BDD: {self._bdd}")
    logger.info(f"[Inversify] Mongo URI: {self._mongo_uri}")
    logger.info(f"[Inversify] Mongo DB Name: {self._mongo_db_name}")
        
    self._dependencies = {}

  def configure(self) -> None:
    """Configure services based on the mode."""
    if self._bdd == "fake":
      self._dependencies["bdd_service"] = FakeBDDService()
    elif self._bdd == "mongo":
      # Placeholder for a real database service in production
      mongo_client = MongoClient(self._mongo_uri)
      logger.info(f"[Inversify] Connected to MongoDB at success")
      self._dependencies["bdd_service"] = MongoBDDService(mongo_client, self._mongo_db_name)
    else:
      raise ValueError(f"Unknown mode: {self._mode}")

  def get(self, service_name: str):
    """Retrieve the configured service by name."""
    service = self._dependencies.get(service_name)
    if not service:
      raise ValueError(f"Service '{service_name}' not configured.")
    return service
  
  def get_bdd(self) -> BDDService:
    """Retrieve the configured database service."""
    return self.get("bdd_service")

# Singleton instance
_inversify_instance: Inversify = None # type: ignore


def get_inversify() -> Inversify:
  """Retrieve the singleton instance of Inversify."""
  global _inversify_instance
  if _inversify_instance is None:
    _inversify_instance = Inversify()
    _inversify_instance.configure()
  return _inversify_instance