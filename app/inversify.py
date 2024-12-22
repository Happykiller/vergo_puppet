# inversify.py
import os
from pathlib import Path
from dotenv import load_dotenv
from app.services.bdd import AbstractBDDService, FakeBDDService

class Inversify:
  """Dependency injector based on the environment."""

  def __init__(self):
      
    # Build the absolute path to the .env files
    env_path = Path(__file__).resolve().parent.parent / ".env"
    env_local_path = Path(__file__).resolve().parent.parent / ".env.local"

    # Load environment variables
    load_dotenv(env_path)
    load_dotenv(env_local_path, override=True)

    self._env = os.getenv("MODE", "dev")  # Default to "dev" if MODE is not set
    self._dependencies = {}

  def configure(self) -> None:
    """Configure services based on the environment."""
    if self._env == "dev":
      self._dependencies["bdd_service"] = FakeBDDService()
    elif self._env == "prod":
      # Placeholder for a real database service in production
      raise NotImplementedError("Real BDD service not yet implemented.")
    else:
      raise ValueError(f"Unknown environment: {self._env}")

  def get(self, service_name: str):
    """Retrieve the configured service by name."""
    service = self._dependencies.get(service_name)
    if not service:
      raise ValueError(f"Service '{service_name}' not configured.")
    return service
  
  def get_bdd(self) -> AbstractBDDService:
    """Retrieve the configured database service."""
    return self.get("bdd_service")

# Singleton instance
_inversify_instance: Inversify = None


def get_inversify() -> Inversify:
  """Retrieve the singleton instance of Inversify."""
  global _inversify_instance
  if _inversify_instance is None:
    _inversify_instance = Inversify()
    _inversify_instance.configure()
  return _inversify_instance