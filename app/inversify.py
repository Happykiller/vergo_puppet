# inversify.py
from app.common import load_env_vars
from app.services.bdd.bdd import BDDService
from app.services.bdd.bdd_fake import FakeBDDService

class Inversify:
  """Dependency injector based on the mode."""

  def __init__(self):
    envs = load_env_vars()
    self._mode = envs["mode"]
    self._dependencies = {}

  def configure(self) -> None:
    """Configure services based on the mode."""
    if self._mode == "dev":
      self._dependencies["bdd_service"] = FakeBDDService()
    elif self._mode == "test":
      self._dependencies["bdd_service"] = FakeBDDService()
    elif self._mode == "prod":
      # Placeholder for a real database service in production
      raise NotImplementedError("Real BDD service not yet implemented.")
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
_inversify_instance: Inversify = None


def get_inversify() -> Inversify:
  """Retrieve the singleton instance of Inversify."""
  global _inversify_instance
  if _inversify_instance is None:
    _inversify_instance = Inversify()
    _inversify_instance.configure()
  return _inversify_instance