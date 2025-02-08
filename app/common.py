# app\common.py
import os
from pathlib import Path
from dotenv import load_dotenv

# Encapsulate environment variable loading
def load_env_vars():
    """
    Load environment variables and ensure required ones are present.
    """
    # Build the absolute path to the .env files
    env_path = Path(__file__).resolve().parent.parent / ".env"
    env_local_path = Path(__file__).resolve().parent.parent / ".env.local"

    # Load environment variables
    load_dotenv(env_path)
    load_dotenv(env_local_path, override=True)

    # Ensure SECRET_KEY is loaded
    secret_key = os.getenv("SECRET_KEY")
    if not secret_key:
        raise EnvironmentError("SECRET_KEY is missing in the environment variables.")
    
    # Ensure SECRET_KEY is loaded
    mode = os.getenv("MODE")
    if not mode:
        raise EnvironmentError("MODE is missing in the environment variables.")
    
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    mongo_db_name = os.getenv("MONGO_DB_NAME", "puppet")
    debug = os.getenv("DEBUG", "false") == "true"
    
    return {
      "secret_key": secret_key,
      "mode": mode,
      "mongo_uri": mongo_uri,
      "mongo_db_name": mongo_db_name,
      "debug": debug
    }