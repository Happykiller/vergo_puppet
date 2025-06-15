# app\common.py
import os
import json
from pathlib import Path
from dotenv import load_dotenv # type: ignore
from fastapi import HTTPException # type: ignore
from typing import Any, Optional, Dict

FILES_DIR = Path("files")

# Internal cache for singleton behavior
_env_cache: Optional[Dict[str, Any]] = None

# Encapsulate environment variable loading
def load_env_vars():
    """
    Load and cache environment variables only once.
    """
    mode = os.getenv("MODE", "test").lower()
    
    global _env_cache
    if _env_cache is not None:
        return _env_cache
    
    if(mode != "test") :
        # Paths to .env files
        root_dir = Path(__file__).resolve().parent.parent
        env_path = root_dir / ".env"
        env_local_path = root_dir / ".env.local"
        env_prod_path = root_dir / ".env.prod"

        # Load in order: .env → .env.local → .env.prod if MODE=prod
        load_dotenv(env_path, override=True)
        load_dotenv(env_local_path, override=True)
        
        if mode == 'prod':
            load_dotenv(env_prod_path, override=True)

    # Validate essential variables
    secret_key = os.getenv("SECRET_KEY")
    if not secret_key:
        raise EnvironmentError("SECRET_KEY is missing in the environment variables.")
    
    _env_cache = {
            "secret_key": secret_key,
            "mode": mode,
            "bdd": os.getenv("BDD", "fake"),
            "mongo_uri": os.getenv("MONGO_URI", "mongodb://localhost:27017"),
            "mongo_db_name": os.getenv("MONGO_DB_NAME", "puppet"),
            "debug": os.getenv("DEBUG", "false").lower() == "true",
        }

    return _env_cache

def parse_input_data(inline_data: Optional[Any], file_name: Optional[str]) -> Any:
    """
    Parse training data either from inline input or from a JSON file.

    :param inline_data: The training data provided directly in the request body.
    :param file_name: The name of the JSON file containing training data.
    :return: Parsed training data (list, dict, etc. depending on file content).
    :raises HTTPException: If the file is not found or if neither inline data
                           nor file name is provided.
    """
    if inline_data is not None:
        # Inline training data provided
        return inline_data

    if file_name is not None:
        # Load training data from file
        file_path = FILES_DIR / file_name
        if not file_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"File '{file_path}' not found."
            )
        try:
            with file_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=422,
                detail=f"Failed to parse JSON from file '{file_path}': {str(e)}"
            )

    # Neither inline data nor file name
    raise HTTPException(
        status_code=422,
        detail="You must provide either data or file."
    )
    
def format_time(milliseconds):
    """Format time from seconds to DD:HH:MM:SS:SSS"""
    milliseconds = int(milliseconds * 1000)  # Convert seconds to milliseconds
    seconds, ms = divmod(milliseconds, 1000)
    minutes, sec = divmod(seconds, 60)
    hours, mins = divmod(minutes, 60)
    days, hrs = divmod(hours, 24)
    return f"{days:02}:{hrs:02}:{mins:02}:{sec:02}:{ms:03}"