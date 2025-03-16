# app\common.py
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from typing import Any, Optional
from fastapi import HTTPException # type: ignore

FILES_DIR = Path("files")

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