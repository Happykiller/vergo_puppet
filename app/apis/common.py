# app\apis\common.py
import json
from typing import Union
from pathlib import Path
from fastapi import HTTPException # type: ignore

FILES_DIR = Path("files")
FILES_DIR.mkdir(parents=True, exist_ok=True)

def load_json_file(filepath: Union[str, Path]) -> Union[dict, list]:
    """
    Loads and parses a JSON file from the given path.
    Raises HTTPException if the file does not exist or is invalid JSON.

    :param filepath: Path to the JSON file (str or Path)
    :return: Parsed JSON object (dict or list)
    """
    file_path = Path(filepath)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File '{file_path}' not found.")

    try:
        with file_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading JSON from '{file_path}': {e}")