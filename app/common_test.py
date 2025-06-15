# app/common_test.py
import json # type: ignore
import pytest # type: ignore
from fastapi import HTTPException # type: ignore
from unittest.mock import patch, mock_open

from app.common import load_env_vars, parse_input_data, FILES_DIR

def test_load_env_vars():
    """
    Test that load_env_vars returns the mocked environment variables.
    """
    # Arrange
    
    # Act
    envs = load_env_vars()

    # Assert
    assert envs["secret_key"] == "mocked-secret-key"
    assert envs["mode"] == "test"

def test_parse_input_data_inline_only():
    """
    Should return the inline data directly when provided.
    """
    # Arrange
    inline_data = {"key": "value"}
    
    # Act
    result = parse_input_data(inline_data, None)

    # Assert
    assert result == inline_data, "Expected to return inline data as-is."


def test_parse_input_data_file_only_valid():
    """
    Should load and return JSON data from a file if inline data is None.
    """
    # Arrange
    file_content = json.dumps({"foo": "bar"})
    with patch("pathlib.Path.exists", return_value=True), \
         patch("pathlib.Path.open", mock_open(read_data=file_content)):
    
        # Act
        result = parse_input_data(None, "dummy_file.json")

        # Assert
        assert result == {"foo": "bar"}, "Expected parsed JSON from file."


def test_parse_input_data_file_not_found():
    """
    Should raise HTTPException(404) if the file does not exist.
    """
    # Arrange
    with patch("pathlib.Path.exists", return_value=False):
        with pytest.raises(HTTPException) as exc_info:
    
            # Act
            parse_input_data(None, "non_existent.json")

        # Assert
        assert exc_info.value.status_code == 404
        assert "not found" in str(exc_info.value.detail).lower()


def test_parse_input_data_file_invalid_json():
    """
    Should raise HTTPException(422) if the JSON content is invalid.
    """
    # Arrange
    invalid_json_content = "{invalid-json}"
    with patch("pathlib.Path.exists", return_value=True), \
         patch("pathlib.Path.open", mock_open(read_data=invalid_json_content)):
        with pytest.raises(HTTPException) as exc_info:
    
            # Act
            parse_input_data(None, "invalid_file.json")

        # Assert
        assert exc_info.value.status_code == 422
        assert "Failed to parse JSON" in exc_info.value.detail


def test_parse_input_data_no_inline_no_file():
    """
    Should raise HTTPException(422) if neither inline data nor file name is provided.
    """
    # Arrange
    with pytest.raises(HTTPException) as exc_info:
    
        # Act
        parse_input_data(None, None)

    # Assert
    assert exc_info.value.status_code == 422
    assert "You must provide either data or file" in exc_info.value.detail
    

def test_parse_input_data_file_path_construction():
    """
    Ensures that the file path is constructed relative to FILES_DIR.
    """
    file_name = "example.json"
    expected_path = FILES_DIR / file_name

    # This function replaces Path.open. 'self' will be the Path instance used,
    # allowing us to verify the path.
    def open_side_effect(self, mode="r", encoding=None):
        # Ensure that the 'self' instance matches the expected path.
        assert str(self) == str(expected_path), (
            f"Expected the path to be {expected_path}, got {self}"
        )
        # Return a file mock (mock_open) to simulate reading the file.
        return mock_open(read_data='{"test": 123}')()

    with patch("pathlib.Path.exists", return_value=True), \
         patch("pathlib.Path.open", new=open_side_effect):
        result = parse_input_data(None, file_name)
        assert result == {"test": 123}, "Should correctly parse the JSON content."
