# app\common_test.py
import pytest
from unittest.mock import patch
from app.common import load_env_vars

@pytest.fixture(autouse=True)
def mock_env_files():
    """
    Automatically mock the loading of .env and .env.local for all tests.
    """
    with patch("dotenv.load_dotenv") as mock_load_dotenv, \
         patch("os.getenv") as mock_getenv:
        # Mock behavior of load_dotenv
        mock_load_dotenv.return_value = None

        # Mock environment variables returned by os.getenv
        mock_getenv.side_effect = lambda key, default=None: {
            "MODE": "test",  # Replace with "dev" or "prod" if needed
            "SECRET_KEY": "mocked-secret-key"
        }.get(key, default)

        yield

def test_load_env_vars():
    """
    Test that load_env_vars returns the mocked environment variables.
    """
    envs = load_env_vars()
    assert envs["secret_key"] == "mocked-secret-key"
    assert envs["mode"] == "test"
