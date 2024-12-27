# generate_token_test.py
from unittest.mock import patch
from app.generate_token import create_token

@patch("app.generate_token.load_env_vars")
def test_create_token(mock_load_env_vars):
    """
    Test JWT token creation with mocked environment variables.
    """
    # Create a mock Inversify instance
    mock_load_env_vars.return_value = {
        "secret_key": "mocked-secret-key",
        "mode": "test"
    }

    user_id = "test_user"
    token = create_token(user_id)
    assert token is not None
    print(f"Generated token: {token}")
