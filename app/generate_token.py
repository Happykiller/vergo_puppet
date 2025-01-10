# app\generate_token.py
import jwt
import datetime
from app.common import load_env_vars
from app.services.logger import logger

# export PYTHONPATH=$(pwd):$PYTHONPATH
# python3 app/generate_token.py

def create_token(user_id: str, expiration_minutes: int = 30):
    """
    Creates a JWT token for a specific user.
    :param user_id: Identifier of the user for whom the token is generated
    :param expiration_minutes: Token validity duration in minutes
    :return: Signed JWT token
    """
    expiration = datetime.datetime.utcnow() + datetime.timedelta(minutes=expiration_minutes)
    payload = {
        "sub": user_id,
        "exp": expiration
    }
    envs = load_env_vars()
    logger.debug(f"Envs: {envs}")
    token = jwt.encode(payload, envs["secret_key"], algorithm="HS256")
    return token

if __name__ == "__main__":
    # Example usage with user ID and token expiration duration in minutes
    user_id = input("Enter user ID: ")
    expiration_minutes = int(input("Token expiration in minutes (default 525,600): ") or 525600)
    token = create_token(user_id, expiration_minutes)
    logger.debug(f"Generated Token: {token}")
