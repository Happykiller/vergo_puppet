# app\generate_token.py
import os
import jwt
import datetime
from pathlib import Path
from dotenv import load_dotenv

# Build the absolute path to the .env files
env_path = Path(__file__).resolve().parent.parent / ".env"
env_local_path = Path(__file__).resolve().parent.parent / ".env.local"

# Load environment variables
load_dotenv(env_path)
load_dotenv(env_local_path, override=True)

# Example of accessing an environment variable
SECRET_KEY = os.getenv("SECRET_KEY")

if not SECRET_KEY:
    raise EnvironmentError("SECRET_KEY is missing in the environment variables.")

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
    print(f"SECRET_KEY: {SECRET_KEY}")
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    return token

if __name__ == "__main__":
    # Example usage with user ID and token expiration duration in minutes
    user_id = input("Enter user ID: ")
    expiration_minutes = int(input("Token expiration in minutes (default 525,600): ") or 525600)
    token = create_token(user_id, expiration_minutes)
    print(f"Generated Token: {token}")
