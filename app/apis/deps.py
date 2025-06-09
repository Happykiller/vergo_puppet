# app/apis/deps.py
import jwt
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer

from app.common import load_env_vars
from app.services.logger import logger

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_access_token(token: str = Depends(oauth2_scheme)):
    envs = load_env_vars()
    try:
        payload = jwt.decode(token, envs["secret_key"], algorithms=["HS256"])
        logger.debug(f"payload: {payload}")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    return payload
