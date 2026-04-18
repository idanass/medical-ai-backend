import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from jose import JWTError, jwt
from passlib.context import CryptContext

# ── Config ───────────────────────────────────────────
SECRET_KEY   = os.getenv("SECRET_KEY", "cxr-diagnostica-secret-key-2026")
ALGORITHM    = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

VALID_API_KEYS = set(
    os.getenv("API_KEYS", "cxr-api-key-doctor-001,cxr-api-key-admin-002").split(",")
)

# ── Password hashing ─────────────────────────────────
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ── Schemes ──────────────────────────────────────────
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# ── Fake user DB (replace with real DB later) ────────
USERS_DB = {
    "admin": {
        "username": "admin",
        "hashed_password": pwd_context.hash("admin123"),
        "role": "admin"
    },
    "doctor": {
        "username": "doctor",
        "hashed_password": pwd_context.hash("doctor123"),
        "role": "doctor"
    }
}


# ── JWT utils ────────────────────────────────────────
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )


def authenticate_user(username: str, password: str):
    user = USERS_DB.get(username)
    if not user:
        return None
    if not pwd_context.verify(password, user["hashed_password"]):
        return None
    return user


# ── Dependencies ─────────────────────────────────────
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
    api_key: str = Security(api_key_header)
):
    """
    Accepts either:
    - Bearer JWT token (Authorization: Bearer <token>)
    - API Key header  (X-API-Key: <key>)
    """
    # Try API Key first
    if api_key and api_key in VALID_API_KEYS:
        return {"username": "api_client", "role": "doctor", "auth": "api_key"}

    # Try JWT token
    if credentials:
        payload = verify_token(credentials.credentials)
        username = payload.get("sub")
        if username and username in USERS_DB:
            user = USERS_DB[username]
            return {"username": username, "role": user["role"], "auth": "jwt"}

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required — provide Bearer token or X-API-Key",
        headers={"WWW-Authenticate": "Bearer"},
    )