"""Auth endpoints: register, login (returns a JWT), and current-user info."""
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel

from app.auth import store
from app.auth.security import create_access_token, decode_token

router = APIRouter()


class Credentials(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    tenant_id: str


@router.post("/register", response_model=TokenResponse)
def register(creds: Credentials) -> TokenResponse:
    try:
        user = store.create_user(creds.email, creds.password)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    token = create_access_token(sub=user["email"], tenant_id=user["tenant_id"])
    return TokenResponse(access_token=token, tenant_id=user["tenant_id"])


@router.post("/login", response_model=TokenResponse)
def login(creds: Credentials) -> TokenResponse:
    user = store.authenticate(creds.email, creds.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(sub=user["email"], tenant_id=user["tenant_id"])
    return TokenResponse(access_token=token, tenant_id=user["tenant_id"])


@router.get("/me")
def me(authorization: str | None = Header(default=None)) -> dict:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    try:
        payload = decode_token(authorization.split(" ", 1)[1])
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    return {"email": payload["sub"], "tenant_id": payload["tenant_id"]}
