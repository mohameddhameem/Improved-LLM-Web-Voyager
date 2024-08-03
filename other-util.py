from fastapi import FastAPI, Depends, HTTPException, Form
from fastapi.security import OAuth2AuthorizationCodeBearer
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import RedirectResponse
from msal import PublicClientApplication
import uvicorn
import secrets
import base64
import hashlib

app = FastAPI()

# Add session middleware
app.add_middleware(SessionMiddleware, secret_key="your-secret-key")

# Azure AD configuration
CLIENT_ID = "your-azure-client-id"
TENANT_ID = "your-azure-tenant-id"
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPE = ["User.Read"]  # Adjust scopes as needed
REDIRECT_URI = "http://localhost:8000/token"

# OAuth2 scheme for Swagger UI
oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl=f"{AUTHORITY}/oauth2/v2.0/authorize",
    tokenUrl=f"{AUTHORITY}/oauth2/v2.0/token",
    scopes={"User.Read": "Read user profile"}
)

# MSAL configuration
msal_app = PublicClientApplication(
    CLIENT_ID,
    authority=AUTHORITY
)

def generate_code_verifier():
    return secrets.token_urlsafe(32)

def generate_code_challenge(code_verifier):
    code_challenge = hashlib.sha256(code_verifier.encode('utf-8')).digest()
    return base64.urlsafe_b64encode(code_challenge).decode('utf-8').replace('=', '')

@app.get("/login")
async def login(request):
    code_verifier = generate_code_verifier()
    code_challenge = generate_code_challenge(code_verifier)
    request.session['code_verifier'] = code_verifier
    
    auth_url = msal_app.get_authorization_request_url(
        SCOPE,
        redirect_uri=REDIRECT_URI,
        code_challenge=code_challenge,
        code_challenge_method="S256"
    )
    return RedirectResponse(auth_url)

@app.get("/token")
async def get_token(request, code: str):
    code_verifier = request.session.get('code_verifier')
    if not code_verifier:
        raise HTTPException(status_code=400, detail="No code verifier found")
    
    result = msal_app.acquire_token_by_authorization_code(
        code,
        SCOPE,
        redirect_uri=REDIRECT_URI,
        code_verifier=code_verifier
    )
    if "access_token" in result:
        return {"access_token": result["access_token"]}
    raise HTTPException(status_code=401, detail="Authentication failed")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return token

@app.get("/protected")
async def protected_route(current_user: str = Depends(get_current_user)):
    return {"message": "This is a protected route", "user": current_user}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)