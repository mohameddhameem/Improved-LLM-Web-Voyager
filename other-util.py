from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.security import OAuth2AuthorizationCodeBearer
from fastapi.responses import RedirectResponse
from msal import ConfidentialClientApplication
import uvicorn

app = FastAPI()

# Azure AD configuration
CLIENT_ID = "your-azure-client-id"
CLIENT_SECRET = "your-azure-client-secret"
TENANT_ID = "your-azure-tenant-id"
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPE = ["User.Read"]  # Adjust scopes as needed
REDIRECT_URI = "http://localhost:8000/docs/oauth2-redirect"  # Swagger UI's redirect URI

# OAuth2 scheme for Swagger UI
oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl=f"{AUTHORITY}/oauth2/v2.0/authorize",
    tokenUrl=f"{AUTHORITY}/oauth2/v2.0/token",
    scopes={"User.Read": "Read user profile"}
)

# MSAL configuration
msal_app = ConfidentialClientApplication(
    CLIENT_ID,
    authority=AUTHORITY,
    client_credential=CLIENT_SECRET
)

@app.get("/login")
async def login():
    auth_url = msal_app.get_authorization_request_url(
        SCOPE,
        redirect_uri=REDIRECT_URI
    )
    return RedirectResponse(auth_url)

@app.get("/token")
async def get_token(request: Request):
    code = request.query_params.get("code")
    if not code:
        raise HTTPException(status_code=400, detail="No authorization code provided")
    
    result = msal_app.acquire_token_by_authorization_code(
        code,
        SCOPE,
        redirect_uri=REDIRECT_URI
    )
    if "access_token" in result:
        return {"access_token": result["access_token"]}
    raise HTTPException(status_code=401, detail="Authentication failed")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    # Here you would typically validate the token
    # For demonstration, we're just checking if it exists
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return token

@app.get("/protected")
async def protected_route(current_user: str = Depends(get_current_user)):
    return {"message": "This is a protected route", "user": current_user}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)