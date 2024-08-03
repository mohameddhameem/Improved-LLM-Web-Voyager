from fastapi import FastAPI, Depends, HTTPException, Form
from fastapi.security import OAuth2AuthorizationCodeBearer
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import RedirectResponse
from msal import ConfidentialClientApplication
import uvicorn

app = FastAPI()

# Add session middleware
app.add_middleware(SessionMiddleware, secret_key="your-secret-key")

# Azure AD configuration
CLIENT_ID = "your-azure-client-id"
CLIENT_SECRET = "your-azure-client-secret"
TENANT_ID = "your-azure-tenant-id"
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPE = ["User.Read"]  # Adjust scopes as needed

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
async def login(request):
    auth_url = msal_app.get_authorization_request_url(SCOPE)
    return RedirectResponse(auth_url)

@app.post("/token")
async def get_token(code: str = Form(...), client_id: str = Form(...)):
    if client_id != CLIENT_ID:
        raise HTTPException(status_code=400, detail="Invalid client_id")
    
    result = msal_app.acquire_token_by_authorization_code(code, SCOPE)
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
    
    
# pip install python-multipart