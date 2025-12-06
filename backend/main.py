import os
import secrets
from pathlib import Path
from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Radiology Transcription Tool")
security = HTTPBasic()

MAINTENANCE_MODE = os.getenv("MAINTENANCE_MODE", "false").lower() == "true"
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
AUTH_USERNAME = os.getenv("AUTH_USERNAME", "")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD", "")


def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, AUTH_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, AUTH_PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


static_dir = Path(__file__).parent.parent / "frontend"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root(username: str = Depends(verify_credentials)):
    if MAINTENANCE_MODE:
        return {"message": "Coming soon - currently in private beta"}
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Frontend not found"}


@app.get("/api/deepgram-key")
async def get_deepgram_key(username: str = Depends(verify_credentials)):
    """Return Deepgram API key for client-side streaming."""
    if not DEEPGRAM_API_KEY:
        raise HTTPException(status_code=500, detail="Deepgram API key not configured")
    return {"key": DEEPGRAM_API_KEY}
