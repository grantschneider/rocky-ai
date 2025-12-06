import os
import secrets
from pathlib import Path
from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx

load_dotenv()

app = FastAPI(title="Radiology Transcription Tool")
security = HTTPBasic()

MAINTENANCE_MODE = os.getenv("MAINTENANCE_MODE", "false").lower() == "true"
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AUTH_USERNAME = os.getenv("AUTH_USERNAME", "")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD", "")

REPORT_PROMPT = """You are a radiology report formatting assistant. Convert the following dictated transcription into a properly structured radiology report.

FORMAT: Standard Radiology Report (ACR Guidelines)

Use this structure:
---
CLINICAL INDICATION:
[Extract the reason for the exam, patient symptoms, or clinical question]

TECHNIQUE:
[Extract imaging modality, contrast use, and any technical details mentioned]

COMPARISON:
[Extract any prior studies mentioned, or state "None available" if not mentioned]

FINDINGS:
[Organize findings by anatomical region/system. Use complete sentences. Include:
- Normal findings (e.g., "The heart is normal in size")
- Abnormal findings with descriptions (size, location, characteristics)
- Pertinent negatives]

IMPRESSION:
[Numbered list of key findings and conclusions, most significant first. Include:
1. Primary diagnosis or finding
2. Secondary findings
3. Recommendations if mentioned]
---

IMPORTANT:
- Preserve all medical terminology exactly as dictated
- If information for a section is not provided, write "Not specified"
- Keep findings factual and objective
- Use standard radiological language

TRANSCRIPTION TO FORMAT:
"""


class ReportRequest(BaseModel):
    transcript: str


class FeedbackRequest(BaseModel):
    rating: str
    comment: str
    transcript: str
    report: str


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


@app.get("/health")
async def health():
    """Health check endpoint (no auth required)."""
    return {"status": "ok"}


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


@app.post("/api/generate-report")
async def generate_report(request: ReportRequest, username: str = Depends(verify_credentials)):
    """Generate a formatted radiology report from transcription using LLM."""
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    if not request.transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript is empty")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "system", "content": REPORT_PROMPT},
                        {"role": "user", "content": request.transcript}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 2000,
                },
            )
            
            if resp.status_code != 200:
                raise HTTPException(status_code=500, detail=f"OpenAI API error: {resp.text}")
            
            result = resp.json()
            report = result["choices"][0]["message"]["content"]
            
            return {
                "report": report,
                "format": "Standard Radiology Report (ACR Guidelines)",
                "model": "gpt-4o"
            }
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest, username: str = Depends(verify_credentials)):
    """Store user feedback on generated reports."""
    import json
    from datetime import datetime
    
    feedback_dir = Path(__file__).parent.parent / "feedback"
    feedback_dir.mkdir(exist_ok=True)
    
    feedback_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "rating": request.rating,
        "comment": request.comment,
        "transcript": request.transcript[:500],  # Truncate for storage
        "report": request.report[:1000],  # Truncate for storage
    }
    
    feedback_file = feedback_dir / "feedback.jsonl"
    with open(feedback_file, "a") as f:
        f.write(json.dumps(feedback_entry) + "\n")
    
    return {"status": "ok"}
