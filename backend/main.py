import tempfile
import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
import httpx

load_dotenv()  # Load from .env if present (local dev only)

app = FastAPI(title="Radiology Transcription Tool")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")


@app.post("/api/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    backend: str = Form(default="openai-whisper"),
):
    suffix = Path(audio.filename).suffix or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        if backend.startswith("openai"):
            result = await transcribe_openai(tmp_path, backend)
        elif backend == "deepgram":
            result = await transcribe_deepgram(tmp_path)
        else:
            result = {"error": f"Unknown backend: {backend}"}
        return result
    finally:
        os.unlink(tmp_path)


async def transcribe_openai(audio_path: str, backend: str) -> dict:
    """Transcribe using OpenAI Whisper API."""
    if not OPENAI_API_KEY:
        return {"error": "OPENAI_API_KEY not set", "backend": backend}

    model = "whisper-1"
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            with open(audio_path, "rb") as f:
                files = {"file": (Path(audio_path).name, f, "audio/webm")}
                data = {"model": model}
                
                # Add prompt with medical terminology hints
                data["prompt"] = (
                    "Radiology dictation. Medical terms: bilateral, inferior, superior, "
                    "anterior, posterior, medial, lateral, proximal, distal, "
                    "carcinoma, metastasis, lesion, nodule, opacity, effusion, "
                    "atelectasis, consolidation, pneumothorax, cardiomegaly"
                )
                
                resp = await client.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                    files=files,
                    data=data,
                )
                
            if resp.status_code != 200:
                return {"error": f"OpenAI API error: {resp.text}", "backend": backend}
                
            result = resp.json()
            return {
                "backend": backend,
                "text": result.get("text", "").strip(),
            }
    except Exception as e:
        return {"error": str(e), "backend": backend}


async def transcribe_deepgram(audio_path: str) -> dict:
    """Transcribe using Deepgram API."""
    if not DEEPGRAM_API_KEY:
        return {"error": "DEEPGRAM_API_KEY not set. Get one at https://console.deepgram.com", "backend": "deepgram"}

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            with open(audio_path, "rb") as f:
                audio_data = f.read()
                
            resp = await client.post(
                "https://api.deepgram.com/v1/listen?model=nova-2-medical&smart_format=true&punctuate=true",
                headers={
                    "Authorization": f"Token {DEEPGRAM_API_KEY}",
                    "Content-Type": "audio/webm",
                },
                content=audio_data,
            )
                
            if resp.status_code != 200:
                return {"error": f"Deepgram API error: {resp.text}", "backend": "deepgram"}
                
            result = resp.json()
            transcript = result.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript", "")
            return {
                "backend": "deepgram",
                "text": transcript.strip(),
            }
    except Exception as e:
        return {"error": str(e), "backend": "deepgram"}


static_dir = Path(__file__).parent.parent / "frontend"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root():
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Frontend not found"}


@app.get("/api/backends")
async def list_backends():
    backends = [
        {
            "id": "openai-whisper",
            "name": "OpenAI Whisper",
            "description": "Cloud API, good accuracy",
            "available": bool(OPENAI_API_KEY),
        },
        {
            "id": "deepgram",
            "name": "Deepgram Nova-2 Medical",
            "description": "Medical-specific model",
            "available": bool(DEEPGRAM_API_KEY),
        },
    ]
    return {"backends": backends}
