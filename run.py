#!/usr/bin/env python3
"""Start the transcription server."""
import uvicorn

if __name__ == "__main__":
    print("Starting Radiology Transcription Tool...")
    print("Open http://localhost:8000 in your browser")
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
