# Radiology Transcription Tool

A minimal tool to compare different speech-to-text backends for radiology dictation.

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the server
python run.py
```

Then open http://localhost:8000 in your browser.

## Features

- Record audio directly in browser or upload audio files
- Compare multiple Whisper model sizes (tiny, base, small, medium, large)
- See transcription results side-by-side with timing info

## Notes

- First transcription with each model will be slower (model loading)
- Larger models are more accurate but slower and need more RAM/GPU
- For best accuracy with medical terms, try `whisper-medium` or `whisper-large`
