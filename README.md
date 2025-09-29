# ClassVision

## Local POC
### Setup
1. `python3 -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. `cp .env.example .env`
### Run API
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000