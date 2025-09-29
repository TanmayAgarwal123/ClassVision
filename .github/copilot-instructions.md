<!-- .github/copilot-instructions.md -->
# How to be productive in the ClassVision repo

This file gives focused, actionable guidance for code-generating AI agents working on this repository.

## Big picture (what to look for)
- The repo contains two primary components in the same workspace:
  - `api/` — intended FastAPI service (entrypoint: `api/main.py`).
  - `edge_agent/` — local/edge process that performs camera/video processing (`edge_agent/edge_agent.py`).
- Dependencies in `requirements.txt` show intended architecture:
  - FastAPI + `uvicorn` for the server
  - `opencv-python`, `mediapipe`, `numpy` for local image processing
  - `requests` for HTTP client calls from the edge agent
  - `python-dotenv` + `loguru` for configuration and logging

Because `api/main.py` and `edge_agent/edge_agent.py` are present but currently empty, prefer small, explicit PRs that implement one endpoint or one edge behavior at a time.

## Typical data flow (inferred and what to check)
- Edge agent captures frames (OpenCV/MediaPipe), transforms them (NumPy arrays, landmarks), then communicates results to the API using HTTP (requests). Search the codebase for `requests` usage to find concrete endpoints.
- On the server side expect JSON body endpoints (FastAPI + Pydantic models). If you add endpoints, place them under `api/` and use Pydantic models for request/response shapes.

## Developer workflows (commands you can run)
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Run the API locally (standard FastAPI pattern):
  ```bash
  uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
  ```
- Run the edge agent locally:
  ```bash
  python edge_agent/edge_agent.py
  ```
- Environment variables: the repo uses `python-dotenv`. Look for `.env` usage; if none exists create `.env` with values like `API_URL=http://localhost:8000` and `LOG_LEVEL=INFO`.

## Project-specific conventions & patterns to preserve
- Logging: `loguru` is listed — prefer `from loguru import logger` and use structured messages (logger.info/debug).
- Image/frame transfer: because binary frames are heavy, prefer one of:
  - send JSON with compact landmark arrays (NumPy -> Python lists), or
  - send base64-encoded JPEG if full images are required (use OpenCV to encode)
  Document any choice near the endpoint so edge/server code match.
- API design: use Pydantic models for request/response and return explicit HTTP status codes. Place models next to `api/main.py` (e.g., `api/schemas.py`) to keep API surface discoverable.

## Integration points & external dependencies
- `requirements.txt` lists the key libraries: `fastapi`, `uvicorn[standard]`, `pydantic`, `python-dotenv`, `loguru`, `requests`, `opencv-python`, `mediapipe`, `numpy`.
- Expect heavy CPU usage for MediaPipe/OpenCV code — keep edge processing and API responsibilities separated.

## Safe edits and small tasks an agent can do immediately
- Implement a minimal FastAPI app in `api/main.py` that exposes a health endpoint (`GET /health`) and a placeholder `POST /ingest` that accepts JSON with a documented Pydantic model.
- Implement a minimal `edge_agent/edge_agent.py` script that reads `API_URL` from env, captures a single frame with OpenCV (or a mocked NumPy array), encodes it (base64 or JSON landmarks), and POSTs to `API_URL`.
- Add a brief `api/schemas.py` with Pydantic models used by the endpoints.

## Where to look first (key files)
- `requirements.txt` — authoritative list of libraries used.
- `api/main.py` — API entry (currently empty placeholder).
- `edge_agent/edge_agent.py` — edge process (currently empty placeholder).
- `README.md` — repository-level context (short at present).

## PR guidance for AI agents
- Keep changes minimal and self-contained.
- Update `requirements.txt` only if a new dependency is strictly required and pin the version in a follow-up if requested.
- When adding an API endpoint, add/modify `api/schemas.py` and include a small unit test or a manual run instruction in the file docstring.

## Final notes
- This repo is small and componentized — aim for explicit, testable steps (health endpoints, single-frame ingest) before adding full streaming or production optimizations.

Please review these instructions and tell me any missing details or preferences (naming, endpoint shapes, or examples) and I will update this guidance.
