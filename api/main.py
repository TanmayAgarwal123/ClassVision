from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from pathlib import Path
import json, os
from loguru import logger


DATA_DIR = Path("data"); DATA_DIR.mkdir(parents=True, exist_ok=True)


app = FastAPI(title="ClassVision API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)


class HeadPose(BaseModel):
    yaw: float = 0.0
    pitch: float = 0.0

class Signal(BaseModel):
    ts: float = Field(..., description="Unix timestamp seconds")
    zone_id: str = "all"
    head_pose_yaw: Optional[float] = None
    head_pose_pitch: Optional[float] = None
    gaze_to_board_prob: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    speaking_prob: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    hand_raise_prob: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    device_use_prob: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    motion_rate: Optional[float] = None


class BatchIn(BaseModel):
    session_id: str
    batch: List[Signal]


@app.get("/healthz")
def healthz():
    return {"ok": True, "time": datetime.utcnow().isoformat()}

@app.post("/v1/sessions/{session_id}/signals")
def ingest_signals(session_id: str, payload: BatchIn):
    if session_id != payload.session_id:
        raise HTTPException(status_code=400, detail="session_id mismatch")
    if len(payload.batch) == 0:
        raise HTTPException(status_code=400, detail="empty batch")

    out_path = DATA_DIR / f"signals_{session_id}.ndjson"
    with out_path.open("a", encoding="utf-8") as f:
        for s in payload.batch:
            rec = s.model_dump()
            rec["session_id"] = session_id
            f.write(json.dumps(rec) + "")
    logger.info(f"Appended {len(payload.batch)} signals → {out_path}")
    return {"written": len(payload.batch)}


@app.get("/v1/sessions/{session_id}/preview")
def preview(session_id: str, limit: int = 5):
    """Read last few records for sanity check"""
    path = DATA_DIR / f"signals_{session_id}.ndjson"
    if not path.exists():
        raise HTTPException(status_code=404, detail="no data yet")
    lines = path.read_text().strip().splitlines()[-limit:]
    return {"lines": [json.loads(x) for x in lines]}