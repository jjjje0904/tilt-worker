# tilt_worker/src/api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import os

from tilt_worker.src.pipelines.tilt_pipeline import process_video

app = FastAPI(title="Tilt Worker API")
BASE_DIR = Path(os.getenv("UPLOADS_DIR", "/var/pks-uploads/videos")).resolve()

class AnalyzeReq(BaseModel):
    video_id: str 
    threshold: float = 3.0

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze")
def analyze(req: AnalyzeReq):
    # 1) 베이스/상대경로 결합
    candidate = (BASE_DIR / req.video_id).resolve()

    # 2) 디렉토리 탈출 방지
    if not str(candidate).startswith(str(BASE_DIR)):
        raise HTTPException(400, "invalid video_id (path traversal)")

    # 3) 존재 확인
    if not candidate.exists():
        raise HTTPException(404, f"file not found: {candidate}")

    # 4) 처리
    try:
        result = process_video(str(candidate), threshold_deg=req.threshold)
        return {"status": "ok", "result": result}
    except Exception as e:
        raise HTTPException(500, f"analyze failed: {e}")
