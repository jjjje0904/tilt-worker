from fastapi import FastAPI
from pydantic import BaseModel
from tilt_worker.src.pipelines.tilt_pipeline import process_video

app = FastAPI(title="Tilt Worker API")

class VideoRequest(BaseModel):
    path: str
    threshold: float = 3.0

@app.post("/analyze-path")
async def analyze_path(req: VideoRequest):
    try:
        result = process_video(req.path, threshold_deg=req.threshold)
        return {"status": "ok", "result": result}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
