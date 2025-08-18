FROM python:3.11-slim

# (선택) pip 업그레이드로 설치 이슈 완화
RUN pip install --no-cache-dir --upgrade pip

# OpenCV/MediaPipe 런타임 의존성
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 libxext6 libsm6 libxrender1 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    OPENCV_LOG_LEVEL=ERROR 

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 비루트 OK
RUN useradd -m appuser
USER appuser

EXPOSE 9500

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
 CMD curl -fsS http://localhost:9500/health || exit 1

CMD ["uvicorn", "tilt_worker.src.api.main:app", "--host", "0.0.0.0", "--port", "9500", "--workers", "1"]