FROM python:3.11-slim

# OpenCV/MediaPipe 런타임 의존성
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 libxext6 libsm6 libxrender1 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 파이썬 캐시 줄이기
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 이게 핵심: /app(프로젝트 루트)을 import 경로에 올려서
# 'tilt_worker.src.api.main' 같은 패키지 import가 동작하게 함
ENV PYTHONPATH=/app

# 의존성 먼저 (캐시 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 복사
COPY . .

# 비루트 권장
RUN useradd -m appuser
USER appuser

EXPOSE 8000

# (선택) /health 엔드포인트가 있다면 헬스체크
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
 CMD curl -fsS http://localhost:8000/health || exit 1

# ⚠️ 모듈 경로는 현재 구조 기준으로 고정
CMD ["uvicorn", "tilt_worker.src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
