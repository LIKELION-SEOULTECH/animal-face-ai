# 1. 베이스 이미지
FROM python:3.9-slim

# 2. 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. 파이썬 라이브러리 설치

RUN pip install --no-cache-dir \
    numpy \
    onnxruntime \
    opencv-python-headless \
    fastapi \
    uvicorn \
    python-multipart

# 4. 파일 복사
COPY model/fastvit.onnx .
COPY model/app.py .
COPY model/fastvit.onnx.data .

# 5. 포트 설정 및 실행
EXPOSE 8000
CMD ["python", "app.py"]