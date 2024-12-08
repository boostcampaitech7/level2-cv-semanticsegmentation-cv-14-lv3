# 베이스 이미지 선택
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 시스템 패키지 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    libfontconfig1 \
    libgl1-mesa-glx

# requirements.txt 파일 복사 (프로젝트 루트에 있어야 함)
COPY requirements.txt .

# Install required packeages
RUN pip install --no-cache-dir -r requirements.txt

# 프로젝트 파일 복사
COPY . .