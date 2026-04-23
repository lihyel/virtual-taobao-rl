FROM --platform=linux/amd64 python:3.7-bullseye

# 1. 시스템 의존성 보강 (OpenCV, Gym 렌더링용 필수 라이브러리)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libgl1-mesa-glx \
    libosmesa6-dev \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 2. pip 및 기초 도구 업데이트
RUN pip install --upgrade pip setuptools==57.5.0 wheel

# 3. 의존성 설치 (Numpy를 먼저 설치하여 버전 충돌 방지)
COPY requirements.txt .
RUN pip install --no-cache-dir numpy==1.18.5
RUN pip install --no-cache-dir -r requirements.txt

# 4. TensorFlow 1.15 설치 (M1 에뮬레이션에서 가장 안정적인 방식)
RUN pip install --no-cache-dir tensorflow==1.15.0

# 5. 소스 코드 복사 및 패키지 설치
COPY . .
# 공식 깃허브의 setup.py를 통해 시뮬레이터 설치
RUN pip install -e .