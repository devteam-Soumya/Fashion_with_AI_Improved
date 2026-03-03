FROM python:3.11-slim

WORKDIR /app

# FIX: libgl1-mesa-glx removed in Debian Trixie → replaced with libgl1
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p ./outputs

EXPOSE 10000

CMD ["python", "ai_agent_pipeline.py"]
