# ---- Base image ----
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# (Optional) If you use rembg and want model cache in a writable place:
# ENV U2NET_HOME=/tmp/.u2net

WORKDIR /app

# ---- System deps ----
# libgl1 + libglib2.0-0 are commonly needed by opencv-python on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

# ---- Install Python deps ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Copy app code ----
COPY . .

# Create outputs folder (your code also creates it, but safe to pre-create)
RUN mkdir -p /app/outputs

# Railway/AWS sets PORT; default 8000
ENV PORT=8000
EXPOSE 8000

# ---- Start server ----
# IMPORTANT: run the correct module path: backend_full_gown_agent:app
CMD ["sh", "-c", "uvicorn backend_full_gown_agent:app --host 0.0.0.0 --port ${PORT} --log-level info"]
