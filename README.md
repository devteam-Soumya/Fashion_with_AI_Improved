# Fashion_with_AI_Improved

# 👗 Fashion_with_AI_Improved

> **Virtual Try-On Pipeline** — 4-Module Agentic AI System  
> Powered by **CatVTON** · **Flux Kontext** · **Gemini Vision** · **FastAPI**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com)
[![GCP Cloud Run](https://img.shields.io/badge/Deploy-Cloud%20Run-orange.svg)](https://cloud.google.com/run)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## 📋 Table of Contents

- [What It Does](#-what-it-does)
- [Pipeline Architecture](#-pipeline-architecture)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Local Setup](#-local-setup)
- [Environment Variables](#-environment-variables)
- [API Endpoints](#-api-endpoints)
- [GCP Cloud Run Deployment](#-gcp-cloud-run-deployment)
- [Cost Estimate](#-cost-estimate)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 What It Does

Upload three images — a **user photo**, a **garment flat-lay**, and an **actress wearing the garment** — and the pipeline automatically:

1. **Classifies** the garment (midi dress vs full-length gown)
2. **Applies** the garment onto the user via CatVTON
3. **Extends** the garment to floor-length if it's a gown (Flux Kontext)
4. **Quality-checks** and auto-fixes any artefacts (Gemini Vision)

---

## 🏗 Pipeline Architecture

```
User Photo ──┐
             │
Flat-lay ────┼──► M0 Garment Inspector ──► M1 CatVTON Try-On ──► M2 Extension ──► M3 QC Fix ──► Result
             │    (Gemini Vision)           (fal-ai/cat-vton)     (Flux Kontext)    (Gemini)
Actress ─────┘
```

### Module Breakdown

| Module | Name | Model Used | Purpose |
|--------|------|-----------|---------|
| M0 | GarmentInspectorAgent | Gemini 2.0 Flash | Classifies garment type and length |
| M1 | TryOnModule | fal-ai/cat-vton | Applies garment onto user |
| M2 | ExtensionModule | fal-ai/flux-kontext/dev | Extends gown to floor-length |
| M3 | AutoFixQCAgent | Gemini 2.0 Flash + Flux Fill | Detects and fixes artefacts |

### Case A — Midi / Short Dress
```
M0: is_full_length = False
M1: Uses ACTRESS image as garment input
M2: SKIPPED
M3: QC fixes seam_bleed, color_mismatch, extra_outerwear
```

### Case B — Full-Length Gown / Maxi
```
M0: is_full_length = True
M1: Uses FLAT-LAY as garment input
M2: Flux Kontext extends to floor-length
M3: QC fixes artefacts after extension
```

---

## 📁 Project Structure

```
Fashion_with_AI_Improved/
├── ai_agent_pipeline.py      # Main FastAPI app — all 4 modules
├── fal_ai_backend.py         # fal.ai helper utilities
├── requirements.txt          # Python dependencies
├── Dockerfile                # Container config for Cloud Run
├── outputs/                  # Generated try-on result images
├── .gitignore
├── LICENSE
└── README.md
```

---

## ✅ Prerequisites

Before running locally or deploying, make sure you have:

| Tool | Version | Link |
|------|---------|------|
| Python | 3.10+ | https://python.org |
| Docker Desktop | Latest | https://docker.com |
| gcloud CLI | Latest | https://cloud.google.com/sdk |
| fal.ai account | — | https://fal.ai |
| Google Gemini API key | — | https://makersuite.google.com |

---

## 💻 Local Setup

### Step 1 — Clone the Repository

```bash
git clone https://github.com/devteam-Soumya/Fashion_with_AI_Improved.git
cd Fashion_with_AI_Improved
```

### Step 2 — Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Create `.env` File

```bash
# Create a .env file in the root folder
touch .env
```

Add your keys to `.env`:

```env
FAL_API_KEY=your_fal_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
OUTPUT_DIR=./outputs
PORT=8000
LOG_LEVEL=info
```

### Step 5 — Run the Server

```bash
uvicorn ai_agent_pipeline:app --host 0.0.0.0 --port 8000 --reload
```

### Step 6 — Open Swagger UI

```
http://localhost:8000/docs
```

---

## 🔧 Environment Variables

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `FAL_API_KEY` | — | ✅ Yes | Your fal.ai API key |
| `GEMINI_API_KEY` | — | ✅ Yes | Your Google Gemini API key |
| `OUTPUT_DIR` | `./outputs` | No | Directory for result images |
| `PORT` | `8000` | No | Server port (use `8080` on Cloud Run) |
| `LOG_LEVEL` | `info` | No | Uvicorn log level |
| `EXTEND_RATIO` | `0.85` | No | Canvas extension ratio for gowns |
| `CATVTON_STEPS` | `30` | No | CatVTON inference steps |
| `KONTEXT_STEPS` | `28` | No | Flux Kontext inference steps |
| `REF_MAX_SIDE` | `1024` | No | Max image dimension for references |

---

## 🌐 API Endpoints

### `GET /health`
Check if the server and all AI services are ready.

**Response:**
```json
{
  "ok": true,
  "ready": {
    "fal": true,
    "gemini": true,
    "rembg": true
  }
}
```

---

### `POST /v1/tryon/actress-garment-to-user` ⭐ Recommended

Upload all three images for the best quality result.

**Form Data:**
| Field | Type | Description |
|-------|------|-------------|
| `actress_image` | File | Model wearing the garment |
| `garment_image` | File | Product flat-lay photo |
| `user_image` | File | Target person to dress |
| `style_hint` | String | Optional e.g. `camel midi wrap dress` |
| `skip_m3` | Int | `1` = skip QC (faster), `0` = full pipeline |

**Example cURL:**
```bash
curl -X POST https://YOUR_URL/v1/tryon/actress-garment-to-user \
  -F "actress_image=@actress.jpg" \
  -F "garment_image=@flatlay.jpg" \
  -F "user_image=@user.jpg" \
  -F "style_hint=camel midi wrap dress" \
  -F "skip_m3=0"
```

**Response:**
```json
{
  "success": true,
  "output_view_urls": ["https://YOUR_URL/view/result_abc123.jpg"],
  "output_download_urls": ["https://YOUR_URL/download/result_abc123.jpg"],
  "pipeline": {
    "m0_garment": { "is_full_length": false, "garment_type": "camel midi wrap dress" },
    "m1_tryon":   { "latency_s": 45.2 },
    "m2_extension": { "extended": false },
    "m3_autofix": { "strategy": "none", "has_issues": false }
  }
}
```

---

### `POST /v1/tryon/garment-to-user`

Flat-lay + user only (no actress reference).

```bash
curl -X POST https://YOUR_URL/v1/tryon/garment-to-user \
  -F "garment_image=@flatlay.jpg" \
  -F "user_image=@user.jpg"
```

---

### `POST /v1/tryon/actress-to-user`

Actress reference + user only (no flat-lay).

```bash
curl -X POST https://YOUR_URL/v1/tryon/actress-to-user \
  -F "actress_image=@actress.jpg" \
  -F "user_image=@user.jpg"
```

---

### `GET /view/{filename}`

Open the result image in a styled browser viewer.

### `GET /download/{filename}`

Download the result image as a JPEG file.

---

## ☁️ GCP Cloud Run Deployment

### Step 1 — Install and Authenticate gcloud

```bash
# Authenticate
gcloud auth login

# Set your project
gcloud config set project aime-471607

# Verify
gcloud config get-value project
```

### Step 2 — Enable Required GCP APIs

```bash
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com \
  containerregistry.googleapis.com \
  artifactregistry.googleapis.com \
  --project=aime-471607
```

### Step 3 — Store API Keys in Secret Manager

```bash
# Create FAL_API_KEY secret
gcloud secrets create FAL_API_KEY \
  --replication-policy="automatic" \
  --project=aime-471607

echo -n "YOUR_FAL_API_KEY" | \
  gcloud secrets versions add FAL_API_KEY --data-file=-

# Create GEMINI_API_KEY secret
gcloud secrets create GEMINI_API_KEY \
  --replication-policy="automatic" \
  --project=aime-471607

echo -n "YOUR_GEMINI_API_KEY" | \
  gcloud secrets versions add GEMINI_API_KEY --data-file=-
```

### Step 4 — Grant Secret Access to Cloud Run

```bash
# Get your project number (replace with yours)
PROJECT_NUMBER=927331324143

gcloud secrets add-iam-policy-binding FAL_API_KEY \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor" \
  --project=aime-471607

gcloud secrets add-iam-policy-binding GEMINI_API_KEY \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor" \
  --project=aime-471607
```

### Step 5 — Build Docker Image

```bash
# Navigate to project folder
cd Fashion_with_AI_Improved

# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/aime-471607/fashion-tryon .
```

> Build takes 5–10 minutes on first run.

### Step 6 — Deploy to Cloud Run

```bash
gcloud run deploy fashion-tryon \
  --image gcr.io/aime-471607/fashion-tryon \
  --platform managed \
  --region us-central1 \
  --port 8080 \
  --memory 2Gi \
  --cpu 1 \
  --timeout 300 \
  --concurrency 2 \
  --min-instances 0 \
  --max-instances 3 \
  --set-secrets="FAL_API_KEY=FAL_API_KEY:latest,GEMINI_API_KEY=GEMINI_API_KEY:latest" \
  --set-env-vars="OUTPUT_DIR=/tmp/outputs" \
  --project=aime-471607 \
  --allow-unauthenticated
```

### Step 7 — Get Your Live URL

```bash
gcloud run services describe fashion-tryon \
  --region us-central1 \
  --project=aime-471607 \
  --format="value(status.url)"
```

### Step 8 — Verify Deployment

```bash
# Store URL in variable
SERVICE_URL=$(gcloud run services describe fashion-tryon \
  --region us-central1 \
  --project=aime-471607 \
  --format="value(status.url)")

# Test health
curl $SERVICE_URL/health

# Open Swagger docs in browser
echo "Swagger UI: $SERVICE_URL/docs"
```

---

### 🔄 Redeployment (After Code Changes)

```bash
# Step 1: Rebuild image
gcloud builds submit --tag gcr.io/aime-471607/fashion-tryon .

# Step 2: Redeploy
gcloud run deploy fashion-tryon \
  --image gcr.io/aime-471607/fashion-tryon \
  --region us-central1 \
  --project=aime-471607
```

---

### 📊 View Logs

```bash
# Live streaming logs
gcloud run services logs tail fashion-tryon \
  --region us-central1 \
  --project=aime-471607

# Recent 50 log lines
gcloud logging read \
  "resource.type=cloud_run_revision AND resource.labels.service_name=fashion-tryon" \
  --limit=50 \
  --project=aime-471607 \
  --format="value(textPayload)"
```

---

## 💰 Cost Estimate

### Recommended Settings (min-instances=0)

| Usage | Cloud Run | fal.ai APIs | Total/Month |
|-------|-----------|-------------|-------------|
| 0 requests (idle) | **$0.00** | $0.00 | **$0.00** |
| 50 requests | $0.00 | ~$5.00 | **~$5/month** |
| 100 requests | ~$0.50 | ~$12.00 | **~$12/month** |
| 500 requests | ~$2.50 | ~$60.00 | **~$62/month** |

> ⚠️ **Warning:** Using `--min-instances 1` costs ~$110/month even with zero requests. Use `--min-instances 0` for cost savings.

### Set a Billing Alert

```
https://console.cloud.google.com/billing/budgets?project=aime-471607
```

Recommended: Set alert at **$20/month**.

---

## 🐛 Troubleshooting

### Container failed to start on port 8080
```bash
# Fix: Make sure Dockerfile uses port 8080
# CMD ["uvicorn", "ai_agent_pipeline:app", "--host", "0.0.0.0", "--port", "8080"]
```

### FAL_KEY missing in health check
```bash
# Fix: Re-run IAM binding
gcloud secrets add-iam-policy-binding FAL_API_KEY \
  --member="serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

### Permission denied on /outputs
```bash
# Fix: Set OUTPUT_DIR to /tmp/outputs
--set-env-vars="OUTPUT_DIR=/tmp/outputs"
```

### Out of memory error
```bash
# Fix: Increase memory to 3Gi
gcloud run services update fashion-tryon \
  --memory 3Gi \
  --region us-central1
```

### gcloud CLI crash (TypeError)
```bash
# Fix: Use full redeploy instead of update
gcloud run deploy fashion-tryon --image gcr.io/aime-471607/fashion-tryon ...
```

---

## 📝 Dockerfile Reference

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgl1-mesa-glx libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p /tmp/outputs

EXPOSE 8080

CMD ["uvicorn", "ai_agent_pipeline:app", "--host", "0.0.0.0", "--port", "8080"]
```

---

## 👥 Contributors

- **devteam-Soumya** — Lead Developer

---

## 📄 License

This project is licensed under the **Apache 2.0 License** — see [LICENSE](LICENSE) for details.
