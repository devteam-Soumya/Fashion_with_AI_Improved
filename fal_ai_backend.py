# backend_fullgrown_tryon_and_remix_catvton_only.py
#
# Rewritten backend with **robust preprocessing** for model-worn "actress" references
# to improve Cat-VTON stability (reduce random coat-like outputs).
#
# Key improvements:
# - Preprocess actress/garment reference:
#     * optional background removal (rembg if installed)
#     * tight crop to non-background
#     * soften/remove face/hands region heuristically (reduces confusion)
#     * normalize to expected portrait size
# - Safer FAL upload (bytes only)
# - Cleaner response schema + consistent URLs
# - Fix uvicorn.run module name
#
# Install (example):
#   pip install fastapi uvicorn python-dotenv anyio requests pillow opencv-python numpy fal-client
# Optional (recommended for best results with worn references):
#   pip install rembg==2.0.61 onnxruntime==1.17.3
#
# ENV:
#   FAL_API_KEY=...
#   (optional) GEMINI_API_KEY=... (if you want prompt shaping for remix)
#
# Run:
#   python backend_fullgrown_tryon_and_remix_catvton_only.py
# or
#   uvicorn backend_fullgrown_tryon_and_remix_catvton_only:app --host 0.0.0.0 --port 8000

import os
import io
import uuid
import traceback
from typing import Optional, Tuple, List, Dict, Any

import anyio
import requests
import numpy as np
import cv2
from PIL import Image
import uvicorn

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from starlette.middleware.base import BaseHTTPMiddleware
from dotenv import load_dotenv

load_dotenv()

# ----------------------------
# CONFIG
# ----------------------------
PORT = int(os.getenv("PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "info").lower()

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FAL_API_KEY = os.getenv("FAL_API_KEY", os.getenv("FAL_KEY", "")).strip()
if FAL_API_KEY:
    os.environ["FAL_API_KEY"] = FAL_API_KEY
    os.environ["FAL_KEY"] = FAL_API_KEY

FAL_FLUX_EDIT_MODEL = os.getenv("FAL_FLUX_EDIT_MODEL", "fal-ai/flux-2/edit").strip()
FAL_CATVTON_MODEL = os.getenv("FAL_CATVTON_MODEL", "fal-ai/cat-vton").strip()

# Flux edit tuning
FLUX_IMAGE_SIZE = os.getenv("FLUX_IMAGE_SIZE", "portrait_4_3").strip()
FLUX_STEPS = int(os.getenv("FLUX_STEPS", "28"))
FLUX_GUIDANCE = float(os.getenv("FLUX_GUIDANCE", "2.5"))
FLUX_ENABLE_PROMPT_EXPANSION = os.getenv("FLUX_ENABLE_PROMPT_EXPANSION", "0").strip().lower() in (
    "1", "true", "yes", "on"
)

# CatVTON tuning
CATVTON_IMAGE_SIZE = os.getenv("CATVTON_IMAGE_SIZE", "portrait_4_3").strip()
CATVTON_STEPS = int(os.getenv("CATVTON_STEPS", "30"))
CATVTON_GUIDANCE = float(os.getenv("CATVTON_GUIDANCE", "2.5"))

# Preprocess knobs (important for actress-to-user)
ENABLE_REF_PREPROCESS = os.getenv("ENABLE_REF_PREPROCESS", "1").strip().lower() in ("1", "true", "yes", "on")
REF_BLUR_FACE_HANDS = os.getenv("REF_BLUR_FACE_HANDS", "1").strip().lower() in ("1", "true", "yes", "on")
REF_CROP_PAD = float(os.getenv("REF_CROP_PAD", "0.12"))  # padding around tight bbox
REF_MAX_SIDE = int(os.getenv("REF_MAX_SIDE", "1024"))    # normalize size for upload

# Gemini optional (prompt shaping)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview").strip()

# ----------------------------
# CLIENTS
# ----------------------------
try:
    import fal_client  # type: ignore
except Exception:
    fal_client = None  # type: ignore

# optional rembg
try:
    from rembg import remove as rembg_remove  # type: ignore
except Exception:
    rembg_remove = None  # type: ignore

genai_client = None
try:
    if GEMINI_API_KEY:
        from google import genai  # type: ignore
        genai_client = genai.Client(api_key=GEMINI_API_KEY)
except Exception:
    genai_client = None


# ----------------------------
# FASTAPI
# ----------------------------
app = FastAPI(title="Outfit Remix + Try-On (Cat-VTON + strong preprocessing for actress refs)")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request.state.request_id = uuid.uuid4().hex[:12]
        resp = await call_next(request)
        resp.headers["X-Request-Id"] = request.state.request_id
        return resp


app.add_middleware(RequestIdMiddleware)

# ----------------------------
# IMAGE UTILS
# ----------------------------
def bytes_to_bgr(data: bytes) -> np.ndarray:
    pil = Image.open(io.BytesIO(data)).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

def bgr_to_png_bytes(bgr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    bgr_to_pil(bgr).save(buf, format="PNG")
    return buf.getvalue()

def save_bgr_jpg(path: str, bgr: np.ndarray):
    bgr_to_pil(bgr).save(path, "JPEG", quality=92, optimize=True)

def safe_output_path(filename: str) -> str:
    return os.path.join(OUTPUT_DIR, os.path.basename(filename))

def build_url(request: Request, filename: str) -> str:
    base = str(request.base_url).rstrip("/")
    return f"{base}/outputs/{filename}"

def build_download_url(request: Request, filename: str) -> str:
    base = str(request.base_url).rstrip("/")
    return f"{base}/download/{filename}"

def build_view_url(request: Request, filename: str) -> str:
    base = str(request.base_url).rstrip("/")
    return f"{base}/view/{filename}"

def deep_find_url(obj: Any) -> Optional[str]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("url", "image_url", "output_url") and isinstance(v, str) and v.startswith("http"):
                return v
            got = deep_find_url(v)
            if got:
                return got
    if isinstance(obj, list):
        for it in obj:
            got = deep_find_url(it)
            if got:
                return got
    return None

def fetch_image_url_to_bgr(url: str, timeout: int = 240) -> np.ndarray:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    pil = Image.open(io.BytesIO(r.content)).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def resize_max_side(bgr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return bgr
    scale = max_side / float(m)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)

# ----------------------------
# PREPROCESS FOR "ACTRESS" / WORN REFERENCES
# ----------------------------
def _tight_bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    # mask: uint8 0/255
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return x1, y1, x2, y2

def remove_bg_with_rembg(bgr: np.ndarray) -> Optional[np.ndarray]:
    if rembg_remove is None:
        return None
    # rembg expects bytes; return RGBA; we convert back
    pil = bgr_to_pil(bgr).convert("RGBA")
    out = rembg_remove(pil)  # PIL RGBA
    out_np = np.array(out)   # RGBA
    rgba = out_np
    # Replace transparent with white background to keep garment edges clean
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    rgb = rgba[:, :, :3].astype(np.float32)
    white = np.ones_like(rgb) * 255.0
    comp = rgb * alpha + white * (1.0 - alpha)
    comp = comp.astype(np.uint8)
    return cv2.cvtColor(comp, cv2.COLOR_RGB2BGR)

def quick_grabcut_fg(bgr: np.ndarray) -> np.ndarray:
    # Fallback if rembg isn't available: coarse foreground extraction
    h, w = bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    # rectangle: assume subject centered
    rx1, ry1 = int(w * 0.08), int(h * 0.05)
    rx2, ry2 = int(w * 0.92), int(h * 0.98)
    rect = (rx1, ry1, rx2 - rx1, ry2 - ry1)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)
        fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        # compose on white
        fg3 = fg[:, :, None].astype(np.float32) / 255.0
        comp = bgr.astype(np.float32) * fg3 + 255.0 * (1.0 - fg3)
        return comp.astype(np.uint8)
    except Exception:
        return bgr

def blur_face_hands_heuristic(bgr: np.ndarray) -> np.ndarray:
    """
    No heavy models. Heuristic: blur top region (face/hair) and side edges (hands)
    to reduce Cat-VTON confusion when the reference contains a person.
    """
    out = bgr.copy()
    h, w = out.shape[:2]

    # blur top ~22% (face/hair)
    y2 = int(h * 0.22)
    if y2 > 10:
        top = out[:y2, :]
        top = cv2.GaussianBlur(top, (0, 0), sigmaX=8, sigmaY=8)
        out[:y2, :] = top

    # blur thin strips on left/right (hands/arms often there)
    strip = int(w * 0.10)
    if strip > 5:
        left = cv2.GaussianBlur(out[:, :strip], (0, 0), sigmaX=6, sigmaY=6)
        right = cv2.GaussianBlur(out[:, w - strip :], (0, 0), sigmaX=6, sigmaY=6)
        out[:, :strip] = left
        out[:, w - strip :] = right

    return out

def preprocess_reference_for_catvton(ref_bgr: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    Make model-worn reference look more like "garment" input:
    - remove background (rembg if possible; else grabcut)
    - crop tight
    - blur face/hands (optional)
    - resize
    """
    warnings: List[str] = []
    bgr = ref_bgr.copy()

    # 1) remove bg
    if rembg_remove is not None:
        try:
            bgr2 = remove_bg_with_rembg(bgr)
            if bgr2 is not None:
                bgr = bgr2
                warnings.append("ref_bg_removed_rembg")
        except Exception:
            warnings.append("ref_bg_remove_failed_rembg")
    else:
        # fallback
        bgr = quick_grabcut_fg(bgr)
        warnings.append("ref_bg_removed_grabcut")

    # 2) build mask for tight crop using "non-white" pixels
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # consider pixels far from white
    mask = (gray < 245).astype(np.uint8) * 255
    # clean mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    bbox = _tight_bbox_from_mask(mask)
    if bbox:
        x1, y1, x2, y2 = bbox
        h, w = bgr.shape[:2]
        pad_x = int((x2 - x1) * REF_CROP_PAD)
        pad_y = int((y2 - y1) * REF_CROP_PAD)
        x1 = max(0, x1 - pad_x); y1 = max(0, y1 - pad_y)
        x2 = min(w - 1, x2 + pad_x); y2 = min(h - 1, y2 + pad_y)
        bgr = bgr[y1 : y2 + 1, x1 : x2 + 1]
        warnings.append("ref_tight_crop")
    else:
        warnings.append("ref_crop_failed_used_full")

    # 3) blur face/hands
    if REF_BLUR_FACE_HANDS:
        bgr = blur_face_hands_heuristic(bgr)
        warnings.append("ref_blur_face_hands")

    # 4) normalize size
    bgr = resize_max_side(bgr, REF_MAX_SIDE)
    return bgr, warnings

# ----------------------------
# OUTPUT VIEW/DOWNLOAD
# ----------------------------
@app.get("/download/{filename}")
def download_output(request: Request, filename: str):
    path = safe_output_path(filename)
    if not os.path.exists(path):
        return JSONResponse(status_code=404, content={"ok": False, "error": "file_not_found"})
    return FileResponse(
        path=path,
        media_type="image/jpeg",
        filename=os.path.basename(filename),
        headers={"Cache-Control": "no-store"},
    )

@app.get("/view/{filename}")
def view_output(request: Request, filename: str):
    path = safe_output_path(filename)
    if not os.path.exists(path):
        return HTMLResponse("<h3>File not found</h3>", status_code=404)

    img_url = build_url(request, os.path.basename(filename))
    dl_url = build_download_url(request, os.path.basename(filename))

    html = f"""
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8"/>
        <meta name="viewport" content="width=device-width,initial-scale=1"/>
        <title>Output</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 24px; }}
          .wrap {{ max-width: 900px; margin: 0 auto; }}
          img {{ width: 100%; height: auto; border-radius: 12px; box-shadow: 0 6px 18px rgba(0,0,0,0.15); }}
          .btns {{ margin-top: 16px; display: flex; gap: 12px; flex-wrap: wrap; }}
          a.button {{ display:inline-block;padding:10px 14px;border-radius:10px;text-decoration:none;border:1px solid #ddd;color:#111; }}
          a.primary {{ background:#111;color:#fff;border-color:#111; }}
        </style>
      </head>
      <body>
        <div class="wrap">
          <h2>Output</h2>
          <img src="{img_url}" alt="Output"/>
          <div class="btns">
            <a class="button primary" href="{dl_url}">Download Image</a>
            <a class="button" href="{img_url}" target="_blank">Open Raw Image</a>
          </div>
        </div>
      </body>
    </html>
    """
    return HTMLResponse(html)

# ----------------------------
# RESPONSE BUILDER
# ----------------------------
def make_response(
    request: Request,
    request_id: str,
    success: bool,
    mode_used: str,
    warnings: List[str],
    error: Optional[Dict[str, Any]] = None,
    external_output_url: Optional[str] = None,
    out_bgr: Optional[np.ndarray] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    output_urls: List[str] = []
    output_download_urls: List[str] = []
    output_view_urls: List[str] = []

    if out_bgr is not None:
        fname = f"out_{mode_used}_{uuid.uuid4().hex[:10]}.jpg"
        save_bgr_jpg(os.path.join(OUTPUT_DIR, fname), out_bgr)
        output_urls = [build_url(request, fname)]
        output_download_urls = [build_download_url(request, fname)]
        output_view_urls = [build_view_url(request, fname)]

    return {
        "request_id": request_id,
        "success": success,
        "mode_used": mode_used,
        "warnings": warnings,
        "error": error,
        "external_output_url": external_output_url,
        "output_urls": output_urls,
        "output_download_urls": output_download_urls,
        "output_view_urls": output_view_urls,
        "meta": meta or {},
    }

# ----------------------------
# FAL HELPERS
# ----------------------------
def fal_upload_png_bytes(png_bytes: bytes) -> str:
    if not fal_client:
        raise RuntimeError("fal_client not installed. pip install fal-client")
    up = fal_client.upload(
        png_bytes,
        content_type="image/png",
        file_name=f"upload_{uuid.uuid4().hex[:8]}.png",
    )
    if isinstance(up, str):
        return up
    if isinstance(up, dict):
        url = up.get("url") or up.get("file_url") or up.get("fileUrl")
        if url:
            return url
    raise RuntimeError(f"Unexpected upload response: {type(up)}")

# ----------------------------
# PROMPT SHAPING (OPTIONAL)
# ----------------------------
def gemini_refine_prompt(prompt: str) -> Tuple[str, Optional[str]]:
    p = (prompt or "").strip()
    if not p:
        return ("Change the outfit to a stylish dress.", "empty_prompt")

    if genai_client is None:
        return (p, "gemini_not_configured")

    sys = (
        "Rewrite the user's fashion instruction to be clearer and more specific.\n"
        "Keep it under 25 words. Include color, fabric, silhouette, neckline/sleeves, length, key details.\n"
        "Return ONLY the rewritten prompt."
    )
    try:
        resp = genai_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=f"{sys}\n\nUser prompt: {p}",
        )
        text = (resp.text or "").strip()
        return (text if text else p, None)
    except Exception as e:
        return (p, f"gemini_failed:{type(e).__name__}")

# ----------------------------
# CAT-VTON CLOTH TYPE
# ----------------------------
def normalize_cloth_type(x: str) -> str:
    s = (x or "").strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "upper_body": "upper",
        "upper": "upper",
        "top": "upper",
        "tops": "upper",
        "shirt": "upper",
        "tshirt": "upper",
        "t_shirt": "upper",
        "lower_body": "lower",
        "lower": "lower",
        "bottom": "lower",
        "bottoms": "lower",
        "pants": "lower",
        "jeans": "lower",
        "skirt": "lower",
        "dress": "overall",
        "dresses": "overall",
        "overall": "overall",
        "inner": "inner",
        "outer": "outer",
    }
    return mapping.get(s, "overall")

# ----------------------------
# PIPELINES
# ----------------------------
def run_flux_remix(request: Request, user_bgr: np.ndarray, user_prompt: str) -> Dict[str, Any]:
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:12])
    warnings: List[str] = []

    if not fal_client:
        return make_response(request, request_id, False, "flux_edit_failed", [],
                             error={"type": "MissingDependency", "message": "fal_client not installed. pip install fal-client"})
    if not FAL_API_KEY:
        return make_response(request, request_id, False, "flux_edit_failed", [],
                             error={"type": "MissingKey", "message": "Missing FAL_API_KEY / FAL_KEY"})

    refined, warn = gemini_refine_prompt(user_prompt)
    if warn:
        warnings.append(warn)

    flux_prompt = (
        f"Change ONLY the outfit to: {refined}. "
        f"Keep same person identity, pose, face, hair, background, lighting. "
        f"Do not change body shape. High realism."
    )

    try:
        user_url = fal_upload_png_bytes(bgr_to_png_bytes(resize_max_side(user_bgr, 1400)))

        result = fal_client.run(
            FAL_FLUX_EDIT_MODEL,
            arguments={
                "prompt": flux_prompt,
                "image_urls": [user_url],
                "guidance_scale": FLUX_GUIDANCE,
                "num_inference_steps": FLUX_STEPS,
                "image_size": FLUX_IMAGE_SIZE,
                "num_images": 1,
                "enable_prompt_expansion": FLUX_ENABLE_PROMPT_EXPANSION,
                "output_format": "png",
            },
        )

        out_url = None
        if isinstance(result, dict):
            imgs = result.get("images")
            if isinstance(imgs, list) and imgs and isinstance(imgs[0], dict):
                out_url = imgs[0].get("url")
        if not out_url:
            out_url = deep_find_url(result)

        if not out_url:
            return make_response(request, request_id, False, "flux_edit_failed", warnings,
                                 error={"type": "NoOutput", "message": "flux edit returned no image url"},
                                 meta={"refined_prompt": refined})

        out_bgr = fetch_image_url_to_bgr(out_url)
        return make_response(request, request_id, True, "outfit_remix_flux", warnings,
                             external_output_url=out_url, out_bgr=out_bgr,
                             meta={"refined_prompt": refined, "flux_prompt": flux_prompt})

    except Exception as e:
        return make_response(request, request_id, False, "flux_edit_failed", warnings,
                             error={"type": "FluxEditFailed", "message": f"{type(e).__name__}: {e}"})


def run_catvton_tryon(request: Request, user_bgr: np.ndarray, garment_bgr: np.ndarray, cloth_type: str) -> Dict[str, Any]:
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:12])
    warnings: List[str] = []

    if not fal_client:
        return make_response(request, request_id, False, "catvton_failed", [],
                             error={"type": "MissingDependency", "message": "fal_client not installed. pip install fal-client"})
    if not FAL_API_KEY:
        return make_response(request, request_id, False, "catvton_failed", [],
                             error={"type": "MissingKey", "message": "Missing FAL_API_KEY / FAL_KEY"})

    try:
        # Normalize sizes for stability
        user_bgr = resize_max_side(user_bgr, 1400)
        garment_bgr = resize_max_side(garment_bgr, REF_MAX_SIDE)

        human_url = fal_upload_png_bytes(bgr_to_png_bytes(user_bgr))
        garm_url = fal_upload_png_bytes(bgr_to_png_bytes(garment_bgr))

        result = fal_client.run(
            FAL_CATVTON_MODEL,
            arguments={
                "human_image_url": human_url,
                "garment_image_url": garm_url,
                "cloth_type": cloth_type,
                "image_size": CATVTON_IMAGE_SIZE,
                "num_inference_steps": CATVTON_STEPS,
                "guidance_scale": CATVTON_GUIDANCE,
            },
        )

        out_url = None
        if isinstance(result, dict):
            img = result.get("image")
            if isinstance(img, dict):
                out_url = img.get("url")
        if not out_url:
            out_url = deep_find_url(result)

        if not out_url:
            return make_response(request, request_id, False, "catvton_failed", warnings,
                                 error={"type": "NoOutput", "message": "cat-vton returned no image url"},
                                 meta={"cloth_type": cloth_type})

        out_bgr = fetch_image_url_to_bgr(out_url)
        return make_response(request, request_id, True, "tryon_catvton", warnings,
                             external_output_url=out_url, out_bgr=out_bgr,
                             meta={"cloth_type": cloth_type})

    except Exception as e:
        return make_response(request, request_id, False, "catvton_failed", warnings,
                             error={"type": "TryOnFailed", "message": f"{type(e).__name__}: {e}"},
                             meta={"cloth_type": cloth_type})

# ----------------------------
# HEALTH
# ----------------------------
@app.get("/")
@app.get("/health")
def health():
    return {
        "ok": True,
        "fal_key_set": bool(FAL_API_KEY),
        "fal_client_ok": bool(fal_client),
        "gemini_configured": bool(genai_client),
        "rembg_available": bool(rembg_remove is not None),
        "preprocess_enabled": ENABLE_REF_PREPROCESS,
        "models": {"flux_edit": FAL_FLUX_EDIT_MODEL, "cat_vton": FAL_CATVTON_MODEL},
        "catvton_cloth_type_permitted": ["upper", "lower", "overall", "inner", "outer"],
        "endpoints": ["/v1/outfit/remix", "/v1/tryon/garment-to-user", "/v1/tryon/actress-to-user"],
    }

# ----------------------------
# ENDPOINTS
# ----------------------------
@app.post("/v1/outfit/remix")
async def outfit_remix(
    request: Request,
    user_image: UploadFile = File(...),
    prompt: str = Form(...),
):
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:12])

    try:
        user_bytes = await user_image.read()
    except Exception:
        return JSONResponse(status_code=200, content=make_response(
            request, request_id, False, "input_error", [],
            error={"type": "ImageReadError", "message": "failed to read user_image"},
        ))

    def work() -> Dict[str, Any]:
        try:
            user_bgr = bytes_to_bgr(user_bytes)
        except Exception as e:
            return make_response(request, request_id, False, "input_error", [],
                                 error={"type": "ImageDecodeError", "message": f"{type(e).__name__}: failed to decode user image"})
        return run_flux_remix(request, user_bgr, prompt)

    return await anyio.to_thread.run_sync(work)


@app.post("/v1/tryon/garment-to-user")
async def garment_to_user(
    request: Request,
    garment_image: UploadFile = File(...),
    user_image: UploadFile = File(...),
    garment_type: str = Form("overall"),
):
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:12])

    try:
        garment_bytes = await garment_image.read()
        user_bytes = await user_image.read()
    except Exception:
        return JSONResponse(status_code=200, content=make_response(
            request, request_id, False, "input_error", [],
            error={"type": "ImageReadError", "message": "failed to read images"},
        ))

    def work() -> Dict[str, Any]:
        try:
            garment_bgr = bytes_to_bgr(garment_bytes)
            user_bgr = bytes_to_bgr(user_bytes)
        except Exception:
            return make_response(request, request_id, False, "input_error", [],
                                 error={"type": "ImageDecodeError", "message": "failed to decode images"})

        cloth_type = normalize_cloth_type(garment_type)
        # For clean garments, no need to preprocess; but you can still resize
        garment_bgr = resize_max_side(garment_bgr, REF_MAX_SIDE)
        return run_catvton_tryon(request, user_bgr, garment_bgr, cloth_type)

    return await anyio.to_thread.run_sync(work)


@app.post("/v1/tryon/actress-to-user")
async def actress_to_user(
    request: Request,
    actress_image: UploadFile = File(...),
    user_image: UploadFile = File(...),
    garment_type: str = Form("overall"),
):
    """
    Use actress_image as a worn reference.
    We preprocess it to look more like a garment reference before Cat-VTON.
    """
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:12])

    try:
        actress_bytes = await actress_image.read()
        user_bytes = await user_image.read()
    except Exception:
        return JSONResponse(status_code=200, content=make_response(
            request, request_id, False, "input_error", [],
            error={"type": "ImageReadError", "message": "failed to read images"},
        ))

    def work() -> Dict[str, Any]:
        try:
            actress_bgr = bytes_to_bgr(actress_bytes)
            user_bgr = bytes_to_bgr(user_bytes)
        except Exception:
            return make_response(request, request_id, False, "input_error", [],
                                 error={"type": "ImageDecodeError", "message": "failed to decode images"})

        cloth_type = normalize_cloth_type(garment_type)

        warnings: List[str] = []
        if ENABLE_REF_PREPROCESS:
            actress_bgr, w = preprocess_reference_for_catvton(actress_bgr)
            warnings.extend(w)

        resp = run_catvton_tryon(request, user_bgr, actress_bgr, cloth_type)
        # merge warnings
        resp["warnings"] = (resp.get("warnings") or []) + warnings
        resp.setdefault("meta", {})
        resp["meta"].update({"preprocess_enabled": ENABLE_REF_PREPROCESS, "cloth_type": cloth_type})
        return resp

    return await anyio.to_thread.run_sync(work)

# ----------------------------
# GLOBAL EXCEPTION HANDLER
# ----------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex[:12])
    tb = traceback.format_exc()
    print(f"[{request_id}] Unhandled exception: {type(exc).__name__}: {exc}\n{tb}", flush=True)
    return JSONResponse(
        status_code=200,
        content=make_response(
            request, request_id, False, "unhandled_exception", [],
            error={"type": type(exc).__name__, "message": "Unhandled server exception"},
        ),
    )

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    # IMPORTANT: module must match this filename (without .py)
    uvicorn.run("fal_ai_backend:app", host="0.0.0.0", port=PORT, log_level=LOG_LEVEL)
