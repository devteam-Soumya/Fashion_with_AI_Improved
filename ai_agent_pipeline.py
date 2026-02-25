# backend_full_gown_agent.py — COMPLETE FINAL VERSION
# ══════════════════════════════════════════════════════════════════════════════
#
#  SWAP MECHANISM EXPLAINED:
#  ─────────────────────────────────────────────────────────────────────────
#
#  CASE A — Midi / Short Dress (like camel wrap dress):
#    M0: Gemini reads flat-lay → is_full_length = False
#    M1: Uses preprocessed ACTRESS as garment input to CatVTON
#        (actress reference = best for same-length garments)
#    M2: SKIPPED — no canvas extension needed
#    M3: Gemini QC → fixes extra_outerwear, seam_bleed, color_mismatch
#
#  CASE B — Full-Length Gown / Maxi:
#    M0: Gemini reads flat-lay → is_full_length = True
#    M1: Uses FLAT-LAY as garment input (actress wears the short version)
#    M2: Flux Kontext extends result to floor length
#    M3: Gemini QC → fixes artefacts after extension
#
#  ALL FIXES:
#  ─────────────────────────────────────────────────────────────────────────
#  FIX 1:  extract_json()        strips ```json``` markdown before json.loads()
#  FIX 2:  fal_upload() in M3    HTTPS URLs not data URIs for fal.ai models
#  FIX 3:  No resolution_mode    removed invalid Flux Kontext parameter
#  FIX 4:  40px canvas sample    smoother fill colour at extension seam
#  FIX 5:  Heuristic fallback    aspect ratio guess when Gemini M0 fails
#  FIX 6:  Seam blending         Gaussian fade removes hard line at canvas join
#  FIX 7:  final_bgr safety      always assigned even when skip_m3=1
#  FIX 8:  Consistent errors     all endpoints return same JSON error shape
#  FIX 9:  sanitize_url()        strips trailing quotes from fal.ai URLs
#                                (Windows OSError WinError 123 root cause fix)
#  FIX 10: sanitize_filename()   removes Windows-illegal chars from file paths
#  FIX 11: extra_outerwear QC    M3 prompt explicitly handles coat/jacket removal
#  FIX 12: Flat-lay watermark    stock photo watermarks inpainted before CatVTON
#  FIX 13: ENHANCED EXTENSION    Increased EXTEND_RATIO to 0.85 for full gowns
#  FIX 14: Better seam blending  50px blend zone instead of 30px
#  FIX 15: Stronger prompts      Explicit "FULL-LENGTH" instructions in M2
#
#  ENDPOINTS:
#   GET  /health
#   GET  /download/{filename}
#   GET  /view/{filename}
#   POST /v1/tryon/garment-to-user
#   POST /v1/tryon/actress-to-user
#   POST /v1/tryon/actress-garment-to-user   ← RECOMMENDED
#
# ══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations
import os, io, re, uuid, json, traceback, time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List, Dict, Any
import requests
import numpy as np, cv2
from PIL import Image
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from starlette.middleware.base import BaseHTTPMiddleware
from dotenv import load_dotenv
load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
PORT       = int(os.getenv("PORT",        "8000"))
LOG_LEVEL  = os.getenv("LOG_LEVEL",       "info")
OUTPUT_DIR = os.getenv("OUTPUT_DIR",      "./outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FAL_KEY = os.getenv("FAL_API_KEY", os.getenv("FAL_KEY", "")).strip()
if FAL_KEY:
    os.environ["FAL_KEY"]     = FAL_KEY
    os.environ["FAL_API_KEY"] = FAL_KEY

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL   = os.getenv("GEMINI_MODEL",   "gemini-2.0-flash")

FAL_CATVTON = "fal-ai/cat-vton"
FAL_KONTEXT = "fal-ai/flux-kontext/dev"
FAL_FILL    = "fal-ai/flux-pro/v1/fill"

CATVTON_STEPS    = int(os.getenv("CATVTON_STEPS",     "30"))
CATVTON_GUIDANCE = float(os.getenv("CATVTON_GUIDANCE", "2.5"))
KONTEXT_STEPS    = int(os.getenv("KONTEXT_STEPS",     "28"))
KONTEXT_GUIDANCE = float(os.getenv("KONTEXT_GUIDANCE", "2.5"))

# FIX 13: Increased from 0.55 to 0.85 for better full-length extension
EXTEND_RATIO     = float(os.getenv("EXTEND_RATIO",    "0.85"))

REF_MAX_SIDE     = int(os.getenv("REF_MAX_SIDE",      "1024"))
REF_CROP_PAD     = float(os.getenv("REF_CROP_PAD",    "0.12"))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CLIENTS
# ══════════════════════════════════════════════════════════════════════════════
try:
    import fal_client
except Exception:
    fal_client = None

try:
    from rembg import remove as rembg_remove
except Exception:
    rembg_remove = None

gemini_client = None
try:
    if GEMINI_API_KEY:
        from google import genai
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
except Exception:
    pass

print(f"\n{'='*60}")
print(f"  VIRTUAL TRY-ON AGENT — Complete Final Version")
print(f"  fal.ai  : {'READY' if FAL_KEY and fal_client else 'MISSING FAL_KEY'}")
print(f"  Gemini  : {'READY' if gemini_client else 'MISSING GEMINI_API_KEY'}")
print(f"  rembg   : {'READY' if rembg_remove else 'not installed (GrabCut fallback)'}")
print(f"{'='*60}\n")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DATA TYPES
# ══════════════════════════════════════════════════════════════════════════════
class QCStrategy(str, Enum):
    NONE    = "none"
    KONTEXT = "kontext"
    FILL    = "fill"
    EXTEND  = "extend"
    BOTH    = "both"

@dataclass
class GarmentInfo:
    """Output of Module 0 — describes the product garment."""
    is_full_length: bool   # True = gown/maxi/palazzo, False = midi/mini/top
    has_palazzo:    bool   # True = wide-leg pant layer visible
    garment_type:   str    # e.g. "camel midi wrap dress"
    style_desc:     str    # rich description for extension prompts
    extend_prompt:  str    # exact Flux Kontext instruction (used in M2)
    cloth_type:     str    # "upper" | "lower" | "overall" for CatVTON

@dataclass
class QCReport:
    """Output of Module 3 Step A — Gemini quality inspection."""
    has_issues:          bool
    strategy:            QCStrategy
    issues:              List[str]
    problem_summary:     str
    region_hint:         Optional[str]
    kontext_instruction: str
    fill_prompt:         str
    negative_prompt:     str

@dataclass
class PipelineState:
    """Single typed object that flows through all 4 modules."""
    rid:          str
    # ── Inputs ─────────────────────────────────────────
    user_bgr:     Optional[np.ndarray] = None
    garment_bgr:  Optional[np.ndarray] = None   # flat-lay product image
    actress_bgr:  Optional[np.ndarray] = None   # worn reference (optional)
    style_hint:   str = ""
    # ── M0 output ───────────────────────────────────────
    garment_info: Optional[GarmentInfo] = None
    # ── M1 output ───────────────────────────────────────
    m1_bgr:       Optional[np.ndarray] = None
    m1_model:     str = ""
    m1_latency:   float = 0.0
    m1_warns:     List[str] = field(default_factory=list)
    # ── M2 output ───────────────────────────────────────
    m2_bgr:       Optional[np.ndarray] = None
    m2_extended:  bool = False
    m2_latency:   float = 0.0
    m2_warns:     List[str] = field(default_factory=list)
    # ── M3 output ───────────────────────────────────────
    qc_report:    Optional[QCReport] = None
    m3_bgr:       Optional[np.ndarray] = None
    m3_strategy:  str = "skipped"
    m3_latency:   float = 0.0
    m3_warns:     List[str] = field(default_factory=list)
    # ── Final ────────────────────────────────────────────
    final_bgr:    Optional[np.ndarray] = None
    failed:       bool = False
    error:        Optional[str] = None
    all_warns:    List[str] = field(default_factory=list)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — IMAGE HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def read_bgr(data: bytes) -> np.ndarray:
    return cv2.cvtColor(
        np.array(Image.open(io.BytesIO(data)).convert("RGB")),
        cv2.COLOR_RGB2BGR
    )

def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

def bgr_to_png(bgr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    bgr_to_pil(bgr).save(buf, "PNG")
    return buf.getvalue()

def pil_to_png(pil: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, "PNG")
    return buf.getvalue()

def rmax(bgr: np.ndarray, ms: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= ms:
        return bgr
    s = ms / float(m)
    return cv2.resize(bgr, (max(1, int(w * s)), max(1, int(h * s))),
                     interpolation=cv2.INTER_AREA)

def fetch_bgr(url: str) -> np.ndarray:
    r = requests.get(url, timeout=240)
    r.raise_for_status()
    return cv2.cvtColor(
        np.array(Image.open(io.BytesIO(r.content)).convert("RGB")),
        cv2.COLOR_RGB2BGR
    )

def find_url(obj: Any) -> Optional[str]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("url", "image_url", "output_url") \
               and isinstance(v, str) and v.startswith("http"):
                return sanitize_url(v)
            u = find_url(v)
            if u:
                return u
    if isinstance(obj, list):
        for it in obj:
            u = find_url(it)
            if u:
                return u
    return None

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — PATH / URL SANITIZATION
# ══════════════════════════════════════════════════════════════════════════════
def sanitize_url(url: str) -> str:
    return url.strip().strip('"').strip("'").rstrip("\\/'")

def sanitize_filename(fn: str) -> str:
    return re.sub(r"[^\w.\-]", "", fn)

def make_result_filename() -> str:
    return sanitize_filename(f"result_{uuid.uuid4().hex[:8]}.jpg")

def safe_path(fn: str) -> str:
    return os.path.join(OUTPUT_DIR, sanitize_filename(os.path.basename(fn)))

def save_jpg(path: str, bgr: np.ndarray):
    bgr_to_pil(bgr).save(path, "JPEG", quality=92)

def ourl(req, fn: str) -> str:
    fn = sanitize_filename(os.path.basename(fn))
    return f"{str(req.base_url).rstrip('/')}/outputs/{fn}"

def durl(req, fn: str) -> str:
    fn = sanitize_filename(os.path.basename(fn))
    return f"{str(req.base_url).rstrip('/')}/download/{fn}"

def vurl(req, fn: str) -> str:
    fn = sanitize_filename(os.path.basename(fn))
    return f"{str(req.base_url).rstrip('/')}/view/{fn}"

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — FAL UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
def fal_upload(png_bytes: bytes) -> str:
    if not fal_client:
        raise RuntimeError("fal_client not installed. pip install fal-client")
    up = fal_client.upload(
        png_bytes,
        content_type="image/png",
        file_name=f"img_{uuid.uuid4().hex[:8]}.png",
    )
    if isinstance(up, str):
        return sanitize_url(up)
    if isinstance(up, dict):
        raw = up.get("url") or up.get("file_url") or up.get("fileUrl") or ""
        return sanitize_url(raw)
    raise RuntimeError(f"fal_upload unexpected response: {up}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — JSON EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
def extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"No valid JSON in Gemini response:\n{text[:300]}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — FLAT-LAY PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
def preprocess_flatlay(bgr: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    Clean flat-lay before CatVTON:
    1. Inpaint stock watermark (bottom 8% height, center 60% width)
    2. Resize to REF_MAX_SIDE
    """
    warns = []
    out   = bgr.copy()
    h, w  = out.shape[:2]
    
    # Watermark zone: bottom 8%, center 60%
    wy1 = int(h * 0.92)
    wx1 = int(w * 0.20)
    wx2 = int(w * 0.80)
    
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[wy1:h, wx1:wx2] = 255
    
    try:
        out = cv2.inpaint(out, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        warns.append("watermark_region_inpainted")
    except Exception:
        # Fallback: flood with sampled colour from just above watermark zone
        sample_y = max(0, wy1 - 10)
        fill_col = out[sample_y, wx1:wx2].mean(axis=0).astype(np.uint8)
        out[wy1:h, wx1:wx2] = fill_col
        warns.append("watermark_region_colorfill_fallback")
    
    return rmax(out, REF_MAX_SIDE), warns

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — ACTRESS PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
def preprocess_actress(bgr: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    Strip background + blur face so CatVTON uses the garment shape only.
    WHY: Actress photo has a real-world background, accessories and skin.
    CatVTON needs a clean garment silhouette — not a portrait.
    Steps:
    1. Background removal  (rembg if available, GrabCut otherwise)
    2. Tight crop to garment foreground bounds
    3. Face blur top 22%  — CatVTON must not copy the face
    4. Side edge blur 10% — reduces arm/shoulder edge confusion
    5. Resize to REF_MAX_SIDE
    """
    warns = []
    out   = bgr.copy()
    
    # ── 1. Background removal ─────────────────────────────────────────────
    if rembg_remove:
        try:
            pil  = bgr_to_pil(out).convert("RGBA")
            rgba = np.array(rembg_remove(pil))
            a    = rgba[:, :, 3:4].astype(np.float32) / 255.0
            comp = (rgba[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
            out  = cv2.cvtColor(comp, cv2.COLOR_RGB2BGR)
            warns.append("rembg_bg_removed")
        except Exception as e:
            warns.append(f"rembg_failed:{e}")
    else:
        h, w = out.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        rect = (int(w * .07), int(h * .04), int(w * .86), int(h * .94))
        bgdM = np.zeros((1, 65), np.float64)
        fgdM = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(out, mask, rect, bgdM, fgdM, 3, cv2.GC_INIT_WITH_RECT)
            fg  = np.where(
                (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
            ).astype(np.uint8)
            a3  = fg[:, :, None].astype(np.float32) / 255.0
            out = (out * a3 + 255 * (1 - a3)).astype(np.uint8)
            warns.append("grabcut_bg_removed")
        except Exception:
            warns.append("bg_removal_failed")
    
    # ── 2. Tight crop ─────────────────────────────────────────────────────
    gray    = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    mask_fg = (gray < 248).astype(np.uint8) * 255
    mask_fg = cv2.morphologyEx(mask_fg, cv2.GC_PR_FGD, np.ones((9, 9), np.uint8))
    ys, xs  = np.where(mask_fg > 0)
    if len(xs):
        h, w = out.shape[:2]
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        px  = int((x2 - x1) * REF_CROP_PAD)
        py  = int((y2 - y1) * REF_CROP_PAD)
        out = out[max(0, y1 - py):min(h, y2 + py),
                 max(0, x1 - px):min(w, x2 + px)]
        warns.append("tight_crop")
    
    # ── 3. Face blur (top 22%) ────────────────────────────────────────────
    h, w   = out.shape[:2]
    y_face = int(h * 0.22)
    if y_face > 10:
        out[:y_face] = cv2.GaussianBlur(out[:y_face], (0, 0), 9, 9)
    
    # ── 4. Side edge blur (10% each side) ────────────────────────────────
    se = int(w * 0.10)
    if se > 4:
        out[:, :se]   = cv2.GaussianBlur(out[:, :se],   (0, 0), 7, 7)
        out[:, w-se:] = cv2.GaussianBlur(out[:, w-se:], (0, 0), 7, 7)
        warns.append("face_and_edges_blurred")
    
    return rmax(out, REF_MAX_SIDE), warns

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 0 — GarmentInspectorAgent
# ══════════════════════════════════════════════════════════════════════════════
class GarmentInspectorAgent:
    """
    Reads the flat-lay product image with Gemini Vision.
    The most important output is is_full_length — this drives the entire
    pipeline split between actress path (midi) and flat-lay path (gown).
    """
    _PROMPT = """
Analyse this garment product flat-lay image.
Reply ONLY in valid JSON — no markdown, no explanation:
{
  "is_full_length": true or false,
  "has_palazzo":    true or false,
  "garment_type":   "short label e.g. camel midi wrap dress",
  "cloth_type":     "upper" or "lower" or "overall",
  "style_desc":     "detailed description: silhouette, length, neckline, sleeve, colour, fabric, embellishments",
  "extend_prompt":  "Flux Kontext instruction to extend this garment downward — be specific about colour, pattern, hem"
}
RULES:
is_full_length = true  → maxi dress, floor-length gown, palazzo set,
                         long anarkali, long kurta set, saree, long jumpsuit
is_full_length = false → midi dress (below knee, above ankle), mini dress,
                         blouse, shirt, top, shorts, jeans, blazer dress
has_palazzo    = true  → separate wide-leg trouser layer visible in the garment
cloth_type             → "overall" for dresses/jumpsuits, "upper" for tops/
                         jackets, "lower" for trousers/skirts
""".strip()
    
    @staticmethod
    def _heuristic_is_full_length(bgr: np.ndarray) -> bool:
        """
        FIX 5: Fallback when Gemini fails.
        Full-length gowns in flat-lay tend to be much taller than wide.
        Threshold 1.6 is conservative to avoid false-positives on midi dresses.
        """
        h, w = bgr.shape[:2]
        return (h / max(w, 1)) > 1.6
    
    def _fallback_info(self, bgr: np.ndarray) -> GarmentInfo:
        return GarmentInfo(
            is_full_length = self._heuristic_is_full_length(bgr),
            has_palazzo    = False,
            garment_type   = "dress",
            style_desc     = "solid colour dress",
            extend_prompt  = (
                "Extend the dress downward to floor length, continuing the "
                "exact same fabric, colour and hem style."
            ),
            cloth_type = "overall",
        )
    
    async def run(self, state: PipelineState) -> PipelineState:
        if state.garment_bgr is None:
            state.garment_info = GarmentInfo(
                is_full_length=False, has_palazzo=False,
                garment_type="dress", style_desc="dress",
                extend_prompt="Extend to floor length.",
                cloth_type="overall",
            )
            state.all_warns.append("M0 skipped — no flat-lay provided")
            return state
        
        fallback = self._fallback_info(state.garment_bgr)
        
        if not gemini_client:
            state.all_warns.append("M0: Gemini not configured — heuristic fallback")
            state.garment_info = fallback
            return state
        
        try:
            from google.genai import types as gt
            img_part = gt.Part.from_bytes(
                data=bgr_to_png(state.garment_bgr), mime_type="image/png"
            )
            resp = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=["Analyse this garment:", img_part, self._PROMPT],
            )
            raw = (getattr(resp, "text", None) or "").strip()
            d   = extract_json(raw)  # FIX 1
            
            state.garment_info = GarmentInfo(
                is_full_length = bool(d.get("is_full_length", fallback.is_full_length)),
                has_palazzo    = bool(d.get("has_palazzo",    False)),
                garment_type   = str(d.get("garment_type",   fallback.garment_type)),
                style_desc     = str(d.get("style_desc",     fallback.style_desc)),
                extend_prompt  = str(d.get("extend_prompt",  fallback.extend_prompt)),
                cloth_type     = str(d.get("cloth_type",     "overall")),
            )
            gi = state.garment_info
            print(f"  [M0] is_full_length={gi.is_full_length}  "
                  f"has_palazzo={gi.has_palazzo}  type='{gi.garment_type}'")
        except Exception as e:
            print(f"  [M0] Gemini error: {e} — using heuristic fallback")
            state.all_warns.append(f"M0 Gemini error: {e}")
            state.garment_info = fallback
        
        return state

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — TryOnModule  (fal-ai/cat-vton)
# ══════════════════════════════════════════════════════════════════════════════
class TryOnModule:
    """
    GARMENT SOURCE DECISION TABLE:
    ┌──────────────────────────────────────────────────────────┐
    │ is_full_length=True  + flat-lay available                │
    │   → flat-lay as garment  (actress has short version)    │
    │   → M2 will extend to floor length                       │
    ├──────────────────────────────────────────────────────────┤
    │ is_full_length=False + actress available    ← THIS CASE  │
    │   → preprocessed actress as garment                      │
    │   → M2 skipped                                          │
    ├──────────────────────────────────────────────────────────┤
    │ actress not available (any length)                        │
    │   → flat-lay with watermark removal (FIX 12)             │
    └──────────────────────────────────────────────────────────┘
    """
    async def run(self, state: PipelineState) -> PipelineState:
        if not fal_client or not FAL_KEY:
            state.failed = True
            state.error  = "FAL_KEY or fal_client missing. pip install fal-client"
            return state
        
        t0 = time.monotonic()
        gi = state.garment_info
        
        garment_input: Optional[np.ndarray] = None
        cloth_type  = gi.cloth_type if gi else "overall"
        source_desc = ""
        
        # ── Decide which image becomes the garment ────────────────────────
        if gi and gi.is_full_length and state.garment_bgr is not None:
            # Full-length path: flat-lay shows complete gown shape
            cleaned, wm_warns = preprocess_flatlay(state.garment_bgr)
            garment_input = cleaned
            state.m1_warns.extend(wm_warns)
            state.m1_warns.append(
                "FULL-LENGTH: using flat-lay as garment "
                "(actress has short version — M2 will extend)"
            )
            source_desc = "flat-lay (full-length gown)"
            print(f"  [M1] FLAT-LAY path — full-length garment")
        elif state.actress_bgr is not None:
            # Midi/short path: actress reference is superior to flat-lay
            # because it shows how the garment drapes on an actual body
            proc, warns = preprocess_actress(state.actress_bgr)
            garment_input = proc
            state.m1_warns.extend(warns)
            state.m1_warns.append(
                "MIDI/SHORT: using actress reference "
                "(garment is not full-length — M2 will be skipped)"
            )
            source_desc = "actress (preprocessed)"
            print(f"  [M1] ACTRESS path — midi/short garment")
        elif state.garment_bgr is not None:
            # No actress available — flat-lay only
            cleaned, wm_warns = preprocess_flatlay(state.garment_bgr)
            garment_input = cleaned
            state.m1_warns.extend(wm_warns)
            source_desc = "flat-lay (no actress provided)"
            print(f"  [M1] FLAT-LAY path — no actress provided")
        else:
            state.failed = True
            state.error  = "No garment or actress image provided."
            return state
        
        print(f"  [M1] Source: {source_desc} | cloth_type={cloth_type}")
        
        try:
            user_url    = fal_upload(bgr_to_png(rmax(state.user_bgr, 1400)))
            garment_url = fal_upload(bgr_to_png(garment_input))
            
            print(f"  [M1] Calling {FAL_CATVTON}…")
            result = fal_client.run(
                FAL_CATVTON,
                arguments={
                    "human_image_url":     user_url,
                    "garment_image_url":   garment_url,
                    "cloth_type":          cloth_type,
                    "num_inference_steps": CATVTON_STEPS,
                    "guidance_scale":      CATVTON_GUIDANCE,
                    "seed":                42,
                },
            )
            
            # CatVTON can return image URL in several shapes
            url = None
            if isinstance(result, dict):
                img = result.get("image")
                if isinstance(img, dict):
                    url = img.get("url")
            if not url:
                url = find_url(result)
            if not url:
                raise RuntimeError(f"CatVTON no image URL in response: {result}")
            
            state.m1_bgr     = fetch_bgr(url)
            state.m1_model   = FAL_CATVTON
            state.m1_latency = round(time.monotonic() - t0, 2)
            print(f"  [M1] Done ({state.m1_latency}s)")
        except Exception as e:
            state.failed = True
            state.error  = f"CatVTON failed: {type(e).__name__}: {e}"
            print(f"  [M1] FAILED: {state.error}")
        
        return state

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — ExtensionModule  (fal-ai/flux-kontext/dev)
# Skipped for midi/short garments. Only runs for full-length gowns.
# ══════════════════════════════════════════════════════════════════════════════
class ExtensionModule:
    """
    Extends the CatVTON result to full floor-length for gowns.
    For the camel midi dress:  SKIPPED (is_full_length=False).
    For full-length gowns:
    1. Extend canvas downward by EXTEND_RATIO (85% for FIX 13)
       Seamless seam blending prevents hard colour line (FIX 4 + FIX 6 + FIX 14)
    2. Call Flux Kontext to generate the lower half continuation
       Low guidance_scale (2.5) preserves the person, generates only garment
    """
    def _extend_canvas(self, bgr: np.ndarray, ratio: float) -> np.ndarray:
        """
        FIX 4: sample 40px (was 15px) for more accurate average fill colour
        FIX 6 + FIX 14: blend last 50px of original into fill colour — no hard seam
        """
        h, w   = bgr.shape[:2]
        add_h  = int(h * ratio)
        sample = bgr[max(0, h - 40):h, :]       # FIX 4: 40px sample
        bottom = sample.mean(axis=(0, 1)).astype(np.uint8)
        fill   = np.full((add_h, w, 3), bottom, dtype=np.uint8)
        
        bgr_copy   = bgr.copy()
        blend_rows = min(50, h)  # FIX 14: increased from 30 to 50
        for i in range(blend_rows):              # FIX 6: seam blend
            alpha         = i / float(blend_rows)
            row           = h - blend_rows + i
            bgr_copy[row] = (bgr_copy[row] * (1 - alpha) + bottom * alpha).astype(np.uint8)
        
        return np.vstack([bgr_copy, fill])
    
    def _build_extension_prompt(self, gi: GarmentInfo, style_hint: str) -> str:
        base = gi.extend_prompt or (
            f"Extend the {gi.garment_type} downward to floor length, continuing "
            f"the exact same fabric, colour and embellishments: {gi.style_desc}."
        )
        if gi.has_palazzo:
            base += (
                " The palazzo/wide-leg trousers must be clearly visible "
                "through the centre slit, matching the product colour."
            )
        if style_hint:
            base += f" Note: {style_hint}."
        
        # FIX 15: Stronger emphasis on FULL-LENGTH
        return (
            f"{base} "
            "CRITICAL: This MUST be a FULL-LENGTH floor-sweeping gown. "
            "Extend significantly downward to show the complete hem touching the floor. "
            "Seamlessly continue from the current garment hem — match exact "
            "fabric texture, colour and hem border. "
            "PRESERVE exactly: face, hair, skin tone, upper body, pose, background. "
            "ONLY generate the lower extension. Photorealistic, high detail, "
            "fashion photography quality."
        )
    
    async def run(self, state: PipelineState) -> PipelineState:
        if state.failed or state.m1_bgr is None:
            return state
        
        gi = state.garment_info
        if not gi or not gi.is_full_length:
            state.m2_bgr = state.m1_bgr
            state.m2_warns.append("M2 skipped — garment is not full-length")
            print(f"  [M2] Skipped (midi/short garment — no extension needed)")
            return state
        
        if not fal_client or not FAL_KEY:
            state.m2_bgr = state.m1_bgr
            state.m2_warns.append("M2 skipped — FAL_KEY missing")
            return state
        
        t0 = time.monotonic()
        print(f"  [M2] Extending canvas → Flux Kontext…")
        
        try:
            extended = self._extend_canvas(state.m1_bgr.copy(), EXTEND_RATIO)
            prompt   = self._build_extension_prompt(gi, state.style_hint)
            ext_url  = fal_upload(bgr_to_png(extended))
            
            print(f"  [M2] Prompt: {prompt[:100]}…")
            
            # FIX 3: "resolution_mode" is NOT a valid Flux Kontext parameter — removed
            resp = fal_client.run(
                FAL_KONTEXT,
                arguments={
                    "image_url":           ext_url,
                    "prompt":              prompt,
                    "num_inference_steps": KONTEXT_STEPS,
                    "guidance_scale":      KONTEXT_GUIDANCE,
                    "num_images":          1,
                    "seed":                42,
                    "output_format":       "jpeg",
                },
            )
            
            url = None
            if isinstance(resp, dict):
                imgs = resp.get("images")
                if isinstance(imgs, list) and imgs:
                    url = imgs[0].get("url") if isinstance(imgs[0], dict) else None
            if not url:
                url = find_url(resp)
            if not url:
                raise RuntimeError(f"Kontext M2 returned no URL: {resp}")
            
            state.m2_bgr      = fetch_bgr(url)
            state.m2_extended = True
            state.m2_latency  = round(time.monotonic() - t0, 2)
            print(f"  [M2] Extended ({state.m2_latency}s)")
        except Exception as e:
            warn = f"Extension failed: {type(e).__name__}: {e}"
            print(f"  [M2] WARNING: {warn} — using M1 result")
            state.m2_warns.append(warn)
            state.m2_bgr = state.m1_bgr
        
        return state

# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — AutoFixQCAgent
# Gemini inspects result → picks strategy → applies Kontext or Fill fix.
# ══════════════════════════════════════════════════════════════════════════════
class AutoFixQCAgent:
    """
    STEP A — Gemini Vision inspection:
    Compares result (Image 3) vs original user (Image 1) + flat-lay (Image 2)
    Detects garment errors and artefacts, selects a fix strategy.
    
    STEP B — Correction:
    "none"    → pass through (QC passed)
    "kontext" → Flux Kontext global edit (wrong garment, extra outerwear)
    "fill"    → Flux Fill masked inpaint (local artefact: seam, edge)
    "extend"  → another extension pass (still too short)
    "both"    → fill local artefact THEN kontext global polish
    
    FIX 11 — extra_outerwear:
    When CatVTON hallucinates a coat, military jacket or blazer that is
    NOT in the product flat-lay, Gemini sets strategy="kontext" and writes
    an explicit instruction to remove the outerwear and reveal only the
    correct product garment.
    """
    _INSPECT_PROMPT = """
You are a quality inspector for a virtual clothing try-on system.
You are given THREE images in this order:
Image 1 = ORIGINAL USER PHOTO   (ground truth: face, hair, skin, body, background)
Image 2 = PRODUCT FLAT-LAY      (ground truth: garment type, length, colour, design)
Image 3 = TRY-ON RESULT         (the output to inspect)

Carefully compare Image 3 against Images 1 and 2.

Detect these GARMENT issues (compare vs Image 2):
wrong_length        — garment in result is shorter OR longer than in product
missing_palazzo     — palazzo/trouser layer missing when product has it
extra_outerwear     — a coat, jacket, blazer, military uniform, or lapels
                      appeared in result that are NOT in the product flat-lay
color_mismatch      — dominant garment colour is wrong
incomplete_pattern  — fabric print, embroidery or embellishment missing
wrong_garment_type  — entirely different garment type was applied

Detect these ARTEFACT issues (compare vs Image 1):
ghost_limbs         — extra, doubled or merged arms/hands
seam_bleed          — garment colour bleeding onto skin, neck or hair
face_changed        — face features, skin tone or hair altered (CRITICAL)
background_corrupt  — background changed, smeared or removed
edge_artifact       — sharp unnatural hard edge on garment border

Choose a fix strategy:
"none"    → Image 3 looks correct — no fix needed
"kontext" → global garment issue: wrong type, extra outerwear, wrong length,
            wrong colour — use Flux Kontext without a mask
"fill"    → single localised artefact in one region — use Flux Fill with mask
"extend"  → garment still too short after extension pass — extend again
"both"    → fix local artefact with fill first, then global kontext polish

CRITICAL — extra_outerwear rule:
If you detect a coat, military jacket, blazer or lapels in Image 3 that are
NOT in the product flat-lay (Image 2), you MUST set strategy="kontext" and
write a kontext_instruction that explicitly removes the outerwear and shows
only the correct product garment underneath.

Reply ONLY in valid JSON with NO markdown fences:
{
  "has_issues":          true or false,
  "strategy":            "none" or "kontext" or "fill" or "extend" or "both",
  "issues":              ["detected_issue_codes"],
  "problem_summary":     "one clear sentence",
  "region_hint":         null or "upper torso" or "lower torso" or "full torso"
                         or "hem" or "lower half" or "collar"
                         or "left sleeve" or "right sleeve",
  "kontext_instruction": "precise Flux Kontext editing instruction",
  "fill_prompt":         "generation prompt for the masked region",
  "negative_prompt":     "things that must NOT appear in the corrected result"
}
""".strip()
    
    def _default_report(self) -> QCReport:
        """Used when Gemini is unavailable."""
        return QCReport(
            has_issues=True,
            strategy=QCStrategy.KONTEXT,
            issues=["gemini_unavailable"],
            problem_summary="Gemini QC unavailable — applying default Kontext correction.",
            region_hint=None,
            kontext_instruction=(
                "Fix any garment artefacts. Remove any hallucinated coat, jacket "
                "or outerwear that is NOT in the product. Correct garment length "
                "and colour. Preserve the exact face, hair, skin, body and background."
            ),
            fill_prompt="Clean realistic garment with natural fabric and drape.",
            negative_prompt=(
                "coat, jacket, blazer, military uniform, lapels, extra collar, "
                "ghost limbs, doubled arms, seam bleed, blurry face, "
                "wrong colour, watermark, artefacts"
            ),
        )
    
    async def _gemini_inspect(
        self,
        user_bgr:    np.ndarray,
        garment_bgr: Optional[np.ndarray],
        result_bgr:  np.ndarray,
    ) -> QCReport:
        if not gemini_client:
            return self._default_report()
        
        try:
            from google.genai import types as gt
            contents = [
                "Image 1 — Original user photo:",
                gt.Part.from_bytes(data=bgr_to_png(user_bgr), mime_type="image/png"),
            ]
            if garment_bgr is not None:
                contents += [
                    "Image 2 — Product garment flat-lay:",
                    gt.Part.from_bytes(data=bgr_to_png(garment_bgr), mime_type="image/png"),
                ]
            contents += [
                "Image 3 — Try-on result to inspect:",
                gt.Part.from_bytes(data=bgr_to_png(result_bgr), mime_type="image/png"),
                self._INSPECT_PROMPT,
            ]
            
            resp = gemini_client.models.generate_content(
                model=GEMINI_MODEL, contents=contents
            )
            raw  = (getattr(resp, "text", None) or "").strip()
            d    = extract_json(raw)  # FIX 1
            
            return QCReport(
                has_issues          = bool(d.get("has_issues", True)),
                strategy            = QCStrategy(d.get("strategy", "kontext")),
                issues              = [str(x) for x in (d.get("issues") or [])],
                problem_summary     = str(d.get("problem_summary", "")),
                region_hint         = d.get("region_hint"),
                kontext_instruction = str(d.get("kontext_instruction", "")),
                fill_prompt         = str(d.get("fill_prompt", "")),
                negative_prompt     = str(d.get("negative_prompt", "")),
            )
        except Exception as e:
            print(f"  [M3] Gemini inspect error: {e}")
            return self._default_report()
    
    def _heuristic_inspect(self, result_bgr: np.ndarray) -> QCReport:
        """Fast edge-detection fallback when Gemini is completely unavailable."""
        gray   = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2GRAY)
        edges  = cv2.Canny(gray, 80, 160)
        issues = ["edge_artifact"] if float(edges.mean() / 255) > 0.18 else []
        
        return QCReport(
            has_issues          = bool(issues),
            strategy            = QCStrategy.KONTEXT if issues else QCStrategy.NONE,
            issues              = issues,
            problem_summary     = f"Heuristic: {issues or 'clean'}",
            region_hint         = None,
            kontext_instruction = "Fix edge artefacts on garment seams, preserve identity.",
            fill_prompt         = "Clean natural garment fabric.",
            negative_prompt     = "sharp edges, seam bleed, ghost limbs",
        )
    
    def _build_mask(self, w: int, h: int, hint: Optional[str]) -> Image.Image:
        """Soft Gaussian mask for the hinted region."""
        REGIONS = {
            "upper torso":  (0.15, 0.50, 0.18, 0.82),
            "lower torso":  (0.45, 0.75, 0.18, 0.82),
            "full torso":   (0.12, 0.80, 0.12, 0.88),
            "hem":          (0.60, 0.95, 0.08, 0.92),
            "lower half":   (0.45, 1.00, 0.05, 0.95),
            "left sleeve":  (0.15, 0.60, 0.00, 0.32),
            "right sleeve": (0.15, 0.60, 0.68, 1.00),
            "collar":       (0.04, 0.22, 0.28, 0.72),
        }
        hl  = (hint or "full torso").lower()
        reg = next((v for k, v in REGIONS.items() if k in hl), REGIONS["full torso"])
        
        mask = np.zeros((h, w), np.uint8)
        mask[int(reg[0]*h):int(reg[1]*h), int(reg[2]*w):int(reg[3]*w)] = 255
        mask = cv2.GaussianBlur(mask, (61, 61), 0)
        return Image.fromarray(mask).convert("RGB")
    
    async def _run_kontext(self, bgr: np.ndarray, report: QCReport) -> np.ndarray:
        """
        FIX 2: fal_upload() — fal.ai requires HTTPS URL, not data URI
        FIX 3: no resolution_mode — invalid parameter removed
        """
        instr = (
            f"{report.kontext_instruction} "
            "Preserve exactly: face, hair, skin tone, body shape, pose, background. "
            "Only correct the garment issue."
        )
        print(f"  [M3-kontext] {instr[:100]}…")
        
        img_url = fal_upload(bgr_to_png(bgr))    # FIX 2
        resp    = fal_client.run(
            FAL_KONTEXT,
            arguments={
                "image_url":           img_url,
                "prompt":              instr,
                "num_inference_steps": KONTEXT_STEPS,
                "guidance_scale":      KONTEXT_GUIDANCE,
                "num_images":          1,
                "seed":                42,
                "output_format":       "jpeg",
                # FIX 3: resolution_mode removed
            },
        )
        
        imgs = resp.get("images", []) if isinstance(resp, dict) else []
        url  = imgs[0].get("url") if imgs and isinstance(imgs[0], dict) else find_url(resp)
        if not url:
            raise RuntimeError(f"Kontext M3 no URL: {resp}")
        return fetch_bgr(url)
    
    async def _run_fill(self, bgr: np.ndarray, report: QCReport) -> np.ndarray:
        """FIX 2: both image and mask uploaded as HTTPS URLs."""
        h, w     = bgr.shape[:2]
        mask     = self._build_mask(w, h, report.region_hint)
        prompt   = f"{report.fill_prompt} Seamless, natural fabric, photorealistic."
        
        print(f"  [M3-fill] region='{report.region_hint}' '{prompt[:60]}…'")
        
        img_url  = fal_upload(bgr_to_png(bgr))    # FIX 2
        mask_url = fal_upload(pil_to_png(mask))    # FIX 2
        
        resp = fal_client.run(
            FAL_FILL,
            arguments={
                "image_url":        img_url,
                "mask_url":         mask_url,
                "prompt":           prompt,
                "num_images":       1,
                "output_format":    "jpeg",
                "safety_tolerance": "6",
                "seed":             42,
            },
        )
        
        imgs = resp.get("images", []) if isinstance(resp, dict) else []
        url  = imgs[0].get("url") if imgs and isinstance(imgs[0], dict) else find_url(resp)
        if not url:
            raise RuntimeError(f"Fill M3 no URL: {resp}")
        return fetch_bgr(url)
    
    async def _run_extend_again(
        self, bgr: np.ndarray, gi: Optional[GarmentInfo], style_hint: str
    ) -> np.ndarray:
        """Trigger a second extension pass when garment is still too short."""
        dummy = PipelineState(
            rid="m3_ext", user_bgr=bgr,
            m1_bgr=bgr, garment_info=gi, style_hint=style_hint
        )
        dummy = await ExtensionModule().run(dummy)
        return dummy.m2_bgr if dummy.m2_bgr is not None else bgr
    
    async def run(self, state: PipelineState) -> PipelineState:
        if state.failed:
            return state
        
        src = state.m2_bgr if state.m2_bgr is not None else state.m1_bgr
        if src is None:
            return state
        
        t0 = time.monotonic()
        print(f"  [M3] Running QC inspection…")
        
        report = (
            await self._gemini_inspect(state.user_bgr, state.garment_bgr, src)
            if gemini_client else self._heuristic_inspect(src)
        )
        state.qc_report = report
        
        print(f"  [M3] has_issues={report.has_issues}  "
              f"strategy={report.strategy}  issues={report.issues}")
        print(f"  [M3] '{report.problem_summary}'")
        
        # No issues — pass through
        if not report.has_issues or report.strategy == QCStrategy.NONE:
            state.m3_bgr      = src
            state.m3_strategy = "none"
            state.m3_latency  = round(time.monotonic() - t0, 2)
            state.m3_warns.append("QC passed — no issues detected")
            state.final_bgr   = src
            return state
        
        # Apply fix strategy
        fixed = src
        try:
            if not fal_client or not FAL_KEY:
                raise RuntimeError("FAL_KEY or fal_client missing")
            
            if report.strategy in (QCStrategy.FILL, QCStrategy.BOTH):
                fixed = await self._run_fill(fixed, report)
            if report.strategy in (QCStrategy.KONTEXT, QCStrategy.BOTH):
                fixed = await self._run_kontext(fixed, report)
            if report.strategy == QCStrategy.EXTEND:
                fixed = await self._run_extend_again(
                    fixed, state.garment_info, state.style_hint
                )
            
            state.m3_strategy = report.strategy.value
        except Exception as e:
            warn = f"M3 fix failed ({report.strategy}): {e}"
            print(f"  [M3] WARNING: {warn}")
            state.m3_warns.append(warn)
            state.m3_strategy = "failed_passthrough"
            fixed = src
        
        state.m3_bgr     = fixed
        state.m3_latency = round(time.monotonic() - t0, 2)
        state.final_bgr  = fixed
        
        print(f"  [M3] Done — {state.m3_strategy} ({state.m3_latency}s)")
        return state

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════
_m0 = GarmentInspectorAgent()
_m1 = TryOnModule()
_m2 = ExtensionModule()
_m3 = AutoFixQCAgent()

async def run_pipeline(
    request:     Request,
    user_bgr:    np.ndarray,
    garment_bgr: Optional[np.ndarray],
    actress_bgr: Optional[np.ndarray],
    style_hint:  str,
    skip_m3:     bool = False,
) -> Dict[str, Any]:
    rid = getattr(request.state, "request_id", uuid.uuid4().hex[:12])
    print(f"\n[{rid}] ══ PIPELINE START ═══════════════════════════")
    
    state = PipelineState(
        rid=rid,
        user_bgr=user_bgr,
        garment_bgr=garment_bgr,
        actress_bgr=actress_bgr,
        style_hint=style_hint,
    )
    
    print(f"[{rid}] ▶ M0 GarmentInspectorAgent")
    state = await _m0.run(state)
    
    print(f"[{rid}] ▶ M1 TryOnModule (CatVTON)")
    state = await _m1.run(state)
    state.all_warns.extend(state.m1_warns)
    
    if state.failed:
        print(f"[{rid}] FAILED at M1: {state.error}")
        return _build_response(request, state, state.error)
    
    print(f"[{rid}] ▶ M2 ExtensionModule")
    state = await _m2.run(state)
    state.all_warns.extend(state.m2_warns)
    
    if not skip_m3:
        print(f"[{rid}] ▶ M3 AutoFixQCAgent")
        state = await _m3.run(state)
        state.all_warns.extend(state.m3_warns)
    else:
        # FIX 7: always assign final_bgr when M3 is skipped
        state.final_bgr   = state.m2_bgr if state.m2_bgr is not None else state.m1_bgr
        state.m3_strategy = "skipped"
    
    # FIX 7: safety net — final_bgr must never be None
    if state.final_bgr is None:
        state.final_bgr = state.m2_bgr if state.m2_bgr is not None else state.m1_bgr
    
    print(f"[{rid}] ══ PIPELINE DONE ════════════════════════════\n")
    return _build_response(request, state)

def _build_response(
    request:   Request,
    state:     PipelineState,
    error_msg: Optional[str] = None,
) -> Dict[str, Any]:
    urls = dls = views = []
    
    if state.final_bgr is not None:
        fn = make_result_filename()       # FIX 9+10: sanitized, no trailing quotes
        save_jpg(safe_path(fn), state.final_bgr)
        urls  = [ourl(request, fn)]
        dls   = [durl(request, fn)]
        views = [vurl(request, fn)]
    
    gi = state.garment_info
    qc = state.qc_report
    
    return {
        "request_id": state.rid,
        "success":    not state.failed,
        "warnings":   state.all_warns,
        "error":      {"message": error_msg} if error_msg else None,
        "pipeline": {
            "m0_garment": {
                "is_full_length": gi.is_full_length if gi else None,
                "has_palazzo":    gi.has_palazzo    if gi else None,
                "garment_type":   gi.garment_type   if gi else None,
                "cloth_type":     gi.cloth_type     if gi else None,
            },
            "m1_tryon": {
                "model":     state.m1_model,
                "latency_s": state.m1_latency,
                "source":    next(
                    (w for w in state.m1_warns if w.startswith(("FULL","MIDI"))), ""
                ),
            },
            "m2_extension": {
                "extended":  state.m2_extended,
                "latency_s": state.m2_latency,
            },
            "m3_autofix": {
                "strategy":   state.m3_strategy,
                "latency_s":  state.m3_latency,
                "has_issues": qc.has_issues      if qc else None,
                "issues":     qc.issues          if qc else None,
                "problem":    qc.problem_summary if qc else None,
            },
        },
        "output_urls":          urls,
        "output_download_urls": dls,
        "output_view_urls":     views,
    }

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — FASTAPI APP
# ══════════════════════════════════════════════════════════════════════════════
app = FastAPI(
    title="Virtual Try-On — 4-Module Agentic Pipeline",
    description=(
        "Supports midi dresses, full-length gowns, palazzo sets and more. "
        "M0 Inspect → M1 CatVTON → M2 Extend (gowns only) → M3 QC Fix"
    ),
)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

class RIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, req, call_next):
        req.state.request_id = uuid.uuid4().hex[:12]
        resp = await call_next(req)
        resp.headers["X-Request-Id"] = req.state.request_id
        return resp

app.add_middleware(RIDMiddleware)

def _err(rid: str, msg: str) -> JSONResponse:
    """FIX 8 — consistent error response shape across all endpoints."""
    return JSONResponse(status_code=200, content={
        "request_id": rid,
        "success":    False,
        "error":      {"message": msg},
        "output_urls": [], "output_download_urls": [], "output_view_urls": [],
    })

# ── Static endpoints ──────────────────────────────────────────────────────────
@app.get("/download/{filename}")
def download_output(filename: str):
    p = safe_path(filename)
    if not os.path.exists(p):
        return JSONResponse(404, content={"ok": False, "error": "not_found"})
    return FileResponse(
        path=p,
        media_type="image/jpeg",
        filename=os.path.basename(p),
        headers={"Cache-Control": "no-store"},
    )

@app.get("/view/{filename}")
def view_output(request: Request, filename: str):
    p = safe_path(filename)
    if not os.path.exists(p):
        return HTMLResponse("<h3>Not found</h3>", status_code=404)
    
    iu = ourl(request, os.path.basename(p))
    du = durl(request, os.path.basename(p))
    
    html = f"""<!doctype html>
<html><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Try-On Result</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Georgia',serif;background:#0c0c0c;color:#f0f0f0;
min-height:100vh;padding:40px 20px}}
.wrap{{max-width:860px;margin:0 auto}}
.tag{{display:inline-flex;gap:10px;margin-bottom:22px;font-size:11px;
letter-spacing:.15em;opacity:.5;text-transform:uppercase;
border:1px solid rgba(255,255,255,.12);padding:6px 14px;border-radius:20px}}
h2{{font-size:26px;font-weight:300;letter-spacing:.04em;margin-bottom:4px}}
.sub{{font-size:12px;opacity:.4;letter-spacing:.1em;margin-bottom:28px}}
img{{width:100%;border-radius:12px;box-shadow:0 30px 80px rgba(0,0,0,.7)}}
.btns{{margin-top:22px;display:flex;gap:14px}}
a.btn{{padding:13px 26px;border-radius:8px;text-decoration:none;
font-size:13px;letter-spacing:.06em;transition:.15s}}
a.p{{background:#fff;color:#000}}
a.s{{border:1px solid rgba(255,255,255,.2);color:#fff}}
a.s:hover{{background:rgba(255,255,255,.07)}}
</style>
</head>
<body><div class="wrap">
<div class="tag">
<span>M0 Inspect</span><span>·</span>
<span>M1 CatVTON</span><span>·</span>
<span>M2 Extend</span><span>·</span>
<span>M3 QC Fix</span>
</div>
<h2>Virtual Try-On Result</h2>
<p class="sub">4-Module Agentic Pipeline</p>
<img src="{iu}" alt="Try-On Result"/>
<div class="btns">
<a class="btn p" href="{du}">Download</a>
<a class="btn s" href="{iu}" target="_blank">Full Size</a>
</div>
</div></body></html>"""
    return HTMLResponse(html)

@app.get("/")
@app.get("/health")
def health():
    return {
        "ok":      True,
        "version": "complete-final",
        "ready": {
            "fal":    bool(FAL_KEY and fal_client),
            "gemini": bool(gemini_client),
            "rembg":  bool(rembg_remove),
        },
        "swap_logic": {
            "midi_dress":  "M0→False → M1 actress path → M2 skipped  → M3 QC",
            "full_length": "M0→True  → M1 flat-lay path → M2 extended → M3 QC",
        },
        "fixes": [
            "FIX1  extract_json()       strips Gemini ```json``` markdown",
            "FIX2  fal_upload() M3      HTTPS URLs not data URIs",
            "FIX3  no resolution_mode   invalid Kontext param removed",
            "FIX4  40px canvas sample   smoother extension seam colour",
            "FIX5  heuristic fallback   aspect ratio when M0 fails",
            "FIX6  seam blending        Gaussian fade at canvas join",
            "FIX7  final_bgr safety     always set even when skip_m3=1",
            "FIX8  consistent errors    all endpoints same JSON shape",
            "FIX9  sanitize_url()       strips trailing quotes (WinError 123)",
            "FIX10 sanitize_filename()  Windows-illegal chars removed",
            "FIX11 extra_outerwear QC   explicit coat/jacket removal prompt",
            "FIX12 watermark removal    flat-lay stock watermarks inpainted",
            "FIX13 ENHANCED EXTENSION   EXTEND_RATIO increased to 0.85",
            "FIX14 Better seam blending 50px blend zone",
            "FIX15 Stronger M2 prompts  Explicit FULL-LENGTH instructions",
        ],
        "endpoints": {
            "RECOMMENDED":  "POST /v1/tryon/actress-garment-to-user",
            "flatlay_only": "POST /v1/tryon/garment-to-user",
            "actress_only": "POST /v1/tryon/actress-to-user",
        },
    }

# ── Endpoint 1: flat-lay + user ───────────────────────────────────────────────
@app.post("/v1/tryon/garment-to-user")
async def garment_to_user(
    request:       Request,
    garment_image: UploadFile = File(..., description="Flat-lay product image"),
    user_image:    UploadFile = File(..., description="Target person photo"),
    style_hint:    str = Form(""),
    skip_m3:       int = Form(0),
):
    """Flat-lay garment → user. Full 4-module pipeline."""
    rid = getattr(request.state, "request_id", "")
    try:
        gb = read_bgr(await garment_image.read())
        ub = read_bgr(await user_image.read())
    except Exception as e:
        return _err(rid, f"Image read failed: {e}")
    
    return await run_pipeline(
        request, ub,
        garment_bgr=gb, actress_bgr=None,
        style_hint=style_hint, skip_m3=bool(skip_m3),
    )

# ── Endpoint 2: actress + user ────────────────────────────────────────────────
@app.post("/v1/tryon/actress-to-user")
async def actress_to_user(
    request:       Request,
    actress_image: UploadFile = File(..., description="Person wearing the garment"),
    user_image:    UploadFile = File(..., description="Target person photo"),
    style_hint:    str = Form(""),
    skip_m3:       int = Form(0),
):
    """Actress reference → user. Less accurate for full-length without flat-lay."""
    rid = getattr(request.state, "request_id", "")
    try:
        ab = read_bgr(await actress_image.read())
        ub = read_bgr(await user_image.read())
    except Exception as e:
        return _err(rid, f"Image read failed: {e}")
    
    return await run_pipeline(
        request, ub,
        garment_bgr=None, actress_bgr=ab,
        style_hint=style_hint, skip_m3=bool(skip_m3),
    )

# ── Endpoint 3: actress + flat-lay + user  ★ RECOMMENDED ─────────────────────
@app.post("/v1/tryon/actress-garment-to-user")
async def actress_garment_to_user(
    request:       Request,
    actress_image: UploadFile = File(...,
                                     description="Model wearing the garment (reference photo)"),
    garment_image: UploadFile = File(...,
                                     description="Product flat-lay image"),
    user_image:    UploadFile = File(...,
                                     description="Target person to dress"),
    style_hint:    str = Form("",
                               description="Optional hint e.g. 'camel midi wrap dress double-breasted'"),
    skip_m3:       int = Form(0,
                               description="1 = skip M3 QC (faster, less quality assurance)"),
):
    """
    RECOMMENDED — actress + flat-lay + user.
    
    Midi dress (your current images):
      M0: Gemini → is_full_length=False
      M1: preprocessed actress used as CatVTON garment input
      M2: SKIPPED
      M3: QC fixes extra_outerwear, seam_bleed, color_mismatch etc.
    
    Full-length gown:
      M0: Gemini → is_full_length=True
      M1: flat-lay used as CatVTON garment input
      M2: Flux Kontext extends to floor-length
      M3: QC fixes artefacts
    """
    rid = getattr(request.state, "request_id", "")
    try:
        ab = read_bgr(await actress_image.read())
        gb = read_bgr(await garment_image.read())
        ub = read_bgr(await user_image.read())
    except Exception as e:
        return _err(rid, f"Image read failed: {e}")   # FIX 8
    
    return await run_pipeline(
        request, ub,
        garment_bgr=gb, actress_bgr=ab,
        style_hint=style_hint, skip_m3=bool(skip_m3),
    )

# ── Global exception handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_handler(request: Request, exc: Exception):
    rid = getattr(request.state, "request_id", "")
    print(f"[{rid}] Unhandled: {type(exc).__name__}: {exc}\n{traceback.format_exc()}")
    return JSONResponse(status_code=200, content={
        "request_id": rid,
        "success":    False,
        "error":      {"type": type(exc).__name__, "message": "Unexpected server error"},
        "output_urls": [], "output_download_urls": [], "output_view_urls": [],
    })

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\nStarting on http://0.0.0.0:{PORT}")
    print(f"Swagger docs: http://localhost:{PORT}/docs\n")
    uvicorn.run(
        "ai_agent_pipeline:app",
        host="0.0.0.0",
        port=PORT,
        reload=True,
        log_level=LOG_LEVEL,
    )
