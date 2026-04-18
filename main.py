import io
import time
import base64
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from prometheus_fastapi_instrumentator import Instrumentator

from gradcam_module.inference import GradCAMInference
from report.pdf_generator import build_pdf
from storage.minio_client import upload_xray, upload_report, ensure_buckets
from mlflow_module.tracker import setup_mlflow, log_prediction
from security.auth import get_current_user, authenticate_user, create_access_token
from security.rate_limiter import limiter

# ── Startup ─────────────────────────────────────────
pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    print("⏳ Loading model...")
    pipeline = GradCAMInference()
    print("✅ Model ready")

    # MinIO buckets
    try:
        ensure_buckets()
        print("✅ MinIO buckets ready")
    except Exception as e:
        print(f"⚠️  MinIO not available: {e} — continuing without storage")

    # MLflow
    try:
        setup_mlflow()
    except Exception as e:
        print(f"⚠️  MLflow not available: {e} — continuing without tracking")

    yield
    print("🛑 Shutting down")

# ── App ─────────────────────────────────────────────
app = FastAPI(
    title="Medical AI — Chest X-Ray API",
    description="DenseNet121 + Grad-CAM for 14 pulmonary pathologies",
    version="1.0.0",
    lifespan=lifespan
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda req, exc: JSONResponse(
    status_code=429,
    content={"detail": "Rate limit exceeded — max 10 requests/minute"}
))
app.add_middleware(SlowAPIMiddleware)

Instrumentator().instrument(app).expose(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Routes ───────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "densenet121-res224-nih",
        "pathologies": 14,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/login")
def login(username: str, password: str):
    """Get JWT token"""
    user = authenticate_user(username, password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": user["username"], "role": user["role"]})
    return {
        "access_token": token,
        "token_type": "bearer",
        "role": user["role"]
    }

@app.post("/predict")
@limiter.limit("10/minute")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    start = time.time()
    try:
        results = pipeline.predict(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    elapsed = round(time.time() - start, 2)

    sorted_results = dict(
        sorted(results.items(), key=lambda x: x[1]["score"], reverse=True)
    )

    # Save X-ray to MinIO (non-blocking)
    try:
        obj_name = upload_xray(image_bytes, file.filename)
        print(f"📦 X-ray saved to MinIO: {obj_name}")
    except Exception as e:
        print(f"⚠️  MinIO upload skipped: {e}")
        
    # Log to MLflow
    try:
        log_prediction(
            filename       = file.filename,
            predictions    = sorted_results,
            inference_time = elapsed,
        )
    except Exception as e:
        print(f"⚠️  MLflow logging skipped: {e}")

    return {
        "filename": file.filename,
        "inference_time_sec": elapsed,
        "predictions": sorted_results
    }


@app.post("/report")
async def generate_report(
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    try:
        pdf_bytes = build_pdf(
            patient_name   = body.get("name",     "N/A"),
            patient_age    = str(body.get("age",  "N/A")),
            patient_gender = body.get("gender",   "N/A"),
            patient_id     = body.get("pid",      "N/A"),
            doctor         = body.get("doctor",   "N/A"),
            notes          = body.get("notes",    ""),
            filename       = body.get("filename", "unknown"),
            predictions    = body.get("predictions", {}),
            orig_b64       = body.get("orig_b64", ""),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Save PDF to MinIO (non-blocking)
    try:
        patient_name = body.get("name", "unknown")
        obj_name = upload_report(pdf_bytes, patient_name)
        print(f"📦 Report saved to MinIO: {obj_name}")
    except Exception as e:
        print(f"⚠️  MinIO report upload skipped: {e}")

    patient_name = body.get("name", "report").replace(" ", "_")
    date_str     = datetime.now().strftime("%Y%m%d_%H%M")
    fname        = f"xray_report_{patient_name}_{date_str}.pdf"

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={fname}"}
    )