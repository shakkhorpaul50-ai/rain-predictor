import sys, os, io, mimetypes
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_api import RainPredictor

app = FastAPI(title="Rain Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = RainPredictor()
FRONTEND = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

last_prediction = {}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/predict")
async def predict(
    image: UploadFile = File(...),
    temperature: float = Form(20.0),
    humidity: float = Form(60.0),
    wind_speed: float = Form(5.0),
    hour: int = Form(12),
    month: int = Form(6),
):
    global last_prediction
    image_bytes = await image.read()
    weather = {
        "temperature": temperature,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "hour": hour,
        "month": month,
    }
    result = predictor.predict(image_bytes, weather)
    result["weather"] = weather
    result["filename"] = image.filename or "cloud_image"
    last_prediction = result
    resp = {k: v for k, v in result.items() if k != "weather"}
    return JSONResponse(content=resp)


@app.get("/api/report")
def download_report():
    if not last_prediction:
        return JSONResponse(status_code=400, content={"error": "No prediction available. Make a prediction first."})

    r = last_prediction
    is_rain = r["prediction"] == 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), facecolor="#f8faff")

    fig.suptitle("Rain Prediction Report", fontsize=16, fontweight="bold", y=0.98)

    ax1.axis("off")
    ax1.set_title("Prediction", fontsize=13, fontweight="bold")
    label_color = "#ef4444" if is_rain else "#22c55e"
    ax1.text(0.5, 0.8, r["label"], fontsize=28, fontweight="bold",
             color=label_color, ha="center", va="center", transform=ax1.transAxes)
    ax1.text(0.5, 0.55, f"{r['probability_rain']*100:.1f}% probability",
             fontsize=13, ha="center", va="center", transform=ax1.transAxes, color="#64748b")

    bars = ["Rain", "No Rain"]
    probs = [r["probability_rain"], r["probability_no_rain"]]
    colors = ["#ef4444", "#22c55e"]
    ax1.barh(bars, probs, color=colors, height=0.5)
    ax1.set_xlim(0, 1)
    ax1.set_xlabel("Probability")
    for i, v in enumerate(probs):
        ax1.text(v + 0.02, i, f"{v*100:.1f}%", va="center", fontsize=11)

    ax2.axis("off")
    ax2.set_title("Weather Conditions", fontsize=13, fontweight="bold")
    weather = r.get("weather", {})
    items = [
        ("Cloud Coverage", f"{r['cloud_coverage']*100:.1f}%"),
        ("Temperature", f"{weather.get('temperature', '-')}\u00b0C"),
        ("Humidity", f"{weather.get('humidity', '-')}%"),
        ("Wind Speed", f"{weather.get('wind_speed', '-')} km/h"),
        ("Hour", f"{weather.get('hour', '-')}:00"),
        ("Month", f"{weather.get('month', '-')}"),
    ]
    for i, (k, v) in enumerate(items):
        ax2.text(0.1, 0.85 - i * 0.12, k, fontsize=11, color="#64748b",
                 transform=ax2.transAxes)
        ax2.text(0.9, 0.85 - i * 0.12, v, fontsize=11, fontweight="bold",
                 ha="right", transform=ax2.transAxes, color="#1e293b")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/png",
                    headers={"Content-Disposition": "attachment; filename=rain_prediction_report.png"})


@app.get("/{path:path}")
async def serve_frontend(path: str):
    if not path:
        path = "index.html"
    file_path = os.path.normpath(os.path.join(FRONTEND, path))
    if not file_path.startswith(os.path.normpath(FRONTEND)):
        return JSONResponse(status_code=403, content={"error": "Forbidden"})
    if os.path.isfile(file_path):
        content_type, _ = mimetypes.guess_type(file_path)
        return FileResponse(file_path, media_type=content_type or "application/octet-stream")
    index_path = os.path.join(FRONTEND, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return JSONResponse(status_code=404, content={"error": "Not found"})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
