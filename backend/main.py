import sys, os, io, mimetypes
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_api import MultiModelRainPredictor

app = FastAPI(title="Rain Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = MultiModelRainPredictor()
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
    pressure: float = Form(1013.25),
    district: str = Form("Dhaka"),
    hour: int = Form(12),
    month: int = Form(6),
):
    global last_prediction
    image_bytes = await image.read()
    weather = {
        "temperature": temperature,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "pressure": pressure,
        "district": district,
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
    ax1.text(0.02, 1.06, "Model Contributions",
             fontsize=10, fontweight="bold", color="#475569", transform=ax1.transAxes)
    contrib = [
        ("Image model", r.get("image_probability", 0.0),
         r.get("ensemble_weight_image", 1.0)),
        ("Weather model", r.get("weather_probability", 0.0),
         r.get("ensemble_weight_weather", 0.0)),
    ]
    for i, (name, prob, weight) in enumerate(contrib):
        y = 0.98 - i * 0.13
        ax1.text(0.02, y, f"{name} ({weight*100:.0f}%)", fontsize=9,
                 color="#64748b", transform=ax1.transAxes)
        ax1.barh([y - 0.015], [prob], height=0.055, left=0.02,
                 color=["#8b5cf6", "#0ea5e9"][i], transform=ax1.transAxes)
        ax1.text(0.02 + prob, y - 0.015, f"{prob*100:.0f}%", fontsize=8,
                 va="center", transform=ax1.transAxes, color="#334155")

    ax2.axis("off")
    ax2.set_title("Weather Conditions", fontsize=13, fontweight="bold")
    weather = r.get("weather", {})
    items = [
        ("Cloud Coverage", f"{r['cloud_coverage']*100:.1f}%"),
        ("Temperature", f"{weather.get('temperature', '-')}\u00b0C"),
        ("Humidity", f"{weather.get('humidity', '-')}%"),
        ("Wind Speed", f"{weather.get('wind_speed', '-')} km/h"),
        ("Pressure", f"{weather.get('pressure', '-')} hPa"),
        ("District", f"{weather.get('district', '-')}"),
        ("Hour", f"{weather.get('hour', '-')}:00"),
        ("Month", f"{weather.get('month', '-')}"),
    ]
    for i, (k, v) in enumerate(items):
        ax2.text(0.1, 0.85 - i * 0.1, k, fontsize=10, color="#64748b",
                 transform=ax2.transAxes)
        ax2.text(0.9, 0.85 - i * 0.1, v, fontsize=10, fontweight="bold",
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
