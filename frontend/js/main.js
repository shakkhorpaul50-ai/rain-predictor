var API_BASE = window.API_BASE || "";
function getElements() {
    return {
        imageInput: document.getElementById("imageInput"),
        preview: document.getElementById("preview"),
        uploadArea: document.getElementById("uploadArea"),
        temperature: document.getElementById("temperature"),
        humidity: document.getElementById("humidity"),
        windSpeed: document.getElementById("windSpeed"),
        hour: document.getElementById("hour"),
        month: document.getElementById("month"),
        predictBtn: document.getElementById("predictBtn"),
        downloadBtn: document.getElementById("downloadBtn"),
        loading: document.getElementById("loading"),
        results: document.getElementById("results"),
        resultLabel: document.getElementById("resultLabel"),
        rainProb: document.getElementById("rainProb"),
        noRainProb: document.getElementById("noRainProb"),
        cloudCoverage: document.getElementById("cloudCoverage"),
        confidenceBar: document.getElementById("confidenceBar"),
        cloudCoverageBar: document.getElementById("cloudCoverageBar"),
        errorMsg: document.getElementById("errorMsg"),
    };
}
function setupImageUpload(el) {
    el.uploadArea.addEventListener("dragover", (e) => {
        e.preventDefault();
        el.uploadArea.classList.add("dragover");
    });
    el.uploadArea.addEventListener("dragleave", () => {
        el.uploadArea.classList.remove("dragover");
    });
    el.uploadArea.addEventListener("drop", (e) => {
        e.preventDefault();
        el.uploadArea.classList.remove("dragover");
        const file = e.dataTransfer?.files?.[0];
        if (file)
            handleFile(file, el);
    });
    el.imageInput.addEventListener("change", () => {
        const file = el.imageInput.files?.[0];
        if (file)
            handleFile(file, el);
    });
}
function handleFile(file, el) {
    if (!file.type.startsWith("image/")) {
        showError("Please upload a valid image file.", el);
        return;
    }
    hideError(el);
    const reader = new FileReader();
    reader.onload = () => {
        el.preview.src = reader.result;
        el.preview.classList.add("visible");
        el.uploadArea.classList.add("has-image");
    };
    reader.readAsDataURL(file);
}
function getWeatherData(el) {
    return {
        temperature: parseFloat(el.temperature.value) || 20,
        humidity: parseFloat(el.humidity.value) || 60,
        wind_speed: parseFloat(el.windSpeed.value) || 5,
        hour: parseInt(el.hour.value) || 12,
        month: parseInt(el.month.value) || 6,
    };
}
async function predict(el) {
    const file = el.imageInput.files?.[0];
    if (!file) {
        showError("Please select a cloud image first.", el);
        return;
    }
    hideError(el);
    el.results.classList.remove("visible");
    el.loading.classList.add("visible");
    el.predictBtn.disabled = true;
    try {
        const weather = getWeatherData(el);
        const formData = new FormData();
        formData.append("image", file);
        formData.append("temperature", String(weather.temperature));
        formData.append("humidity", String(weather.humidity));
        formData.append("wind_speed", String(weather.wind_speed));
        formData.append("hour", String(weather.hour));
        formData.append("month", String(weather.month));
        const res = await fetch(`${API_BASE}/api/predict`, { method: "POST", body: formData });
        if (!res.ok)
            throw new Error(`Server error: ${res.status}`);
        const result = await res.json();
        showResult(result, el);
    }
    catch (err) {
        const msg = err instanceof Error ? err.message : "Prediction failed. Is the server running?";
        showError(msg, el);
    }
    finally {
        el.loading.classList.remove("visible");
        el.predictBtn.disabled = false;
    }
}
function showResult(r, el) {
    const isRain = r.prediction === 1;
    el.resultLabel.textContent = r.label;
    el.resultLabel.className = `result-badge ${isRain ? "rain" : "no-rain"}`;
    el.rainProb.textContent = `${(r.probability_rain * 100).toFixed(1)}%`;
    el.noRainProb.textContent = `${(r.probability_no_rain * 100).toFixed(1)}%`;
    el.cloudCoverage.textContent = `${(r.cloud_coverage * 100).toFixed(1)}%`;
    el.confidenceBar.style.width = `${r.probability_rain * 100}%`;
    el.cloudCoverageBar.style.width = `${r.cloud_coverage * 100}%`;
    el.results.classList.add("visible");
    el.downloadBtn.style.display = "block";
}
function downloadReport() {
    const a = document.createElement("a");
    a.href = `${API_BASE}/api/report`;
    a.download = "rain_prediction_report.png";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}
function showError(msg, el) {
    el.errorMsg.textContent = msg;
    el.errorMsg.classList.add("visible");
}
function hideError(el) {
    el.errorMsg.classList.remove("visible");
    el.errorMsg.textContent = "";
}
function init() {
    const el = getElements();
    setupImageUpload(el);
    el.predictBtn.addEventListener("click", () => predict(el));
    el.downloadBtn.addEventListener("click", downloadReport);
}
document.addEventListener("DOMContentLoaded", init);
