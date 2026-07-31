var API_BASE = window.API_BASE || "";
function getElements() {
    return {
        imageInput: document.getElementById("imageInput"),
        preview: document.getElementById("preview"),
        uploadArea: document.getElementById("uploadArea"),
        temperature: document.getElementById("temperature"),
        humidity: document.getElementById("humidity"),
        windSpeed: document.getElementById("windSpeed"),
        pressure: document.getElementById("pressure"),
        district: document.getElementById("district"),
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
        imageProb: document.getElementById("imageProb"),
        weatherProb: document.getElementById("weatherProb"),
        imageBar: document.getElementById("imageBar"),
        weatherBar: document.getElementById("weatherBar"),
        imageWeight: document.getElementById("imageWeight"),
        weatherWeight: document.getElementById("weatherWeight"),
        imageRaw: document.getElementById("imageRaw"),
        weatherRaw: document.getElementById("weatherRaw"),
        blendFormula: document.getElementById("blendFormula"),
        hint: document.getElementById("hint"),
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
        pressure: parseFloat(el.pressure.value) || 1013,
        district: el.district.value || "Dhaka",
        hour: parseInt(el.hour.value) || 12,
        month: parseInt(el.month.value) || 6,
    };
}
function validateWeather(el) {
    const h = parseFloat(el.humidity.value);
    const p = parseFloat(el.pressure.value);
    if (h < 0 || h > 100) {
        showError("Humidity must be between 0 and 100%.", el);
        return false;
    }
    if (p < 950 || p > 1060) {
        showError("Pressure looks unusual (typical range 950-1060 hPa).", el);
        return false;
    }
    return true;
}
async function predict(el) {
    const file = el.imageInput.files?.[0];
    if (!file) {
        showError("Please select a cloud image first.", el);
        return;
    }
    hideError(el);
    if (!validateWeather(el))
        return;
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
        formData.append("pressure", String(weather.pressure));
        formData.append("district", String(weather.district));
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
    const imgP = r.image_probability || 0;
    const weaP = r.weather_probability;
    const wImg = r.ensemble_weight_image || 0.6;
    const wWea = r.ensemble_weight_weather || 0.4;
    const imgContrib = imgP * wImg;
    el.imageWeight.textContent = `(${(wImg * 100).toFixed(0)}%)`;
    el.weatherWeight.textContent = `(${(wWea * 100).toFixed(0)}%)`;
    el.imageProb.textContent = `${(imgContrib * 100).toFixed(1)}%`;
    el.imageRaw.textContent = `raw ${(imgP * 100).toFixed(1)}%`;
    el.imageBar.style.width = `${imgContrib * 100}%`;
    if (weaP === null || weaP === undefined) {
        el.weatherProb.textContent = "N/A";
        el.weatherBar.style.width = "0%";
        el.weatherRaw.textContent = "";
    } else {
        const weaContrib = weaP * wWea;
        el.weatherProb.textContent = `${(weaContrib * 100).toFixed(1)}%`;
        el.weatherRaw.textContent = `raw ${(weaP * 100).toFixed(1)}%`;
        el.weatherBar.style.width = `${weaContrib * 100}%`;
    }
    if (weaP === null || weaP === undefined) {
        el.blendFormula.textContent = "Weather model unavailable - result based on image model only.";
    } else {
        el.blendFormula.textContent =
            `Final = ${(wImg * 100).toFixed(0)}% x ${(imgP * 100).toFixed(1)}% + ` +
            `${(wWea * 100).toFixed(0)}% x ${(weaP * 100).toFixed(1)}% = ${(r.probability_rain * 100).toFixed(1)}%`;
    }
    if (r.hint) {
        el.hint.textContent = r.hint;
        el.hint.classList.add("visible");
    }
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
