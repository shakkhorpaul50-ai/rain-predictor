import os, time, csv, sys
import requests

OUT_DIR = os.path.join("datasets", "weather")
OUT_CSV = os.path.join(OUT_DIR, "bangladesh_hourly_weather.csv")
OUT_DISTRICTS = os.path.join(OUT_DIR, "districts.csv")
START_DATE = "2023-01-01"
END_DATE = "2024-12-31"
RAIN_LOOKAHEAD_HOURS = 3
RAIN_THRESHOLD_MM = 0.2

DISTRICTS = [
    ("Dhaka", 23.8103, 90.4125), ("Gazipur", 24.0023, 90.4264),
    ("Kishoreganj", 24.4449, 90.7766), ("Manikganj", 23.8644, 90.0045),
    ("Munshiganj", 23.5422, 90.5305), ("Narayanganj", 23.6238, 90.5000),
    ("Narsingdi", 23.9322, 90.7151), ("Tangail", 24.2513, 89.9167),
    ("Faridpur", 23.6070, 89.8426), ("Gopalganj", 23.0050, 89.8267),
    ("Madaripur", 23.1640, 90.1895), ("Rajbari", 23.7574, 89.6460),
    ("Shariatpur", 23.2423, 90.4348),
    ("Chattogram", 22.3569, 91.7832), ("Cox's Bazar", 21.4272, 92.0058),
    ("Bandarban", 22.1953, 92.2184), ("Khagrachhari", 23.1193, 91.9847),
    ("Rangamati", 22.6478, 92.1571), ("Feni", 23.0159, 91.3976),
    ("Lakshmipur", 22.9447, 90.8277), ("Noakhali", 22.8696, 91.0994),
    ("Chandpur", 23.2331, 90.6634), ("Comilla", 23.4607, 91.1809),
    ("Brahmanbaria", 23.9608, 91.1115),
    ("Rajshahi", 24.3745, 88.6042), ("Bogura", 24.8465, 89.3726),
    ("Chapainawabganj", 24.5965, 88.2775), ("Joypurhat", 25.0968, 89.0220),
    ("Naogaon", 24.7936, 88.9442), ("Natore", 24.4206, 89.0000),
    ("Pabna", 24.0064, 89.2372), ("Sirajganj", 24.4534, 89.7007),
    ("Khulna", 22.8158, 89.5685), ("Bagerhat", 22.6602, 89.7895),
    ("Chuadanga", 23.6401, 88.8256), ("Jashore", 23.1667, 89.2083),
    ("Jhenaidah", 23.5528, 89.1790), ("Kushtia", 23.9013, 89.1205),
    ("Magura", 23.4870, 89.4199), ("Meherpur", 23.7622, 88.6317),
    ("Narail", 23.1655, 89.4997), ("Satkhira", 22.7185, 89.0705),
    ("Barishal", 22.7010, 90.3535), ("Barguna", 22.1507, 90.1262),
    ("Bhola", 22.6859, 90.6492), ("Jhalokathi", 22.6406, 90.1987),
    ("Patuakhali", 22.3596, 90.3290), ("Pirojpur", 22.5781, 89.9729),
    ("Sylhet", 24.8949, 91.8687), ("Habiganj", 24.3745, 91.4156),
    ("Moulvibazar", 24.4829, 91.7774), ("Sunamganj", 25.0658, 91.4010),
    ("Rangpur", 25.7468, 89.2508), ("Dinajpur", 25.6217, 88.6354),
    ("Gaibandha", 25.3288, 89.5438), ("Kurigram", 25.8054, 89.6360),
    ("Lalmonirhat", 25.9170, 89.4553), ("Nilphamari", 25.9312, 88.8563),
    ("Panchagarh", 26.3411, 88.5541), ("Thakurgaon", 26.0337, 88.4618),
    ("Mymensingh", 24.7471, 90.4203), ("Jamalpur", 24.9375, 89.9372),
    ("Netrokona", 24.8700, 90.7287), ("Sherpur", 25.0200, 90.0175),
]

HOURLY_VARS = [
    "temperature_2m", "relative_humidity_2m",
    "wind_speed_10m", "surface_pressure", "precipitation",
]

API_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_district(name, lat, lon, retries=3):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "Asia/Dhaka",
    }
    for attempt in range(retries):
        try:
            r = requests.get(API_URL, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"    attempt {attempt + 1} failed: {e}", flush=True)
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
    return None


def write_districts():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_DISTRICTS, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["district", "latitude", "longitude"])
        for name, lat, lon in DISTRICTS:
            w.writerow([name, lat, lon])


def main():
    write_districts()
    print(f"Fetching hourly weather for {len(DISTRICTS)} districts "
          f"({START_DATE} to {END_DATE})...", flush=True)

    header = ["district", "datetime", "hour", "month"] + HOURLY_VARS + ["rain_next_3h"]
    first = True
    total_rows = 0
    for i, (name, lat, lon) in enumerate(DISTRICTS):
        print(f"[{i + 1}/{len(DISTRICTS)}] {name} ({lat:.2f}, {lon:.2f})", flush=True)
        data = fetch_district(name, lat, lon)
        if data is None or "hourly" not in data:
            print(f"    FAILED, skipping {name}", flush=True)
            continue
        h = data["hourly"]
        times = h.get("time", [])
        cols = {v: h.get(v, []) for v in HOURLY_VARS}
        n = len(times)
        if n == 0:
            print("    no data, skipping", flush=True)
            continue

        import datetime
        rows = []
        for j in range(n):
            hour = int(times[j][11:13])
            month = int(times[j][5:7])
            precip = cols["precipitation"][j] or 0.0
            lookahead = sum(
                (cols["precipitation"][k] or 0.0)
                for k in range(j + 1, min(j + 1 + RAIN_LOOKAHEAD_HOURS, n))
            )
            if j + RAIN_LOOKAHEAD_HOURS >= n:
                continue
            rows.append([name, times[j], hour, month] +
                        [(cols[v][j] if cols[v][j] is not None else "") for v in HOURLY_VARS] +
                        [1 if (precip + lookahead) > RAIN_THRESHOLD_MM else 0])

        with open(OUT_CSV, "a" if not first else "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if first:
                w.writerow(header)
                first = False
            w.writerows(rows)
        total_rows += len(rows)
        print(f"    {len(rows)} rows (total {total_rows})", flush=True)
        time.sleep(1)

    print(f"\nDone. Saved {total_rows} rows to {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
