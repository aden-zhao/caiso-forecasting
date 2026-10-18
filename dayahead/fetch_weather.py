"""Download archived weather forecasts (and observed weather) for CAISO load centers.

Forecasts come from Open-Meteo's Previous Runs API: `<var>_previous_dayN` is what
the model predicted N days before the valid time. That is what a forecaster
actually had, unlike observed weather.

Writes data/wx_fcst_<model>.parquet and data/wx_obs.parquet, long format:
    time_utc, city, variable, value
"""
import sys
import time
from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).parent / "data"

# Load centers inside the CAISO footprint, with the TAC that serves them.
CITIES = {
    "los_angeles":  (34.05, -118.24, "SCE-TAC"),
    "riverside":    (33.95, -117.40, "SCE-TAC"),
    "santa_ana":    (33.75, -117.87, "SCE-TAC"),
    "palm_springs": (33.83, -116.55, "SCE-TAC"),
    "san_diego":    (32.72, -117.16, "SDGE-TAC"),
    "san_jose":     (37.34, -121.89, "PGE-TAC"),
    "oakland":      (37.80, -122.27, "PGE-TAC"),
    "sacramento":   (38.58, -121.49, "PGE-TAC"),
    "fresno":       (36.74, -119.79, "PGE-TAC"),
    "bakersfield":  (35.37, -119.02, "PGE-TAC"),
}
VARS = ["temperature_2m", "dew_point_2m", "relative_humidity_2m",
        "shortwave_radiation", "cloud_cover", "wind_speed_10m"]


def get(url: str, params: dict) -> dict:
    for attempt in range(20):
        r = requests.get(url, params=params, timeout=300)
        if r.status_code == 200:
            return r.json()
        print("  http", r.status_code, r.text[:200], file=sys.stderr, flush=True)
        # the free tier has minute, hour and day limits; wait out the hourly one
        time.sleep(600 if r.status_code == 429 else 30)
    raise RuntimeError(url)


def to_long(js: dict, city: str) -> pd.DataFrame:
    h = pd.DataFrame(js["hourly"])
    h["time_utc"] = pd.to_datetime(h.pop("time"), utc=True)
    long = h.melt("time_utc", var_name="variable", value_name="value")
    long["city"] = city
    return long.dropna()


def fetch_forecasts(model: str, start: str, end: str) -> None:
    hourly = [f"{v}_previous_day{d}" for v in VARS for d in (1, 2)]
    raw = OUT / "wx_raw"
    raw.mkdir(exist_ok=True)
    frames = []
    for city, (lat, lon, _) in CITIES.items():
        # one year per request keeps responses small; cache so reruns resume
        for y0 in range(int(start[:4]), int(end[:4]) + 1):
            path = raw / f"{model}_{city}_{y0}.parquet"
            if not path.exists():
                s, e = max(start, f"{y0}-01-01"), min(end, f"{y0}-12-31")
                js = get("https://previous-runs-api.open-meteo.com/v1/forecast", {
                    "latitude": lat, "longitude": lon, "start_date": s, "end_date": e,
                    "hourly": ",".join(hourly), "models": model, "timezone": "GMT"})
                to_long(js, city).to_parquet(path)
                print(model, city, y0, flush=True)
                time.sleep(2)
            frames.append(pd.read_parquet(path))
    pd.concat(frames).to_parquet(OUT / f"wx_fcst_{model}.parquet", index=False)


def fetch_observed(start: str, end: str) -> None:
    frames = []
    for city, (lat, lon, _) in CITIES.items():
        js = get("https://archive-api.open-meteo.com/v1/archive", {
            "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
            "hourly": ",".join(VARS), "timezone": "GMT"})
        frames.append(to_long(js, city))
        print("obs", city, len(frames[-1]), flush=True)
        time.sleep(5)
    pd.concat(frames).to_parquet(OUT / "wx_obs.parquet", index=False)


if __name__ == "__main__":
    what, start, end = sys.argv[1:4]
    OUT.mkdir(exist_ok=True)
    if what == "obs":
        fetch_observed(start, end)
    else:
        fetch_forecasts(what, start, end)
