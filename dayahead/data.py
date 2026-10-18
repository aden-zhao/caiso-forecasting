"""Assemble the day-ahead forecasting table.

Setup: at 09:00 Pacific on day D-1 we forecast every hour of day D (Pacific
operating day, 23-25 hours). Everything in a row must be known at that time:
  - load actuals through 08:00 on D-1
  - weather forecasts from model runs that were out by then (see weather_at_issue)
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
TZ = "America/Los_Angeles"
ISSUE_HOUR = 9          # local hour on D-1 when the forecast is made
LAST_KNOWN_HOUR = 7     # last load interval (start hour, local) known at issue

# Load centers inside the CAISO footprint: (lat, lon, rough population served in millions).
# The population weights the CAISO-wide weather averages.
CITIES = {
    "los_angeles":  (34.05, -118.24, 4.0),
    "riverside":    (33.95, -117.40, 2.5),
    "santa_ana":    (33.75, -117.87, 3.0),
    "palm_springs": (33.83, -116.55, 0.5),
    "san_diego":    (32.72, -117.16, 3.3),
    "san_jose":     (37.34, -121.89, 2.0),
    "oakland":      (37.80, -122.27, 2.5),
    "sacramento":   (38.58, -121.49, 1.5),
    "fresno":       (36.74, -119.79, 1.0),
    "bakersfield":  (35.37, -119.02, 0.9),
}
CITY_WEIGHT = {city: w for city, (_, _, w) in CITIES.items()}
WEATHER_VARS = ["temperature_2m", "dew_point_2m", "relative_humidity_2m",
                "shortwave_radiation", "cloud_cover", "wind_speed_10m"]

# When an archived weather forecast was usable (see weather_at_issue)
DAY1_MIN_LEAD = pd.Timedelta(hours=23)     # day1 values come from runs >= 23 h before valid time
PUBLISH_DELAY = pd.Timedelta(hours=6)      # a run is fully published ~6 h after its init time


def oasis(run: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA / f"oasis_{run}.parquet")
    return df.pivot(index="time_utc", columns="tac", values="mw").sort_index()


@lru_cache
def hourly() -> pd.DataFrame:
    """One row per UTC hour: load, CAISO's forecasts, and local calendar keys."""
    act = oasis("ACTUAL")
    df = pd.DataFrame(index=pd.date_range(act.index.min(), act.index.max(), freq="h"))
    # Target: Outlook "Current demand", which like CAISO's forecast excludes
    # battery charging. OASIS actuals include it (kept as load_oasis, for the README figure).
    df["load"] = pd.read_parquet(DATA / "outlook_demand.parquet")["Current demand"]
    df["load_oasis"] = act["CA ISO-TAC"]
    fuel = pd.read_parquet(DATA / "outlook_fuel.parquet")
    df["batteries"] = fuel["Batteries"]
    df["solar"] = fuel["Solar"]
    df["caiso_dam"] = oasis("DAM")["CA ISO-TAC"]     # CAISO's day-ahead forecast (the benchmark)
    df["caiso_2da"] = oasis("2DA")["CA ISO-TAC"]     # two-day-ahead, published on D-2
    local = df.index.tz_convert(TZ)
    df["day"] = pd.to_datetime(local.date)
    df["hour"] = local.hour
    # issue time for the operating day this hour belongs to
    issue_local = (df["day"] - pd.Timedelta(days=1)).dt.tz_localize(TZ) + pd.Timedelta(hours=ISSUE_HOUR)
    df["issue_utc"] = issue_local.dt.tz_convert("UTC")
    return df


def weather_at_issue(model: str, policy: str = "legit") -> pd.DataFrame:
    """Weather forecast for each valid hour, as known at that day's issue time.

    Open-Meteo's `_previous_day1` comes from the run initialised 24-29 h before
    the valid time (GFS) or 23-28 h (ECMWF); `_previous_day2` from 47-54 h before.
    Runs are fully published about 6 h after init, so day1 is usable when
    valid - DAY1_MIN_LEAD + PUBLISH_DELAY <= issue (the first few hours of D),
    and day2 for every hour of D. We checked this mapping against Open-Meteo's
    single-run archive for Sep 2026.

    policy='day1' uses day1 everywhere: fresher than we could really have, an
    upper bound for ablations only.
    """
    wx = pd.read_parquet(DATA / f"wx_fcst_{model}.parquet")
    wx["lead"] = wx.variable.str[-1].astype(int)
    wx["var"] = wx.variable.str.replace(r"_previous_day\d$", "", regex=True)
    wide = wx.pivot_table(index="time_utc", columns=["city", "var", "lead"], values="value")
    issue = hourly()["issue_utc"].reindex(wide.index)
    day1_ok = (wide.index - DAY1_MIN_LEAD + PUBLISH_DELAY) <= issue
    if policy == "day1":
        day1_ok = np.ones(len(wide), dtype=bool)
    out = {}
    for city, var in {(c, v) for c, v, _ in wide.columns}:   # prefer day1, fall back to day2
        d1 = wide.get((city, var, 1))
        d2 = wide.get((city, var, 2))
        col = d2.copy() if d2 is not None else pd.Series(np.nan, index=wide.index)
        if d1 is not None:
            col = col.where(~day1_ok | d1.isna(), d1)
        out[f"{var}|{city}"] = col
    return pd.DataFrame(out).sort_index(axis=1)


def observed_weather() -> pd.DataFrame:
    wx = pd.read_parquet(DATA / "wx_obs.parquet")
    wide = wx.pivot_table(index="time_utc", columns=["variable", "city"], values="value")
    wide.columns = [f"{v}|{c}" for v, c in wide.columns]
    return wide


def weighted(wide: pd.DataFrame, var: str) -> pd.Series:
    """Population-weighted mean of one variable over the cities that have a value."""
    cols = {c: CITY_WEIGHT[c.split("|")[1]] for c in wide.columns if c.split("|")[0] == var}
    vals = wide[list(cols)]
    w = pd.Series(cols)
    return (vals * w).sum(axis=1, min_count=1) / vals.notna().mul(w).sum(axis=1)
