"""Download CAISO Today's Outlook demand history (5-minute).

OASIS "actual load" includes battery charging, which CAISO's demand forecast
excludes; Outlook "Current demand" is on the same basis as the forecast.
Also grabs the fuel mix, whose Batteries column (negative = charging) explains the gap.

Writes data/outlook_demand.parquet and data/outlook_fuel.parquet (hourly means,
indexed by UTC interval start).
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).parent / "data"
RAW = OUT / "outlook_raw"
URL = "https://www.caiso.com/outlook/history/{d:%Y%m%d}/{what}.csv"


def fetch(day: pd.Timestamp, what: str) -> None:
    path = RAW / f"{what}_{day:%Y%m%d}.csv"
    if path.exists():
        return
    for attempt in range(5):
        try:
            r = requests.get(URL.format(d=day, what=what), timeout=60)
            if r.status_code == 200 and r.text.startswith("Time"):
                path.write_text(r.text)
                return
            if r.status_code == 404:
                return
        except requests.RequestException:
            pass
        time.sleep(5 * (attempt + 1))
    print("failed", what, day.date(), file=sys.stderr)


def hourly(what: str) -> pd.DataFrame:
    frames = []
    for path in sorted(RAW.glob(f"{what}_*.csv")):
        df = pd.read_csv(path)
        day = pd.Timestamp(path.stem.split("_")[1])
        # times are local clock times; on DST days the repeated/missing hour is ambiguous
        local = day + pd.to_timedelta(df.pop("Time") + ":00")
        df.index = local.dt.tz_localize("America/Los_Angeles", ambiguous="NaT", nonexistent="NaT")
        df = df[df.index.notna()]
        frames.append(df.apply(pd.to_numeric, errors="coerce"))
    df = pd.concat(frames)
    df.index = df.index.tz_convert("UTC")
    return df.resample("h").mean()


def main(first: str = "2020-01-01", last: str = "2026-10-15") -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    days = pd.date_range(first, last)
    jobs = [(d, w) for d in days for w in ("demand", "fuelsource")]
    with ThreadPoolExecutor(4) as pool:
        for i, _ in enumerate(pool.map(lambda j: fetch(*j), jobs)):
            if i % 200 == 0:
                print(i, "/", len(jobs), flush=True)
    hourly("demand").to_parquet(OUT / "outlook_demand.parquet")
    hourly("fuelsource").to_parquet(OUT / "outlook_fuel.parquet")
    print("done")


if __name__ == "__main__":
    main(*sys.argv[1:])
