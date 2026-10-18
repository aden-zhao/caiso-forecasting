"""Download CAISO load from OASIS: actuals and CAISO's own forecasts.

Writes data/oasis_<run>.parquet with columns
    time_utc (interval start), tac, opr_dt, opr_hr, mw
for run in ACTUAL, DAM (day-ahead), 2DA, 7DA.
"""
import io
import re
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

URL = "https://oasis.caiso.com/oasisapi/SingleZip"
TACS = ["CA ISO-TAC", "PGE-TAC", "SCE-TAC", "SDGE-TAC", "VEA-TAC", "MWD-TAC"]
OUT = Path(__file__).parent / "data"


def fetch_month(run: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    params = {
        "queryname": "SLD_FCST",
        "market_run_id": run,
        "startdatetime": start.strftime("%Y%m%dT%H:%M-0000"),
        "enddatetime": end.strftime("%Y%m%dT%H:%M-0000"),
        "version": 1,
        "resultformat": 6,
    }
    for attempt in range(6):
        r = requests.get(URL, params=params, timeout=120)
        try:
            z = zipfile.ZipFile(io.BytesIO(r.content))
            name = z.namelist()[0]
            if name.endswith(".csv"):
                return pd.read_csv(z.open(name))
            # an .xml inside the zip is an error message (usually rate limiting)
            msg = z.read(name).decode()
            print("  oasis error:", re.findall(r"<m:ERR_DESC>(.*?)</m:", msg), file=sys.stderr)
        except zipfile.BadZipFile:
            print("  bad zip, status", r.status_code, file=sys.stderr)
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"failed {run} {start}")


def main(run: str, first: str = "2020-01-01", last: str = "2026-10-16") -> None:
    OUT.mkdir(exist_ok=True)
    # OASIS caps queries at 31 days (and miscounts around DST), so use 20-day
    # chunks. Each chunk is cached so an interrupted download resumes.
    raw = OUT / "oasis_raw"
    raw.mkdir(exist_ok=True)
    starts = pd.date_range(first, last, freq="20D", inclusive="left") + pd.Timedelta(hours=8)
    frames = []
    for s in starts:
        path = raw / f"{run}_{s:%Y%m%d}.parquet"
        if not path.exists():
            e = min(s + pd.Timedelta(days=20), pd.Timestamp(last) + pd.Timedelta(hours=8))
            df = fetch_month(run, s, e)
            df[df.TAC_AREA_NAME.isin(TACS)].to_parquet(path)
            print(run, s.date(), flush=True)
            time.sleep(6)  # OASIS rate limit is roughly one request per 5 s
        frames.append(pd.read_parquet(path))
    df = pd.concat(frames)
    out = pd.DataFrame({
        "time_utc": pd.to_datetime(df.INTERVALSTARTTIME_GMT, utc=True),
        "tac": df.TAC_AREA_NAME.values,
        "opr_dt": df.OPR_DT.values,
        "opr_hr": df.OPR_HR.values,
        "mw": df.MW.values,
    }).drop_duplicates(["time_utc", "tac"]).sort_values(["tac", "time_utc"])
    out.to_parquet(OUT / f"oasis_{run}.parquet", index=False)
    print("wrote", len(out), "rows")


if __name__ == "__main__":
    main(*sys.argv[1:])
