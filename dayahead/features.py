"""Feature table: one row per target hour, using only information known at issue time.

Load features come from days D-8 .. D-2 and the morning of D-1; weather features
are forecasts as of the issue time (data.weather_at_issue).

These are the 50 features left after an ablation on 2024: the other 19 we had
(thermal lags, degree days, humidity/wind, ...) duplicated information that is
already here. The full 69-feature version is at git tag v1-full-features.
"""
import holidays
import numpy as np
import pandas as pd

from data import CITIES, CITY_WEIGHT, LAST_KNOWN_HOUR, TZ, hourly, weather_at_issue, weighted

ONE = pd.Timedelta(days=1)
TOTAL_WEIGHT = sum(CITY_WEIGHT.values())


def day_hour(s: pd.Series, h: pd.DataFrame) -> pd.DataFrame:
    """Pivot an hourly series to day x local-hour (fall-back hour averaged)."""
    return pd.DataFrame({"day": h.day, "hour": h.hour, "v": s}).pivot_table(
        index="day", columns="hour", values="v", aggfunc="mean")


def lookup(dh: pd.DataFrame, days: pd.Series, hours: pd.Series) -> np.ndarray:
    """Read a day x hour table at the given (day, hour) pairs."""
    stacked = dh.stack()
    idx = pd.MultiIndex.from_arrays([days, hours])
    return stacked.reindex(idx).to_numpy()


def load_features(h: pd.DataFrame, col: str = "load") -> pd.DataFrame:
    L = day_hour(h[col], h)
    D, hr = h.day, h.hour
    f = pd.DataFrame(index=h.index)
    # same hour 2-8 days ago: the last week's shape, incl. how deep the midday solar dip is lately
    for k in range(2, 9):
        f[f"lag{k}"] = lookup(L, D - k * ONE, hr)
    # yesterday's same hour is only known for the morning hours
    f["lag1"] = np.where(hr <= LAST_KNOWN_HOUR, lookup(L, D - ONE, hr), np.nan)
    morning = L.loc[:, :LAST_KNOWN_HOUR].mean(axis=1)
    daily_mean = L.mean(axis=1)
    f["d1_morning"] = morning.reindex(D - ONE).to_numpy()      # the latest known load level
    f["d1_last"] = L[LAST_KNOWN_HOUR].reindex(D - ONE).to_numpy()
    f["d2_morning"] = morning.reindex(D - 2 * ONE).to_numpy()
    f["d2_mean"] = daily_mean.reindex(D - 2 * ONE).to_numpy()
    f["d2_max"] = L.max(axis=1).reindex(D - 2 * ONE).to_numpy()
    f["d7_mean"] = daily_mean.reindex(D - 7 * ONE).to_numpy()
    f["level_change"] = f["d1_morning"] / f["d2_morning"]
    return f


def clear_sky(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Haurwitz clear-sky GHI (W/m2) at each city, from solar geometry alone."""
    t = index.tz_convert("UTC")
    doy = t.dayofyear.to_numpy()
    hours = t.hour.to_numpy() + 0.5            # middle of the interval
    decl = np.radians(23.44) * np.sin(2 * np.pi * (284 + doy) / 365)
    out = {}
    for city, (lat, lon, _) in CITIES.items():
        solar_time = hours + lon / 15.0
        ha = np.radians(15 * (solar_time - 12))
        phi = np.radians(lat)
        cosz = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(ha)
        cosz = np.clip(cosz, 0, None)
        out[city] = np.where(cosz > 0, 1098 * cosz * np.exp(-0.057 / np.maximum(cosz, 1e-3)), 0.0)
    return pd.DataFrame(out, index=index)


def weather_features(h: pd.DataFrame, wide: pd.DataFrame, prefix: str, variables: tuple,
                     per_city: bool, daily: bool) -> pd.DataFrame:
    """Hourly weather features from a wide `<var>|<city>` table.

    variables are averaged over cities (population-weighted); per_city adds each
    city's temperature; daily adds daily temperature summaries.
    """
    wide = wide.reindex(h.index)
    f = pd.DataFrame(index=h.index)
    if per_city:
        for c in wide.columns:
            if c.startswith("temperature_2m|"):
                f[f"{prefix}t_{c.split('|')[1]}"] = wide[c]
    for var in variables:
        f[f"{prefix}w_{var}"] = weighted(wide, var)
    if "cloud_cover" in variables:
        # cloud-attenuated clear-sky irradiance, weighted over cities: a rooftop-solar proxy
        # (Kasten-Czeplak cloud attenuation)
        cs = clear_sky(h.index)
        f[f"{prefix}solar_proxy"] = sum(
            CITY_WEIGHT[c] * cs[c] * (1 - 0.75 * (wide[f"cloud_cover|{c}"] / 100) ** 3.4)
            for c in cs.columns) / TOTAL_WEIGHT
    if not daily:
        return f
    t = f[f"{prefix}w_temperature_2m"]
    T = day_hour(t, h)
    for name, days in [("d0", h.day), ("d1", h.day - ONE), ("d2", h.day - 2 * ONE)]:
        f[f"{prefix}tmax_{name}"] = T.max(axis=1).reindex(days).to_numpy()
        f[f"{prefix}tmean_{name}"] = T.mean(axis=1).reindex(days).to_numpy()
    f[f"{prefix}tmin_d0"] = T.min(axis=1).reindex(h.day).to_numpy()
    # temperature at the same hour two days ago, to pair with the load lag
    f[f"{prefix}t_lag2"] = lookup(T, h.day - 2 * ONE, h.hour)
    f[f"{prefix}t_delta2"] = t - f[f"{prefix}t_lag2"]
    return f


def calendar_features(h: pd.DataFrame) -> pd.DataFrame:
    hol = holidays.country_holidays("US", subdiv="CA", years=range(2019, 2028))
    days = h.day.dt.date
    f = pd.DataFrame(index=h.index)
    f["hour"] = h.hour
    f["dow"] = h.day.dt.dayofweek
    f["holiday"] = days.map(lambda d: d in hol).astype(int)
    one = ONE.to_pytimedelta()
    f["holiday_adj"] = days.map(lambda d: (d - one) in hol or (d + one) in hol).astype(int)
    # was D-2 / D-7 (the load anchors) a holiday or weekend?
    f["lag2_offday"] = (h.day - 2 * ONE).dt.date.map(lambda d: d in hol or d.weekday() >= 5).astype(int)
    f["lag7_holiday"] = (h.day - 7 * ONE).dt.date.map(lambda d: d in hol).astype(int)
    doy = h.day.dt.dayofyear
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    f["dst"] = (h.index.tz_convert(TZ).strftime("%Z") == "PDT").astype(int)
    return f


def build() -> pd.DataFrame:
    """The full table: one row per UTC hour.

    Main weather is GFS (temperatures from 2021-03; cloud and sunshine from 2024),
    with temperature gaps filled from JMA. JMA's cloud cover is added as its own
    features because it is the only cloud forecast that goes back to 2020.
    """
    h = hourly()
    gfs = weather_at_issue("gfs_seamless")
    jma = weather_at_issue("jma_seamless")
    gfs = gfs.reindex(gfs.index.union(jma.index))
    for c in gfs.columns:
        if c.startswith("temperature_2m|"):
            gfs[c] = gfs[c].fillna(jma[c])
    parts = [h, load_features(h),
             weather_features(h, gfs, "", ("temperature_2m", "shortwave_radiation", "cloud_cover"),
                              per_city=True, daily=True),
             calendar_features(h),
             weather_features(h, jma, "jma_", ("cloud_cover",), per_city=False, daily=False)]
    f = pd.concat(parts, axis=1)
    f = f.loc[:, ~f.columns.duplicated()]        # 'hour' is in both h and the calendar
    # a linear trend, last so column order (which LightGBM's feature sampling uses) stays fixed
    f["t_years"] = (f.day - pd.Timestamp("2021-01-01")).dt.days / 365.25
    return f
