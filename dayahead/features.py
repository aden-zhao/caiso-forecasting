"""Feature table: one row per target hour, using only information known at issue time.

Load features come from days D-8 .. D-2 and the morning of D-1; weather features
are forecasts as of the issue time (data.weather_at_issue).
"""
import holidays
import numpy as np
import pandas as pd

from data import (CITIES, CITY_WEIGHT, LAST_KNOWN_HOUR, TZ, WEATHER_VARS, hourly,
                  observed_weather, weather_at_issue, weighted)

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


def load_features(h: pd.DataFrame) -> pd.DataFrame:
    L = day_hour(h["load"], h)
    D, hr = h.day, h.hour
    f = pd.DataFrame(index=h.index)
    for k in range(2, 9):
        f[f"lag{k}"] = lookup(L, D - k * ONE, hr)
    # yesterday's same hour is only known for the morning hours
    f["lag1"] = np.where(hr <= LAST_KNOWN_HOUR, lookup(L, D - ONE, hr), np.nan)
    # recent same-hour level: tracks the midday dip as rooftop solar grows
    recent = f[[f"lag{k}" for k in range(2, 9)]]
    f["sh_mean"] = recent.mean(axis=1)
    f["sh_median"] = recent.median(axis=1)
    f["sh_min"] = recent.min(axis=1)
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


def weather_features(h: pd.DataFrame, wide: pd.DataFrame, prefix: str, per_city: bool,
                     daily: bool) -> pd.DataFrame:
    """Hourly weather features from a wide `<var>|<city>` table.

    per_city adds each city's temperature; daily adds temperature history and
    daily summaries (used for the main weather source only).
    """
    wide = wide.reindex(h.index)
    f = pd.DataFrame(index=h.index)
    if per_city:
        for c in wide.columns:
            if c.startswith("temperature_2m|"):
                f[f"{prefix}t_{c.split('|')[1]}"] = wide[c]
    for var in WEATHER_VARS:
        if any(c.startswith(var + "|") for c in wide.columns):
            f[f"{prefix}w_{var}"] = weighted(wide, var)
    if f"{prefix}w_cloud_cover" in f:
        # cloud-attenuated clear-sky irradiance, weighted over cities: a rooftop-solar proxy
        # (Kasten-Czeplak cloud attenuation)
        cs = clear_sky(h.index)
        f[f"{prefix}solar_proxy"] = sum(
            CITY_WEIGHT[c] * cs[c] * (1 - 0.75 * (wide[f"cloud_cover|{c}"] / 100) ** 3.4)
            for c in cs.columns) / TOTAL_WEIGHT
    if not daily:
        return f
    t = f[f"{prefix}w_temperature_2m"]
    # buildings lag the air temperature: recent hours and the trailing day
    for k in (1, 2, 3):
        f[f"{prefix}t_prev{k}"] = t.shift(k)
    f[f"{prefix}t_roll24"] = t.rolling(24, min_periods=12).mean()
    f[f"{prefix}t_roll6max"] = t.rolling(6, min_periods=3).max()
    f[f"{prefix}cdd"] = (t - 18).clip(lower=0)
    f[f"{prefix}hdd"] = (15 - t).clip(lower=0)
    T = day_hour(t, h)
    for name, days in [("d0", h.day), ("d1", h.day - ONE), ("d2", h.day - 2 * ONE)]:
        f[f"{prefix}tmax_{name}"] = T.max(axis=1).reindex(days).to_numpy()
        f[f"{prefix}tmean_{name}"] = T.mean(axis=1).reindex(days).to_numpy()
    f[f"{prefix}tmin_d0"] = T.min(axis=1).reindex(h.day).to_numpy()
    # multi-day heat build-up
    f[f"{prefix}heat3"] = f[f"{prefix}tmax_d0"] + f[f"{prefix}tmax_d1"] + f[f"{prefix}tmax_d2"]
    # temperature at the same hour two days ago, to pair with the load lag
    f[f"{prefix}t_lag2"] = lookup(T, h.day - 2 * ONE, h.hour)
    f[f"{prefix}t_delta2"] = t - f[f"{prefix}t_lag2"]
    return f


def calendar_features(h: pd.DataFrame) -> pd.DataFrame:
    """Calendar features. t_years (a linear trend) is opt-in: see backtest.feature_cols."""
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
    f["clear_sky"] = (clear_sky(h.index) * pd.Series(CITY_WEIGHT)).sum(axis=1) / TOTAL_WEIGHT
    f["t_years"] = (h.day - pd.Timestamp("2021-01-01")).dt.days / 365.25
    return f


def forecast_weather(model: str, fallback: str | None = None, policy: str = "legit") -> pd.DataFrame:
    """Forecasts from `model`, with missing temperatures filled from `fallback`."""
    wide = weather_at_issue(model, policy)
    if fallback:
        fb = weather_at_issue(fallback, policy)
        wide = wide.reindex(wide.index.union(fb.index))
        # only temperature is filled: GFS's other variables start in 2024, and JMA's
        # are available separately as their own features
        for c in fb.columns:
            if c in wide and c.startswith("temperature_2m|"):
                wide[c] = wide[c].fillna(fb[c])
    return wide


def build(main: str = "gfs_seamless", extra: tuple = ("jma_seamless",),
          policy: str = "legit") -> pd.DataFrame:
    """The full table: one row per UTC hour.

    main: weather source for the primary features. 'obs' uses observed weather,
      an oracle for ablations only. Gaps in GFS temperatures are filled from JMA.
    extra: more forecast models, added as prefixed CAISO-wide features (JMA's
      cloud cover is the only cloud forecast that goes back to 2020).
    policy: see data.weather_at_issue.
    """
    h = hourly()
    if main == "obs":
        wide = observed_weather()
    else:
        wide = forecast_weather(main, "jma_seamless" if main == "gfs_seamless" else None, policy)
    parts = [h, load_features(h), weather_features(h, wide, "", per_city=True, daily=True),
             calendar_features(h)]
    for m in extra:
        parts.append(weather_features(h, weather_at_issue(m, policy), m.split("_")[0] + "_",
                                      per_city=False, daily=False))
    f = pd.concat(parts, axis=1)
    return f.loc[:, ~f.columns.duplicated()]     # 'hour' is in both h and the calendar
