# Day-ahead CAISO load: can we beat CAISO?

**Setup.** At 09:00 PT on D-1, forecast every hour of D, the same timing as CAISO's day-ahead forecast. Only data available by 09:00 is used:
- load through 08:00
- weather forecasts from model runs published by then (GFS, JMA, via Open-Meteo's archive of past runs)

Models retrain monthly. Choices were made on 2024 (dev). 2025-01 → 2026-09 (test) was run once, at the end.

**Target gotcha.** OASIS "actual load" includes battery charging (≈ −7 GW at noon in 2026), and CAISO's forecast doesn't. Scored against OASIS, CAISO looks 2× worse than it is. We score against Today's Outlook "Current demand", which is on the same basis as CAISO's forecast.

**Results (test, 15,311 hours, MAE in MW)**

| | MAE | MAPE | vs CAISO (95% CI, day bootstrap) |
|---|---|---|---|
| CAISO day-ahead | 496 | 2.09% | |
| A: ours, no CAISO inputs | 498 | 2.11% | +2 (−23 .. +27), a tie |
| B: ours + CAISO's forecast | 426 | 1.80% | **−69 (−81 .. −57), 14% better** |

- A = mean of a whole-day MLP (`nn.py`), LightGBM, and per-hour LightGBM.
- B = mean of CAISO's forecast, A, and LightGBM correcting CAISO's forecast. This is usable by a market participant: CAISO's forecast is final ~09:30, and day-ahead bids close at 10:00.
- Dev (2024): CAISO 519, A 517, B 429.

**What mattered (dev MAE)**
- Tree baseline 644 → JMA cloud forecasts and solar features 630 → tuning and thermal lags 597 → whole-day MLP 562 → A ensemble 517.
- Weather is not the bottleneck: observed weather (an oracle) only gives 621 vs 630.
- Midday error (rooftop solar) and heat-wave weekends are where single models lose. The noon/8pm load ratio fell from 0.79 to 0.66 between 2020 and 2026.
- Didn't help: linear-leaf trees, rolling bias correction, sh_mean anchors, l1 loss.

**Run.**
```
pip install -r requirements.txt
python fetch_oasis.py ACTUAL; python fetch_oasis.py DAM; python fetch_oasis.py 2DA
python fetch_outlook.py
python fetch_weather.py gfs_seamless 2021-03-01 2026-10-16   # also jma_seamless (2020-), obs
python test_leakage.py
python final.py test
```
