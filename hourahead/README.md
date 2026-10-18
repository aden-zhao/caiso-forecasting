# hourahead

The original class project: forecast CAISO load **one hour ahead** with classical and deep models. Built by Srikar Karra, Aydan Gerber, Mihir Shankar and Aden Zhao. It's kept here for reference; the day-ahead work in [`../dayahead`](../dayahead) grew out of it.

| Folder | What's in it |
|---|---|
| `data/` | Hourly CAISO + sub-region load with weather, 2020–2024 |
| `classical/` | Baseline, linear/ridge/lasso, random forest, XGBoost, feature importance, multi-region runs |
| `deep/` | BiLSTM, ST-CALNet (CNN + LSTM + attention), its tuning script, and a Colab tuning notebook |
| `outputs/` | Saved metrics, predictions, checkpoints and plots from those runs |
| `compare_load_models.py` | Bar charts comparing the models' reported errors |

```
cd hourahead
pip install -r requirements.txt
python classical/xgboost_model.py      # scripts can be run from any directory
```

Things to know before reusing these numbers. They're why the day-ahead project starts from scratch:
- **The models are scored on different test sets** (from 6k to 43k hours), so the comparison table isn't apples to apples.
- **One hour ahead is easy.** "Last hour's load + yesterday's change at the same hour" gets 377 MW MAE on ST-CALNet's test split, better than every model here.
- **The load data includes battery charging**, which CAISO's own forecasts leave out (see the top-level README).
- **The `multi_region_forecast*.py` scripts need `data/filtered.csv`**, which was never committed.
