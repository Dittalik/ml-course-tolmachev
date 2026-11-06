# 🏠 Module 2 — House Prices: предобработка и регрессия

**Автор:** Толмачев Виталий, ПКТб-23-1
**Дата обучения:** 2026-11-05 17:41
**Датасет:** House Prices — Advanced Regression Techniques (Kaggle)

## 📊 Результаты (hold-out, 292 дома)

| Модель | RMSE (log) | MAE (log) | R² (log) | RMSE ($) | MAE ($) |
|--------|-----------|-----------|----------|---------|---------|
| Linear Regression | 0.1289 | 0.0884 | 0.9015 | $21 205 | $14 981 |
| Ridge (alpha=300) | 0.1308 | 0.0874 | 0.8985 | $20 854 | $14 896 |
| Lasso (alpha=0.003) | 0.1223 | 0.0815 | 0.9113 | $19 476 | $13 827 |
| ElasticNet (alpha=0.003  l1_ratio=0.95) — лучшая по CV | 0.1224 | 0.0815 | 0.9112 | $19 475 | $13 824 |

**Время обучения:** < 0.1 сек на каждую линейную модель. Бонус: Gradient Boosting и Stacking (R² log ≈ 0.913).
Метрики в log-пространстве считаются по `log1p(SalePrice)`; метрики в $ — после `expm1`.

## 🚀 Быстрый старт

```python
import joblib, json, requests
from io import BytesIO

BASE_URL = "https://raw.githubusercontent.com/username/ml-course-surname/main/module-2-house-prices"
model  = joblib.load(BytesIO(requests.get(f"{BASE_URL}/models/best_model.pkl").content))
scaler = joblib.load(BytesIO(requests.get(f"{BASE_URL}/models/scaler.pkl").content))
features = requests.get(f"{BASE_URL}/models/feature_cols.json").json()
# дальше: predict_price(house_dict, model, scaler, features) — функция определена в notebook.ipynb (раздел 9)
```

## 📁 Структура

- `notebook.ipynb` — полный отчёт (14 разделов)
- `data/` — `train.csv`, `test.csv`, `house_prices_info.md`
- `models/` — модели (`best_model.pkl`, `lr/ridge/lasso/enet_model.pkl`), `scaler.pkl`, `feature_cols.json`, `metrics.json`
- `examples/` — графики (`eda_plots.png`, `residuals.png`, `predictions_scatter.png` и др.)
- `requirements.txt` — зависимости (версии закреплены; нужен Python ≥ 3.11)

Запуск: `pip install -r requirements.txt`, затем выполнить `notebook.ipynb` из папки модуля (пути относительные).
