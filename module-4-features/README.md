# 🛒 Module 4 — Online Shoppers: Инженерия признаков

**Автор:** Толмачев Виталий, ПКТб-23-1
**Дата:** 2026-11-__

## 📊 Результаты (test, порог 0.5)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 0.8706 | 0.5591 | 0.7801 | 0.6514 | 0.9237 |
| XGBoost | 0.8828 | 0.5996 | 0.7330 | 0.6596 | 0.9205 |
| LightGBM | 0.8755 | 0.5725 | 0.7749 | 0.6585 | 0.9248 |
| CatBoost | 0.8702 | 0.5581 | 0.7801 | 0.6507 | 0.9284 |
| LightGBM (Optuna) | 0.8662 | 0.5451 | 0.8220 | 0.6555 | 0.9315 |

Лучшая модель: **LightGBM (Optuna)**, CV ROC-AUC = 0.9351, test ROC-AUC = 0.9315.

Созданных признаков: 10 (TotalPages, TotalDuration, AvgPageValue, BounceExitRatio, AvgTimePerPage, ProductShare, IsHighTraffic, IsNewVisitor, IsWeekend, MonthNum).
Эффект признаков (test ROC-AUC): без созданных — 0.9269, со всеми — 0.9248.

## 🚀 Быстрый старт

```python
import joblib, requests
from io import BytesIO

BASE_URL = "https://raw.githubusercontent.com/[username]/ml-course-[фамилия]/main/module-4-features"
model = joblib.load(BytesIO(requests.get(f"{BASE_URL}/models/best_model.pkl").content))
```

## 📁 Структура
- `notebook.ipynb` — полный отчёт
- `models/` — 5 моделей, scaler, список признаков, метрики
- `data/` — датасет и описание
- `examples/` — графики EDA, ROC, Feature Importance, SHAP
- `requirements.txt` — зависимости
