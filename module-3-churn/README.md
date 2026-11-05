# 📉 Module 3 — Customer Churn: Ансамблевые методы

**Автор:** [ФИО], [группа]
**Дата:** 2026-11-__

## 📊 Результаты (test, порог 0.5)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 0.7622 | 0.5373 | 0.7513 | 0.6265 | 0.8383 |
| XGBoost | 0.7693 | 0.5514 | 0.7032 | 0.6181 | 0.8292 |
| LightGBM | 0.7601 | 0.5344 | 0.7487 | 0.6236 | 0.8379 |
| CatBoost | 0.7559 | 0.5279 | 0.7594 | 0.6228 | 0.8381 |
| LightGBM (Optuna) | 0.7459 | 0.5137 | 0.8021 | 0.6263 | 0.8504 |

Лучшая модель: **LightGBM (Optuna)**, CV ROC-AUC = 0.8495, test ROC-AUC = 0.8504.

## 🚀 Быстрый старт

```python
import joblib, requests
from io import BytesIO

BASE_URL = "https://raw.githubusercontent.com/[username]/ml-course-[фамилия]/main/module-3-churn"
model = joblib.load(BytesIO(requests.get(f"{BASE_URL}/models/best_model.pkl").content))
```

## 📁 Структура
- `notebook.ipynb` — полный отчёт
- `models/` — 5 моделей, scaler, кодировщики, метрики
- `data/` — датасет и описание
- `examples/` — графики EDA, ROC, Feature Importance, SHAP
- `requirements.txt` — зависимости
