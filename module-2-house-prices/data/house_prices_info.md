# House Prices — Advanced Regression Techniques

**Источник:** https://www.kaggle.com/c/house-prices-advanced-regression-techniques
**Исходные данные:** Dean De Cock, Ames Housing dataset (Journal of Statistics Education, 19(3), 2011).
**Задача:** регрессия — предсказать `SalePrice` (цена продажи дома, $).

## Файлы
- `train.csv` — 1460 строк × 81 столбец (79 признаков + `Id` + `SalePrice`)
- `test.csv` — 1459 строк × 80 столбцов (без `SalePrice`)

## Целевая переменная
Среднее $180 921, медиана $163 000, min $34 900, max $755 000; skew = 1.88; kurtosis = 6.54 → применяется `log1p`.

## Особенности
- 19 столбцов с пропусками; в большинстве из них пропуск означает «объекта нет» (гараж, подвал, бассейн и т. д.)
- Выбросы по `GrLivArea` (2 аномальных дома: Id 524 и 1299)
- Мультиколлинеарность (GarageCars ↔ GarageArea, GrLivArea ↔ TotRmsAbvGrd, TotalBsmtSF ↔ 1stFlrSF)
- 36 числовых и 43 категориальных признака (после кодирования в работе — 180 признаков)
