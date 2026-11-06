# Online Shoppers Purchasing Intention — описание датасета

**Источник:** https://www.kaggle.com/datasets/roshansharma/online-shoppers-intention
(оригинал: UCI Machine Learning Repository, ID 468)

- Примеров (сессий): 12330
- Признаков: 17 (+ целевая Revenue)
- Числовых: 10 (Administrative, Administrative_Duration, Informational, Informational_Duration, ProductRelated, ProductRelated_Duration, BounceRates, ExitRates, PageValues, SpecialDay)
- Категориальных: 7 (Month, OperatingSystems, Browser, Region, TrafficType, VisitorType, Weekend)

## Распределение классов
- False (нет покупки): 10422 (84.5%)
- True (покупка): 1908 (15.5%)

Дисбаланс сильный, поэтому используются взвешивание классов и метрика ROC-AUC.
