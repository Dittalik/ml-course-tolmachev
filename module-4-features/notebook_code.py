

# ------------------------------------------------------------

# Базовые библиотеки
import os                                      # работа с файлами и папками
import json                                    # чтение/запись JSON-файлов
import time                                    # замер времени обучения
import warnings                                # управление предупреждениями

import numpy as np                             # числовые массивы
import pandas as pd                            # таблицы данных (DataFrame)
import matplotlib.pyplot as plt                # графики
import seaborn as sns                          # статистические графики
import joblib                                  # сохранение и загрузка моделей
from IPython.display import Markdown, display  # красивый вывод Markdown-таблиц в ноутбуке

warnings.filterwarnings('ignore')              # убираем лишние предупреждения в выводе
sns.set_theme(style='whitegrid')               # единый стиль графиков
plt.rcParams['figure.dpi'] = 100               # разрешение графиков в ноутбуке
RANDOM_STATE = 42                              # единый seed для воспроизводимости

# Порядок месяцев; в датасете июнь записан как 'June', остальные — трёхбуквенно
MONTH_ORDER = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

def md_table(frame, digits=4):  # функция: DataFrame → Markdown-таблица
    # Превращаем DataFrame в Markdown-таблицу (для вывода в ноутбуке и для README)
    cols = list(frame.columns)                                     # заголовки столбцов
    rows = ['| ' + ' | '.join(map(str, cols)) + ' |',              # строка заголовка
            '|' + '|'.join(['---'] * len(cols)) + '|']             # разделитель
    for _, r in frame.iterrows():                                  # по строкам таблицы
        rows.append('| ' + ' | '.join(                             # числа форматируем, остальное — как есть
            f'{v:.{digits}f}' if isinstance(v, (float, np.floating)) else str(v) for v in r.values) + ' |')  # числа форматируем, остальное — как есть
    return '\n'.join(rows)                                         # собираем таблицу в одну строку

# Папки модуля (ноутбук лежит в module-4-features/, поэтому пути относительные)
for folder in ['data', 'models', 'examples']:                      # создаём рабочие папки модуля
    os.makedirs(folder, exist_ok=True)                             # exist_ok — без ошибки, если папка уже есть

# Загрузка датасета
csv_path = 'data/online_shoppers_intention.csv'    # относительный путь внутри module-4-features/
if not os.path.exists(csv_path):                   # если файла нет — останавливаемся с понятной ошибкой
    raise FileNotFoundError(  # останавливаем выполнение с понятной ошибкой
        'Положите файл в data/online_shoppers_intention.csv '  # инструкция пользователю
        '(https://www.kaggle.com/datasets/roshansharma/online-shoppers-intention)')  # ссылка на датасет

df = pd.read_csv(csv_path)                         # читаем CSV в DataFrame

print(df.shape)                                    # (строки, столбцы)
print(df.head())                                   # первые 5 строк

# Статистика датасета: типы признаков и распределение классов
target_counts = df['Revenue'].value_counts()                      # абсолютные частоты классов (False/True)
target_share = df['Revenue'].value_counts(normalize=True) * 100   # доли в процентах

# 10 числовых признаков (счётчики, длительности, метрики Google Analytics)
num_features = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',  # числовые признаки
                'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay']
# Остальные признаки (кроме целевой) считаем категориальными: Month, OperatingSystems, Browser, Region,
# TrafficType, VisitorType, Weekend
cat_features = [c for c in df.columns if c not in num_features + ['Revenue']]  # остальные признаки считаем категориальными

print(f'Примеров: {len(df)}')                                                  # число строк
print(f'Признаков (без Revenue): {len(num_features) + len(cat_features)}')      # число входных признаков
print(f'Числовых: {len(num_features)}, категориальных: {len(cat_features)}')    # разбивка по типам
print('Категориальные:', cat_features)                                          # какие именно
print()  # пустая строка
print('Распределение классов:')                                                 # заголовок блока
for cls in target_counts.index:                                                 # по каждому классу
    print(f'  {cls}: {target_counts[cls]} ({target_share[cls]:.1f}%)')          # число и доля класса

# Описание датасета сохраняем в data/dataset_info.md (требование задания)
info_md = f"""# Online Shoppers Purchasing Intention — описание датасета

**Источник:** https://www.kaggle.com/datasets/roshansharma/online-shoppers-intention
(оригинал: UCI Machine Learning Repository, ID 468)

- Примеров (сессий): {len(df)}
- Признаков: {len(num_features) + len(cat_features)} (+ целевая Revenue)
- Числовых: {len(num_features)} ({', '.join(num_features)})
- Категориальных: {len(cat_features)} ({', '.join(cat_features)})

## Распределение классов
- False (нет покупки): {target_counts.get(False, 0)} ({target_share.get(False, 0):.1f}%)
- True (покупка): {target_counts.get(True, 0)} ({target_share.get(True, 0):.1f}%)

Дисбаланс сильный, поэтому используются взвешивание классов и метрика ROC-AUC.
"""
with open('data/dataset_info.md', 'w', encoding='utf-8') as f:    # открываем файл для записи
    f.write(info_md)                                              # записываем описание датасета
print('\n✅ data/dataset_info.md сохранён')                         # подтверждение сохранения

# Первичный осмотр данных
df.info()                                                  # типы столбцов и количество непустых значений
print()  # пустая строка
print(df.describe().T)                                     # статистика по числовым столбцам (транспонирована для читаемости)
print()  # пустая строка
print('Полных дубликатов строк:', df.duplicated().sum())    # одинаковые сессии (в этом датасете они встречаются)

# Анализ пропусков: явные пропуски (NaN)
missing = df.isna().sum()                                  # число пропусков по столбцам
print('Всего явных пропусков:', int(missing.sum()))        # по описанию датасета их нет

plt.figure(figsize=(10, 4))                                # размер графика
sns.barplot(x=missing.index, y=missing.values, color='steelblue')   # столбец на каждый признак
plt.xticks(rotation=90)                                    # поворачиваем подписи оси X
plt.title('Количество пропусков по столбцам (все столбцы заполнены)')   # заголовок
plt.ylabel('Число пропусков')                              # подпись оси Y
plt.tight_layout()                                         # подгоняем отступы
plt.show()                                                 # показываем график

# Скрытые пропуски: отрицательные значения там, где их быть не может (время и счётчики >= 0)
neg = (df[num_features] < 0).sum()                         # число отрицательных значений по числовым столбцам
print('\nОтрицательные значения по столбцам (только ненулевые):')   # заголовок вывода
print(neg[neg > 0] if (neg > 0).any() else 'не найдены')   # показываем найденные или сообщаем, что их нет

# Обработка скрытых пропусков: длительность -1 означает «нет данных», а не отрицательное время
pairs = [('Administrative_Duration', 'Administrative'),    # (столбец длительности, столбец счётчика страниц)
         ('Informational_Duration', 'Informational'),
         ('ProductRelated_Duration', 'ProductRelated')]
n_fixed = 0                                                # сколько значений исправлено
for dur, cnt in pairs:                                     # по каждой паре
    bad = df[dur] < 0                                      # маска некорректных длительностей
    ok_median = df.loc[~bad & (df[cnt] > 0), dur].median() # медиана по корректным сессиям, где страницы были
    df.loc[bad & (df[cnt] == 0), dur] = 0                  # страниц нет → времени на них 0
    df.loc[bad & (df[cnt] > 0), dur] = ok_median           # страницы были → заполняем медианой (устойчива к выбросам)
    n_fixed += int(bad.sum())                              # считаем исправленные значения
print('Исправлено некорректных длительностей:', n_fixed)    # 0 — значит, в вашем файле их нет
print('Осталось отрицательных значений:', int((df[num_features] < 0).sum().sum()))   # проверка результата

# Анализ целевой переменной
print(df['Revenue'].value_counts())                        # абсолютные значения по классам
print()  # пустая строка
print((df['Revenue'].value_counts(normalize=True) * 100).round(2))   # доли в %

# EDA: сетка 2x3 — 6 графиков
fig, axes = plt.subplots(2, 3, figsize=(20, 11))           # сетка 2x3 для 6 графиков

# 1. Countplot — распределение целевой переменной
sns.countplot(data=df, x='Revenue', hue='Revenue', palette='Set2', legend=False, ax=axes[0, 0])   # график 1
axes[0, 0].set_title('1. Распределение целевой переменной')       # заголовок
axes[0, 0].set_xlabel('Revenue (False — нет покупки, True — покупка)')   # подпись оси X
axes[0, 0].set_ylabel('Число сессий')                             # подпись оси Y

# 2. Гистограмма ProductRelated_Duration с разбивкой по Revenue (логарифм: распределение очень скошено)
plot_df = df.assign(LogProductDuration=np.log1p(df['ProductRelated_Duration']))   # log(1+x) сглаживает хвост
sns.histplot(data=plot_df, x='LogProductDuration', hue='Revenue', kde=True,  # график 2: гистограмма tenure с KDE по классам
             stat='density', common_norm=False, ax=axes[0, 1])    # график 2: формы распределений сравнимы
axes[0, 1].set_title('2. Время на товарных страницах по Revenue')  # заголовок
axes[0, 1].set_xlabel('log(1 + ProductRelated_Duration)')         # подпись оси X

# 3. Boxplot PageValues по Revenue
sns.boxplot(data=df, x='Revenue', y='PageValues', hue='Revenue', palette='Set2', legend=False, ax=axes[0, 2])   # график 3
axes[0, 2].set_title('3. PageValues по Revenue')                  # заголовок
axes[0, 2].set_ylabel('PageValues')                               # подпись оси Y

# 4. Heatmap корреляций числовых признаков и целевой
corr_df = df[num_features].copy()                                 # числовые столбцы
corr_df['Revenue'] = df['Revenue'].astype(int)                    # целевая в виде 0/1
sns.heatmap(corr_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0,  # график 4: тепловая карта корреляций
            annot_kws={'size': 6}, ax=axes[1, 0])                 # график 4: тепловая карта корреляций
axes[1, 0].set_title('4. Корреляционная матрица')                 # заголовок

# 5. Доля покупок по типу посетителя
visitor_rate = df.groupby('VisitorType')['Revenue'].mean() * 100  # доля сессий с покупкой (%) по типам посетителей
sns.barplot(x=visitor_rate.index, y=visitor_rate.values, color='tomato', ax=axes[1, 1])   # график 5
axes[1, 1].set_title('5. Доля покупок по типу посетителя')        # заголовок
axes[1, 1].set_ylabel('Доля покупок, %')                          # подпись оси Y
axes[1, 1].set_xlabel('VisitorType')                              # подпись оси X

# 6. Доля покупок по месяцам (в календарном порядке)
month_rate = (df.groupby('Month')['Revenue'].mean() * 100).reindex(
    [m for m in MONTH_ORDER if m in df['Month'].unique()])        # доля покупок (%) по месяцам в порядке календаря
sns.barplot(x=month_rate.index, y=month_rate.values, color='darkorange', ax=axes[1, 2])   # график 6
axes[1, 2].set_title('6. Доля покупок по месяцам')                # заголовок
axes[1, 2].set_ylabel('Доля покупок, %')                          # подпись оси Y
axes[1, 2].set_xlabel('Month')                                    # подпись оси X

plt.tight_layout()                                                # подгоняем отступы
plt.savefig('examples/eda_plots.png', dpi=120, bbox_inches='tight')   # сохраняем для репозитория
plt.show()                                                        # показываем график

# Feature Engineering: новые признаки на основе поведения пользователя
df['TotalPages'] = df['Administrative'] + df['Informational'] + df['ProductRelated']   # всего страниц за сессию
df['TotalDuration'] = (df['Administrative_Duration'] + df['Informational_Duration']
                       + df['ProductRelated_Duration'])                               # всего времени в сессии, сек
df['AvgPageValue'] = df['PageValues'] / (df['TotalPages'] + 1)                         # ценность на страницу (+1 — защита от /0)
df['BounceExitRatio'] = df['BounceRates'] / (df['ExitRates'] + 0.001)                  # отношение отказов к выходам
df['AvgTimePerPage'] = df['TotalDuration'] / (df['TotalPages'] + 1)                    # среднее время на страницу
df['ProductShare'] = df['ProductRelated'] / (df['TotalPages'] + 1)                     # доля товарных страниц

df['IsNewVisitor'] = (df['VisitorType'] == 'New_Visitor').astype(int)                  # флаг нового посетителя
df['IsWeekend'] = df['Weekend'].astype(int)                                            # выходной: bool → 0/1
df['IsHighTraffic'] = (df['TrafficType'] > 10).astype(int)                             # бинарный признак по порогу TrafficType

# Месяц → число 1-12 (to_datetime с форматом %b не подходит: в датасете июнь записан как 'June')
MONTH_MAP = {m: i + 1 for i, m in enumerate(MONTH_ORDER)}                              # словарь «месяц → номер»
df['MonthNum'] = df['Month'].map(MONTH_MAP)                                            # применяем словарь
unknown_months = df.loc[df['MonthNum'].isna(), 'Month'].unique()                       # месяцы, которых нет в словаре
assert len(unknown_months) == 0, f'Неизвестные названия месяцев: {unknown_months}'     # останавливаемся, если такие есть

# Списки новых признаков (понадобятся для сравнения «с признаками / без»)
engineered_numeric = ['TotalPages', 'TotalDuration', 'AvgPageValue', 'BounceExitRatio',
                      'AvgTimePerPage', 'ProductShare', 'IsHighTraffic']                # чисто сконструированные признаки
new_features = engineered_numeric + ['IsNewVisitor', 'IsWeekend', 'MonthNum']           # все 10 новых признаков
print('Создано новых признаков:', len(new_features))                                   # должно быть 10
print(df[new_features].head())                                                         # проверяем результат

# Копия с читаемыми значениями — понадобится для бизнес-анализа в разделе 8
df_eda = df.copy()                                                                     # копия до кодирования
df_eda['Revenue'] = df_eda['Revenue'].astype(int)                                      # целевая как 0/1 для расчёта долей

# Кодирование
df['Revenue'] = df['Revenue'].astype(int)                  # целевая: True → 1, False → 0

# One-Hot для VisitorType: 3 категории = 3 столбца; New_Visitor уже есть как IsNewVisitor,
# Returning_Visitor — базовая категория (все нули), поэтому добавляем только VisitorType_Other
visitor_ohe = pd.get_dummies(df['VisitorType'], prefix='VisitorType', dtype=int)       # One-Hot по типу посетителя
df = pd.concat([df, visitor_ohe[['VisitorType_Other']]], axis=1)                       # добавляем нужный столбец

# Исходные строковые/булевы столбцы заменены числовыми версиями — удаляем
df = df.drop(columns=['Month', 'VisitorType', 'Weekend'])                              # остаются только числа

# OperatingSystems, Browser, Region, TrafficType — идентификаторы без порядка; для деревьев оставляем
# целочисленные коды (Target Encoding проверим в бонусе)
non_numeric = df.select_dtypes(exclude='number').columns.tolist()                      # столбцы, которые не числа
assert not non_numeric, f'Остались нечисловые столбцы: {non_numeric}'                  # проверка: всё закодировано
print(df.head())                                                                       # результат кодирования

# Разделение на train/test со стратификацией (сохраняем долю классов)
from sklearn.model_selection import train_test_split         # разбиение на train/test
from sklearn.preprocessing import StandardScaler             # масштабирование числовых признаков

feature_cols = [c for c in df.columns if c != 'Revenue']     # список признаков модели
X = df[feature_cols]                                         # матрица признаков
y = df['Revenue']                                            # целевая переменная

X_train, X_test, y_train, y_test = train_test_split(  # 80% train / 20% test
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)   # 80% train / 20% test, stratify=y сохраняет доли классов

print('Признаков в модели:', len(feature_cols))              # итоговое число признаков
print('train:', X_train.shape, ' test:', X_test.shape)       # размеры выборок
print('Доля покупок — train: %.3f, test: %.3f' % (y_train.mean(), y_test.mean()))   # доля класса 1 одинакова

# Scaler для непрерывных признаков. Деревьям масштабирование не нужно, поэтому модели обучаем на
# немасштабированных данных; scaler обучаем только на train и сохраняем для совместимости
scale_cols = num_features + ['TotalPages', 'TotalDuration', 'AvgPageValue',  # числовые признаки для масштабирования
                             'BounceExitRatio', 'AvgTimePerPage', 'ProductShare']      # непрерывные признаки
scaler = StandardScaler().fit(X_train[scale_cols])           # обучаем scaler только на train

# Импорт моделей
from sklearn.ensemble import RandomForestClassifier   # Random Forest (bagging)
from xgboost import XGBClassifier                     # XGBoost (boosting)
from lightgbm import LGBMClassifier                   # LightGBM (boosting)
from catboost import CatBoostClassifier               # CatBoost (boosting)

train_times = {}   # сюда складываем время обучения каждой модели

def fit_timed(model, name):  # обучение с замером времени
    # Обучаем модель и замеряем время
    start = time.time()                                       # старт таймера
    model.fit(X_train, y_train)                               # обучаем модель
    train_times[name] = time.time() - start                   # сохраняем время обучения
    print(f'⏱️ Время обучения {name}: {train_times[name]:.3f} сек')   # выводим время
    return model                                              # возвращаем обученную модель

# 1. Random Forest (bagging)
rf_model = fit_timed(RandomForestClassifier(  # Random Forest: 200 деревьев
    n_estimators=200, max_depth=10, class_weight='balanced',  # число деревьев, глубина, баланс классов
    random_state=RANDOM_STATE, n_jobs=-1), 'Random Forest')   # воспроизводимость, все ядра

# 2. XGBoost (boosting)
xgb_model = fit_timed(XGBClassifier(  # XGBoost
    n_estimators=200, max_depth=6, learning_rate=0.1,         # число деревьев, глубина, скорость обучения
    subsample=0.9, colsample_bytree=0.9,                      # доли объектов и признаков на дерево
    scale_pos_weight=5,                                       # компенсация дисбаланса (84.5/15.5 ≈ 5.4)
    random_state=RANDOM_STATE, eval_metric='logloss'), 'XGBoost')   # воспроизводимость, метрика логирования

# 3. LightGBM (boosting); параметры вынесены в словарь — переиспользуем в сравнении «с признаками / без» и в Optuna
lgbm_params = dict(n_estimators=200, num_leaves=31, max_depth=7, learning_rate=0.05,   # листьев на дерево — главный параметр
                   class_weight='balanced', random_state=RANDOM_STATE, verbose=-1)     # баланс классов, без логов
lgbm_model = fit_timed(LGBMClassifier(**lgbm_params), 'LightGBM')                      # обучаем LightGBM

# 4. CatBoost (boosting). Признаки уже закодированы числами, поэтому cat_features не задаём
cat_model = fit_timed(CatBoostClassifier(  # CatBoost
    iterations=200, depth=6, learning_rate=0.1,               # итерации, глубина, скорость обучения
    auto_class_weights='Balanced', random_state=RANDOM_STATE,  # баланс классов, без логов и без папки catboost_info
    verbose=0, allow_writing_files=False), 'CatBoost')        # без логов и без папки catboost_info

# Таблица моделей с фактическим временем обучения (вывод как Markdown-таблица)
models_info = pd.DataFrame([  # строки таблицы моделей
    ['Random Forest', 'Bagging',  'n_estimators=200, max_depth=10, class_weight=balanced', train_times['Random Forest']],  # Random Forest
    ['XGBoost',       'Boosting', 'n_estimators=200, max_depth=6, lr=0.1, scale_pos_weight=5', train_times['XGBoost']],  # XGBoost
    ['LightGBM',      'Boosting', 'n_estimators=200, num_leaves=31, max_depth=7, lr=0.05, class_weight=balanced', train_times['LightGBM']],  # LightGBM
    ['CatBoost',      'Boosting', 'iterations=200, depth=6, lr=0.1, auto_class_weights=Balanced', train_times['CatBoost']],  # CatBoost
], columns=['Модель', 'Тип', 'Параметры', 'Время обучения, сек'])   # названия столбцов
display(Markdown(md_table(models_info, 3)))                          # таблица с реальными значениями времени

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,  # метрики классификации
                             roc_auc_score, roc_curve, confusion_matrix, classification_report)   # метрики классификации

def evaluate_model(model, X_test, y_test, name):  # считаем все метрики одной модели
    # Считаем набор метрик для одной модели
    y_pred = model.predict(X_test)                    # предсказанные классы (порог 0.5)
    y_proba = model.predict_proba(X_test)[:, 1]       # вероятность покупки
    return {  # словарь с метриками
        'Model': name,                                # название модели
        'Accuracy':  accuracy_score(y_test, y_pred),  # доля верных ответов
        'Precision': precision_score(y_test, y_pred), # точность по классу «покупка»
        'Recall':    recall_score(y_test, y_pred),    # полнота по классу «покупка»
        'F1':        f1_score(y_test, y_pred),        # гармоническое среднее Precision и Recall
        'ROC-AUC':   roc_auc_score(y_test, y_proba),  # качество ранжирования по вероятностям
    }

models = {'Random Forest': rf_model, 'XGBoost': xgb_model,  # все модели в одном словаре
          'LightGBM': lgbm_model, 'CatBoost': cat_model}               # все модели в одном словаре

results = [evaluate_model(m, X_test, y_test, name) for name, m in models.items()]   # метрики для каждой модели
results_df = pd.DataFrame(results)                                     # таблица результатов
results_df['Время, сек'] = results_df['Model'].map(train_times)        # добавляем время обучения
print(results_df.round(4).to_string(index=False))                      # таблица метрик текстом
display(Markdown(md_table(results_df)))                                # то же в виде Markdown-таблицы

# Лучшая модель по ROC-AUC и её матрица ошибок
best_name = results_df.loc[results_df['ROC-AUC'].idxmax(), 'Model']    # лучшая модель по ROC-AUC
best_baseline = models[best_name]                                      # объект лучшей модели
y_pred_best = best_baseline.predict(X_test)                            # её предсказания
print('Лучшая модель по ROC-AUC:', best_name)                          # название лучшей модели

cm = confusion_matrix(y_test, y_pred_best)                             # матрица ошибок
plt.figure(figsize=(6, 5))                                             # создаём фигуру
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',  # рисуем матрицу ошибок
            xticklabels=['Нет покупки', 'Покупка'],
            yticklabels=['Нет покупки', 'Покупка'])                    # рисуем матрицу ошибок, подписи классов
plt.xlabel('Предсказано')                                              # подпись оси X
plt.ylabel('Истинно')                                                  # подпись оси Y
plt.title(f'Матрица ошибок: {best_name}')                              # заголовок графика
plt.tight_layout()                                                     # подгоняем отступы
plt.show()                                                             # показываем график

print(classification_report(y_test, y_pred_best, target_names=['Нет покупки', 'Покупка']))   # подробный отчёт по классам

# ROC-кривые всех 4 моделей на одном графике
plt.figure(figsize=(10, 7))                                            # создаём фигуру
for name, model in models.items():                                     # кривая для каждой модели
    y_proba = model.predict_proba(X_test)[:, 1]                        # вероятность покупки
    fpr, tpr, _ = roc_curve(y_test, y_proba)                           # точки ROC-кривой
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc_score(y_test, y_proba):.3f})')   # кривая с AUC в легенде

plt.plot([0, 1], [0, 1], 'k--', label='Случай (AUC = 0.5)')            # диагональ — случайное угадывание
plt.xlabel('False Positive Rate')                                      # подпись оси X
plt.ylabel('True Positive Rate')                                       # подпись оси Y
plt.title('ROC-кривые моделей')                                        # заголовок графика
plt.legend()                                                           # легенда
plt.grid(alpha=0.3)                                                    # сетка
plt.savefig('examples/roc_curves.png', dpi=120, bbox_inches='tight')   # сохраняем картинку в examples/
plt.show()                                                             # показываем график

# Автоматический анализ результатов (числа подставляются из таблицы)
fastest = results_df.loc[results_df['Время, сек'].idxmin(), 'Model']   # самая быстрая модель
best_recall = results_df.loc[results_df['Recall'].idxmax(), 'Model']   # модель с лучшим Recall
spread = results_df['ROC-AUC'].max() - results_df['ROC-AUC'].min()     # разброс ROC-AUC между моделями

print(f"• Лучшая модель по ROC-AUC: {best_name} ({results_df['ROC-AUC'].max():.4f})")      # лучшая по ROC-AUC
print(f"• Разброс ROC-AUC между моделями: {spread:.4f}")                                    # разброс качества
print(f"• Быстрее всех обучается: {fastest} ({results_df['Время, сек'].min():.3f} сек)")    # самая быстрая
print(f"• Наибольший Recall: {best_recall} ({results_df['Recall'].max():.4f})")             # лучший Recall
for _, r in results_df.iterrows():                                     # характер ошибок каждой модели
    tendency = 'Precision > Recall (модель осторожна в предсказании «покупка»)' if r['Precision'] > r['Recall'] \
        else 'Recall > Precision (модель чаще предсказывает покупку, пропускает меньше покупателей ценой ложных срабатываний)'   # Precision выше или ниже Recall
    print(f"• {r['Model']}: {tendency}")                               # вывод по модели

# Feature Importance для Random Forest и LightGBM
imp_rf = pd.DataFrame({'Feature': feature_cols, 'Importance': rf_model.feature_importances_}) \
    .sort_values('Importance', ascending=False)             # важность признаков Random Forest, по убыванию
imp_lgbm = pd.DataFrame({'Feature': feature_cols, 'Importance': lgbm_model.feature_importances_}) \
    .sort_values('Importance', ascending=False)             # важность признаков LightGBM, по убыванию
importances = imp_lgbm                                      # используем как ориентир в разделе SHAP

fig, axes = plt.subplots(1, 2, figsize=(16, 8))             # два графика рядом
for ax, data, title in [(axes[0], imp_rf, 'Random Forest (снижение неопределённости)'),  # RF слева, LightGBM справа
                        (axes[1], imp_lgbm, 'LightGBM (число разбиений)')]:   # RF слева, LightGBM справа
    sns.barplot(data=data.head(15), x='Importance', y='Feature', hue='Feature',  # ТОП-15 признаков
                palette='viridis', legend=False, ax=ax)     # ТОП-15 признаков, цветовая палитра
    ax.set_title(f'ТОП-15 важных признаков: {title}')       # заголовок подграфика
plt.tight_layout()                                          # подгоняем отступы
plt.savefig('examples/feature_importance.png', dpi=120, bbox_inches='tight')   # сохраняем картинку
plt.show()                                                  # показываем график

# Топ-5 признаков по обеим моделям (Markdown-таблица); звёздочка — созданный нами признак
mark = lambda f: f + (' ★' if f in new_features else '')    # помечаем новые признаки
top5 = pd.DataFrame({'#': range(1, 6),
                     'Random Forest': [mark(f) for f in imp_rf['Feature'].head(5)],
                     'LightGBM': [mark(f) for f in imp_lgbm['Feature'].head(5)]})   # топ-5 по каждой модели
display(Markdown('**Топ-5 признаков, влияющих на покупку** (★ — созданный нами признак):\n\n' + md_table(top5)))   # таблица топ-5

# Бизнес-гипотезы по исходным (некодированным) данным; формулировки подставляются автоматически
rate = lambda col: (df_eda.groupby(col)['Revenue'].mean() * 100)      # доля покупок (%) по группам признака

pv = rate(df_eda['PageValues'] > 0)                                   # PageValues > 0 против = 0
vt = rate('VisitorType')                                              # по типу посетителя
mr = rate('Month')                                                    # по месяцам
wk = rate('IsWeekend')                                                # будни / выходные
sd = rate(df_eda['SpecialDay'] > 0)                                   # рядом с особым днём или нет
corr_bounce = df_eda['BounceRates'].corr(df_eda['Revenue'])           # корреляция отказов с покупкой
med_dur = df_eda.groupby('Revenue')['ProductRelated_Duration'].median()   # медианное время на товарных страницах

lines = []                                                            # собираем строки вывода
lines.append(f"- Сессии с PageValues > 0 заканчиваются покупкой в {pv.get(True, float('nan')):.1f}% случаев, "
             f"с PageValues = 0 — в {pv.get(False, float('nan')):.1f}%.")           # эффект PageValues
lines.append(f"- Время на товарных страницах (медиана): покупатели — {med_dur.get(1, float('nan')):.0f} сек, "
             f"не купившие — {med_dur.get(0, float('nan')):.0f} сек.")              # эффект времени
lines.append(f"- Корреляция BounceRates с покупкой: {corr_bounce:+.2f} "
             f"({'отказы снижают вероятность покупки' if corr_bounce < 0 else 'отказы не снижают вероятность покупки'}).")   # эффект отказов
if 'New_Visitor' in vt.index and 'Returning_Visitor' in vt.index:     # сравниваем новых и вернувшихся
    word = 'чаще' if vt['New_Visitor'] > vt['Returning_Visitor'] else 'реже'   # чаще или реже покупают новые
    lines.append(f"- Новые посетители покупают {word} вернувшихся: {vt['New_Visitor']:.1f}% против "
                 f"{vt['Returning_Visitor']:.1f}%.")                  # эффект типа посетителя
lines.append(f"- Лучший месяц по доле покупок — {mr.idxmax()} ({mr.max():.1f}%), худший — {mr.idxmin()} ({mr.min():.1f}%).")   # сезонность
lines.append(f"- Выходные: {wk.get(1, float('nan')):.1f}% покупок, будни: {wk.get(0, float('nan')):.1f}%.")   # эффект выходных
lines.append(f"- Рядом с особым днём (SpecialDay > 0): {sd.get(True, float('nan')):.1f}% покупок, "
             f"в остальные дни: {sd.get(False, float('nan')):.1f}%.")                # эффект особого дня
display(Markdown('**Бизнес-гипотезы, проверенные по данным:**\n\n' + '\n'.join(lines)))   # выводим список

# Влияние инженерии признаков: LightGBM на исходных признаках и на полном наборе
from sklearn.model_selection import StratifiedKFold, cross_val_score    # стратифицированная кросс-валидация

base_cols = [c for c in feature_cols if c not in engineered_numeric]   # набор без конструируемых признаков (только кодировка исходных)
skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)      # одинаковые 5 фолдов для обоих наборов

rows = []                                                              # сюда складываем строки сравнения
for label, cols in [('Исходные + кодирование', base_cols), ('С созданными признаками', feature_cols)]:   # два набора
    cv_auc = cross_val_score(LGBMClassifier(**lgbm_params), X_train[cols], y_train,
                             cv=skf, scoring='roc_auc').mean()         # средний ROC-AUC по фолдам на train
    m = LGBMClassifier(**lgbm_params).fit(X_train[cols], y_train)      # обучаем на всём train
    test_auc = roc_auc_score(y_test, m.predict_proba(X_test[cols])[:, 1])   # ROC-AUC на test
    rows.append([label, len(cols), cv_auc, test_auc])                  # запоминаем результат

ablation_df = pd.DataFrame(rows, columns=['Набор признаков', 'Число признаков', 'CV ROC-AUC', 'Test ROC-AUC'])   # итоговая таблица
display(Markdown(md_table(ablation_df)))                               # выводим таблицу
delta_cv = ablation_df.loc[1, 'CV ROC-AUC'] - ablation_df.loc[0, 'CV ROC-AUC']       # разница по CV
delta_test = ablation_df.loc[1, 'Test ROC-AUC'] - ablation_df.loc[0, 'Test ROC-AUC']  # разница по test
print(f'Изменение ROC-AUC от созданных признаков: CV {delta_cv:+.4f}, test {delta_test:+.4f}')   # итог сравнения

# Установка (если нужно): !pip install optuna shap
import optuna                                                           # байесовская оптимизация гиперпараметров
from sklearn.model_selection import StratifiedKFold, cross_val_score    # стратифицированная кросс-валидация

optuna.logging.set_verbosity(optuna.logging.WARNING)   # меньше шума в выводе

def objective(trial):  # целевая функция для Optuna
    # Пространство поиска гиперпараметров
    params = {  # пространство поиска гиперпараметров
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),                 # число деревьев
        'num_leaves': trial.suggest_int('num_leaves', 10, 100, step=5),                       # число листьев
        'max_depth': trial.suggest_int('max_depth', 3, 12),                                   # максимальная глубина
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),           # скорость обучения (лог-шкала)
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),                              # доля объектов на дерево
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),                # доля признаков на дерево
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),                   # минимум объектов в листе
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),                    # L1-регуляризация
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),                  # L2-регуляризация
    }
    model = LGBMClassifier(**params, subsample_freq=1, class_weight='balanced',  # модель с предложенными параметрами
                           random_state=RANDOM_STATE, verbose=-1)   # модель с предложенными параметрами
    cv_scores = cross_val_score(  # 5-fold CV по train
        model, X_train, y_train,                                    # данные для кросс-валидации
        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),   # 5 стратифицированных фолдов
        scoring='roc_auc')                                          # метрика — ROC-AUC
    return cv_scores.mean()   # максимизируем средний ROC-AUC по фолдам

# CV-оценка baseline LightGBM для честного сравнения (те же фолды, что и в Optuna)
baseline_cv = cross_val_score(  # CV baseline LightGBM
    LGBMClassifier(**lgbm_params), X_train, y_train,                # baseline-параметры
    cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE), # те же фолды
    scoring='roc_auc').mean()                                       # средний ROC-AUC по фолдам
print(f'Baseline LightGBM, CV ROC-AUC: {baseline_cv:.4f}')          # качество baseline

# Запуск оптимизации (TPE с фиксированным seed)
N_TRIALS = 30                                                       # число испытаний
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))   # максимизируем AUC
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)   # запускаем поиск

print(f'\nBest CV ROC-AUC: {study.best_value:.4f}')                 # лучший результат CV
for key, value in study.best_params.items():                        # лучшие параметры
    print(f'  {key}: {value}')                                      # параметр: значение

# История оптимизации
trials_df = study.trials_dataframe()                                # результаты испытаний в таблице
plt.figure(figsize=(9, 4))                                          # создаём фигуру
plt.plot(trials_df['number'], trials_df['value'], 'o-', alpha=0.6, label='ROC-AUC испытания')   # качество каждого испытания
plt.plot(trials_df['number'], trials_df['value'].cummax(), 'r-', lw=2, label='Лучшее на данный момент')   # лучшее на данный момент
plt.xlabel('Номер испытания')                                       # подпись оси X
plt.ylabel('CV ROC-AUC')                                            # подпись оси Y
plt.title('История оптимизации Optuna')                             # заголовок графика
plt.legend()                                                        # легенда
plt.grid(alpha=0.3)                                                 # сетка
plt.show()                                                          # показываем график

# Обучаем лучшую модель на всём train и сравниваем с baseline на test
best_lgbm = LGBMClassifier(**study.best_params, subsample_freq=1, class_weight='balanced',  # модель с лучшими параметрами
                           random_state=RANDOM_STATE, verbose=-1)   # модель с лучшими параметрами
best_lgbm.fit(X_train, y_train)                                     # обучаем на всём train

y_pred_tuned = best_lgbm.predict(X_test)                            # классы на test
y_proba_best = best_lgbm.predict_proba(X_test)[:, 1]                # вероятности на test

auc_base = roc_auc_score(y_test, lgbm_model.predict_proba(X_test)[:, 1])   # ROC-AUC baseline на test
auc_tuned = roc_auc_score(y_test, y_proba_best)                     # ROC-AUC после Optuna на test
print(f'CV ROC-AUC  baseline: {baseline_cv:.4f} → Optuna: {study.best_value:.4f}')   # сравнение по CV
print(f'Test ROC-AUC baseline: {auc_base:.4f} → Optuna: {auc_tuned:.4f}  (прирост {auc_tuned - auc_base:+.4f})')   # сравнение на test

# Итоговая таблица: все модели + тюнингованный LightGBM
tuned_row = evaluate_model(best_lgbm, X_test, y_test, 'LightGBM (Optuna)')   # метрики тюнингованной модели
final_df = pd.concat([results_df.drop(columns='Время, сек'), pd.DataFrame([tuned_row])], ignore_index=True)   # общая таблица всех моделей
print()  # пустая строка
print(final_df.round(4).to_string(index=False))                     # итоговая таблица

# Итог Optuna в формате методички (числа подставляются автоматически)
params_md = '\n'.join(f'- {k}: {v:.4g}' if isinstance(v, float) else f'- {k}: {v}'  # лучшие параметры списком Markdown
                      for k, v in study.best_params.items())        # лучшие параметры списком Markdown
display(Markdown(  # вывод итога в формате методички
    f'**Optuna результаты:**\n\n- Проведено {N_TRIALS} испытаний\n'                                      # число испытаний
    f'- Baseline ROC-AUC (test): {auc_base:.4f}\n- Optuna ROC-AUC (test): {auc_tuned:.4f}\n'               # метрики до и после
    f'- Прирост: {(auc_tuned - auc_base) * 100:+.2f} п.п.\n\n**Лучшие параметры:**\n\n{params_md}'))      # прирост в п.п.

from sklearn.preprocessing import TargetEncoder        # Target Encoding с встроенной кросс-валидацией
from sklearn.compose import ColumnTransformer          # применяем кодировщик только к выбранным столбцам
from sklearn.pipeline import make_pipeline             # кодировщик + модель в одном объекте
from sklearn.calibration import CalibratedClassifierCV # калибровка вероятностей
from sklearn.metrics import brier_score_loss           # метрика качества вероятностей

# 1. Target Encoding идентификаторов (обучается только на train внутри pipeline)
te_cols = ['OperatingSystems', 'Browser', 'Region', 'TrafficType']        # номинальные идентификаторы
te_pipe = make_pipeline(
    ColumnTransformer([('te', TargetEncoder(target_type='binary', random_state=RANDOM_STATE), te_cols)],
                      remainder='passthrough'),                           # остальные столбцы без изменений
    LGBMClassifier(**study.best_params, subsample_freq=1, class_weight='balanced',  # лучшая модель Optuna
                   random_state=RANDOM_STATE, verbose=-1))                # LightGBM с лучшими параметрами Optuna
te_pipe.fit(X_train, y_train)                                             # обучаем кодировщик и модель
te_row = evaluate_model(te_pipe, X_test, y_test, 'LightGBM (Optuna) + TargetEncoding')   # метрики модели с Target Encoding

# 2. Калибровка вероятностей лучшей модели (isotonic, 5-fold)
calibrated = CalibratedClassifierCV(  # калибровка вероятностей
    LGBMClassifier(**study.best_params, subsample_freq=1, class_weight='balanced',  # лучшая модель Optuna
                   random_state=RANDOM_STATE, verbose=-1), method='isotonic', cv=5)   # isotonic-калибровка, 5 фолдов
calibrated.fit(X_train, y_train)                                          # обучаем калибратор
cal_row = evaluate_model(calibrated, X_test, y_test, 'LightGBM (Optuna) + калибровка')   # метрики калиброванной модели

bonus_df = pd.DataFrame([tuned_row, te_row, cal_row])                     # сводка бонусов
print(bonus_df.round(4).to_string(index=False))                           # таблица бонусов

# Brier score: чем меньше, тем лучше откалиброваны вероятности
# (class_weight='balanced' систематически завышает вероятность покупки)
brier_before = brier_score_loss(y_test, y_proba_best)                     # Brier до калибровки
brier_after = brier_score_loss(y_test, calibrated.predict_proba(X_test)[:, 1])   # Brier после калибровки
print('\nBrier score LightGBM (Optuna):             %.4f' % brier_before)  # Brier до калибровки
print('Brier score LightGBM (Optuna) + калибровка: %.4f' % brier_after)    # Brier после калибровки

import shap                                              # SHAP: объяснение предсказаний модели

explainer = shap.TreeExplainer(best_lgbm)                # быстрый точный explainer для деревьев
shap_values = explainer.shap_values(X_test)              # SHAP-значения на тестовой выборке

# Разные версии shap возвращают разные форматы: список по классам, 2D- или 3D-массив.
# Приводим всё к виду «значения для класса 1 (покупка)» и скалярной базовой величине.
expected = np.atleast_1d(explainer.expected_value)       # базовое значение (среднее предсказание)
if isinstance(shap_values, list):                        # формат 1: список по классам
    shap_values_class1 = np.asarray(shap_values[1])      # берём класс «покупка»
    base_value = float(expected[1] if len(expected) > 1 else expected[0])   # базовое значение для класса 1
else:                                                    # иначе — массив
    shap_values = np.asarray(shap_values)                # приводим к numpy
    if shap_values.ndim == 3:                            # (объекты, признаки, классы)
        shap_values_class1 = shap_values[:, :, 1]        # 3D → класс 1
        base_value = float(expected[1] if len(expected) > 1 else expected[0])   # базовое значение для класса 1
    else:                                                # (объекты, признаки)
        shap_values_class1 = shap_values                 # 2D — уже для класса 1
        base_value = float(expected[-1])                 # базовое значение

print('Форма SHAP-значений:', shap_values_class1.shape, ' базовое значение (log-odds): %.3f' % base_value)   # проверка размерности

# Summary plot (dot): глобальная важность и направление влияния
shap.summary_plot(shap_values_class1, X_test, plot_type='dot', show=False)   # график: важность и направление влияния
plt.title('SHAP summary plot (LightGBM, Optuna)')                            # заголовок графика
plt.savefig('examples/shap_summary.png', dpi=120, bbox_inches='tight')       # сохраняем картинку
plt.show()                                                                   # показываем график

# Bar plot: средний модуль SHAP-значений
shap.summary_plot(shap_values_class1, X_test, plot_type='bar', show=False)   # график: средний |SHAP|
plt.title('Средний |SHAP| по признакам')                                     # заголовок графика
plt.show()                                                                   # показываем график

# Waterfall plot: сессия с наибольшей предсказанной вероятностью покупки
best_idx = int(np.argmax(y_proba_best))                  # индекс сессии с максимальной вероятностью покупки
print(f'Сессия #{best_idx} (в тестовой выборке), вероятность покупки = {y_proba_best[best_idx]:.4f}, '
      f'истинная метка = {int(y_test.iloc[best_idx])}')  # индекс, вероятность и фактический исход

explanation = shap.Explanation(  # объект для waterfall
    values=shap_values_class1[best_idx],                 # вклады признаков
    base_values=base_value,                              # стартовое значение
    data=X_test.iloc[best_idx].values,                   # значения признаков сессии
    feature_names=X_test.columns.tolist())               # названия признаков

shap.plots.waterfall(explanation, max_display=12, show=False)                # рисуем waterfall
plt.savefig('examples/shap_waterfall.png', dpi=120, bbox_inches='tight')     # сохраняем картинку
plt.show()                                                                   # показываем график

# Топ-3 вклада текстом (для выводов)
contrib = pd.Series(shap_values_class1[best_idx], index=X_test.columns)      # вклады признаков для этой сессии
for feat in contrib.abs().sort_values(ascending=False).head(3).index:        # три самых сильных вклада
    print(f'  {feat} = {X_test.iloc[best_idx][feat]:.4g}: вклад {contrib[feat]:+.3f} (log-odds)')   # признак, значение, вклад

# Dependence plot: самый важный по SHAP признак
mean_abs_shap = pd.Series(np.abs(shap_values_class1).mean(axis=0), index=X_test.columns).sort_values(ascending=False)   # средний модуль SHAP по признакам
top_feature = mean_abs_shap.index[0]                     # самый влиятельный признак
print('Топ-5 признаков по среднему |SHAP|:', [mark(f) for f in mean_abs_shap.head(5).index])   # топ-5 по SHAP (★ — созданные нами)

shap.dependence_plot(top_feature, shap_values_class1, X_test, show=False)    # зависимость вклада от значения признака
plt.title(f'SHAP dependence plot: {top_feature}')                            # заголовок графика
plt.show()                                                                   # показываем график

os.makedirs('models', exist_ok=True)                     # папка для моделей

# Модели
joblib.dump(rf_model, 'models/rf_model.pkl')             # сохраняем в .pkl
joblib.dump(xgb_model, 'models/xgb_model.pkl')           # XGBoost
joblib.dump(lgbm_model, 'models/lgbm_model.pkl')         # LightGBM (baseline)
joblib.dump(cat_model, 'models/cat_model.pkl')           # CatBoost
joblib.dump(best_lgbm, 'models/best_model.pkl')          # лучшая: LightGBM после Optuna

# Препроцессинг: scaler
joblib.dump(scaler, 'models/scaler.pkl')                 # scaler обучен на train

# Список признаков (порядок важен при предсказании!)
with open('models/feature_cols.json', 'w', encoding='utf-8') as f:   # список признаков в JSON
    json.dump(feature_cols, f, ensure_ascii=False, indent=2)         # порядок признаков важен при предсказании

# Метрики всех моделей + время обучения + результаты Optuna
def metrics_of(row_df, name):  # метрики одной модели из таблицы
    # Метрики одной модели из таблицы
    row = row_df[row_df['Model'] == name].iloc[0]        # строка нужной модели
    return {k: float(row[k]) for k in ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']}   # числа → float для JSON

metrics = {  # словарь метрик для metrics.json
    'random_forest': {**metrics_of(results_df, 'Random Forest'), 'train_time_sec': float(train_times['Random Forest'])},   # Random Forest
    'xgboost':       {**metrics_of(results_df, 'XGBoost'),       'train_time_sec': float(train_times['XGBoost'])},         # XGBoost
    'lightgbm':      {**metrics_of(results_df, 'LightGBM'),      'train_time_sec': float(train_times['LightGBM'])},        # LightGBM
    'catboost':      {**metrics_of(results_df, 'CatBoost'),      'train_time_sec': float(train_times['CatBoost'])},        # CatBoost
    'lightgbm_optuna': {k: float(v) for k, v in tuned_row.items() if k != 'Model'},    # тюнингованный LightGBM
    'best_model': 'LightGBM (Optuna)',                    # название лучшей модели
    'best_roc_auc': float(study.best_value),              # как в методичке: лучший CV ROC-AUC Optuna
    'best_cv_roc_auc': float(study.best_value),           # CV ROC-AUC после Optuna
    'baseline_cv_roc_auc': float(baseline_cv),            # CV ROC-AUC baseline
    'best_test_roc_auc': float(auc_tuned),                # ROC-AUC на test
    'n_trials': N_TRIALS,                                 # число испытаний
    'best_params': {k: (float(v) if isinstance(v, float) else int(v)) for k, v in study.best_params.items()},   # лучшие гиперпараметры
    'fixed_params': {'subsample_freq': 1, 'class_weight': 'balanced', 'random_state': RANDOM_STATE},            # фиксированные параметры
    'new_features': new_features,                         # созданные признаки
    'feature_engineering_effect': {                       # сравнение «без / с созданными признаками»
        'base_features': int(ablation_df.loc[0, 'Число признаков']),
        'all_features': int(ablation_df.loc[1, 'Число признаков']),
        'cv_roc_auc_base': float(ablation_df.loc[0, 'CV ROC-AUC']),
        'cv_roc_auc_all': float(ablation_df.loc[1, 'CV ROC-AUC']),
        'test_roc_auc_base': float(ablation_df.loc[0, 'Test ROC-AUC']),
        'test_roc_auc_all': float(ablation_df.loc[1, 'Test ROC-AUC'])},
    'bonus': {'target_encoding': {k: float(v) for k, v in te_row.items() if k != 'Model'},     # Target Encoding
              'calibrated': {k: float(v) for k, v in cal_row.items() if k != 'Model'},         # калибровка
              'brier_before': float(brier_before), 'brier_after': float(brier_after)},         # Brier до и после
}
with open('models/metrics.json', 'w', encoding='utf-8') as f:        # записываем метрики
    json.dump(metrics, f, indent=2, ensure_ascii=False)              # красивый JSON, кириллица без экранирования

print('✅ Все модели и метаданные сохранены:')                        # итог сохранения
for fn in sorted(os.listdir('models')):                              # список сохранённых файлов
    print(f'  models/{fn}  ({os.path.getsize("models/" + fn) / 1024:.0f} КБ)')   # файл и размер

# README.md модуля с таблицей результатов (генерируется из актуальных метрик)
readme = f"""# 🛒 Module 4 — Online Shoppers: Инженерия признаков

**Автор:** [ФИО], [группа]
**Дата:** 2026-11-__

## 📊 Результаты (test, порог 0.5)

{md_table(final_df)}

Лучшая модель: **LightGBM (Optuna)**, CV ROC-AUC = {study.best_value:.4f}, test ROC-AUC = {auc_tuned:.4f}.

Созданных признаков: {len(new_features)} ({', '.join(new_features)}).
Эффект признаков (test ROC-AUC): без созданных — {ablation_df.loc[0, 'Test ROC-AUC']:.4f}, со всеми — {ablation_df.loc[1, 'Test ROC-AUC']:.4f}.

## 🚀 Быстрый старт

```python
import joblib, requests
from io import BytesIO

BASE_URL = "https://raw.githubusercontent.com/[username]/ml-course-[фамилия]/main/module-4-features"
model = joblib.load(BytesIO(requests.get(f"{{BASE_URL}}/models/best_model.pkl").content))
```

## 📁 Структура
- `notebook.ipynb` — полный отчёт
- `models/` — 5 моделей, scaler, список признаков, метрики
- `data/` — датасет и описание
- `examples/` — графики EDA, ROC, Feature Importance, SHAP
- `requirements.txt` — зависимости
"""
with open('README.md', 'w', encoding='utf-8') as f:      # записываем README модуля
    f.write(readme)                                      # текст README
print('✅ README.md модуля сохранён')                     # подтверждение

import requests                                         # HTTP-запросы (загрузка файлов из GitHub)
from io import BytesIO                                   # чтение байтов как файла (для joblib.load)

# Замените на свои данные
BASE_URL = 'https://raw.githubusercontent.com/username/ml-course-ivanov/main/module-4-features'   # базовая raw-ссылка репозитория

def load_from_repo(path, kind='pkl'):  # скачиваем файл; если нельзя — берём локальный
    # Скачиваем файл; если репозиторий недоступен, берём локальную копию
    url = f'{BASE_URL}/{path}'                           # полная ссылка на файл
    try:                                                 # пробуем скачать
        resp = requests.get(url, timeout=30)             # GET-запрос
        resp.raise_for_status()                          # ошибка при коде ответа ≥ 400
        print(f'✅ Загружено из репозитория: {url}')       # успех
        return joblib.load(BytesIO(resp.content)) if kind == 'pkl' else resp.json()   # .pkl — через joblib, .json — как JSON
    except Exception as e:                               # репозиторий недоступен
        print(f'⚠️ Не удалось загрузить {url}: {e}\n   Использую локальный файл {path}')   # предупреждение пользователю
        if kind == 'pkl':                                # локальный .pkl
            return joblib.load(path)                     # загрузка с диска
        with open(path, encoding='utf-8') as f:          # локальный .json
            return json.load(f)                          # чтение JSON

best_lgbm_loaded = load_from_repo('models/best_model.pkl', 'pkl')            # модель из репозитория
feature_cols_loaded = load_from_repo('models/feature_cols.json', 'json')     # список признаков из репозитория
print(f'✅ Загружено признаков: {len(feature_cols_loaded)}')                  # число признаков

# Проверка: предсказания загруженной и исходной модели на 3 сессиях должны совпасть
X_check = X_test[feature_cols_loaded].iloc[:3]           # порядок столбцов — из загруженного списка признаков

y_pred_loaded = best_lgbm_loaded.predict(X_check)                # классы загруженной модели
y_pred_original = best_lgbm.predict(X_check)                     # классы исходной модели
proba_loaded = best_lgbm_loaded.predict_proba(X_check)[:, 1]     # вероятности загруженной модели
proba_original = best_lgbm.predict_proba(X_check)[:, 1]          # вероятности исходной модели

print(f'Загруженная модель:  {y_pred_loaded}  вероятности: {np.round(proba_loaded, 4)}')   # результат загруженной модели
print(f'Исходная модель:     {y_pred_original}  вероятности: {np.round(proba_original, 4)}')   # результат исходной модели
print(f'Классы совпадают: {np.array_equal(y_pred_loaded, y_pred_original)}')               # проверка классов
print(f'Вероятности совпадают: {np.allclose(proba_loaded, proba_original)}')               # проверка вероятностей

# Автоматическая сводка ключевых чисел для выводов
print(f"Лучшая модель на этапе сравнения: {best_name} — ROC-AUC {results_df['ROC-AUC'].max():.4f}")   # лучшая модель сравнения
print(f"LightGBM baseline → Optuna (test ROC-AUC): {auc_base:.4f} → {auc_tuned:.4f} ({auc_tuned - auc_base:+.4f})")   # эффект Optuna
print(f"Метрики итоговой модели: " + ', '.join(f'{k}={v:.3f}' for k, v in tuned_row.items() if k != 'Model'))   # метрики итоговой модели
print(f"Эффект созданных признаков (test ROC-AUC): {ablation_df.loc[0, 'Test ROC-AUC']:.4f} → {ablation_df.loc[1, 'Test ROC-AUC']:.4f} ({delta_test:+.4f})")   # эффект признаков
print(f"Target Encoding ROC-AUC: {te_row['ROC-AUC']:.4f}; калибровка ROC-AUC: {cal_row['ROC-AUC']:.4f}; Brier {brier_before:.4f} → {brier_after:.4f}")   # бонусы

# requirements.txt с версиями, под которыми реально запускался ноутбук
from importlib.metadata import version, PackageNotFoundError   # версии установленных пакетов

packages = ['pandas', 'numpy', 'scikit-learn', 'xgboost', 'lightgbm', 'catboost',  # библиотеки проекта
            'optuna', 'shap', 'matplotlib', 'seaborn', 'joblib', 'requests']   # библиотеки проекта
lines = []                                               # строки requirements
for p in packages:                                       # версия каждой библиотеки
    try:                                                 # пробуем узнать версию
        lines.append(f'{p}=={version(p)}')               # pin версии
    except PackageNotFoundError:                         # пакет не найден
        lines.append(p)                                  # без версии

with open('requirements.txt', 'w', encoding='utf-8') as f:    # записываем requirements.txt
    f.write('\n'.join(lines) + '\n')                          # по строке на пакет
print('\n'.join(lines))                                       # показываем результат

# Выгрузка всего кода ноутбука в .py (для приложения)
if os.path.exists('notebook.ipynb'):                          # ноутбук сохранён рядом
    with open('notebook.ipynb', encoding='utf-8') as f:       # читаем сам ноутбук
        nb = json.load(f)                                     # JSON ноутбука
    code_cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']   # тексты ячеек с кодом
    with open('notebook_code.py', 'w', encoding='utf-8') as f:     # пишем .py
        f.write('\n\n# ' + '-' * 60 + '\n\n'.join([''] + code_cells))   # склеиваем ячейки через разделитель
    print(f'\n✅ notebook_code.py сохранён ({len(code_cells)} ячеек кода)')   # подтверждение