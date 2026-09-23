# ============================================================
# src/data_checks.py — проверки качества данных (шаг 3, ЛР №1)
# Каждая функция отвечает за один пункт раздела 3.3 методички.
# ============================================================

import numpy as np
import pandas as pd

# Числовые признаки и целевая переменная датасета Fish.
NUMERIC_COLS = ["Weight", "Length1", "Length2", "Length3", "Height", "Width"]
TARGET = "Weight"


# ------------------------------------------------------------
# 3.1 Форма, типы, дубликаты
# ------------------------------------------------------------

def check_shape_dtypes(df, expected=None):
    """
    Что делает: показывает форму таблицы, типы столбцов,
                 число непустых и уникальных значений, сверяет схему
                 с ожидаемым списком признаков.
    Зачем: первичный контроль — совпадает ли структура файла с паспортом.
    """
    info = pd.DataFrame({
        "dtype": df.dtypes.astype(str),          # тип каждого столбца
        "Непустых": df.notna().sum(),            # сколько не-NaN
        "Уникальных": df.nunique(),              # сколько различных значений
    })
    print(f"Форма: {df.shape}")
    if expected:
        print("Нет:", set(expected) - set(df.columns) or "—")   # отсутствующие
        print("Лишние:", set(df.columns) - set(expected) or "—") # лишние
    return info


def check_duplicates(df, id_candidates=None):
    """
    Что делает: считает полные дубликаты строк, дубликаты индекса
                 и дубликаты значений в столбцах-кандидатах в ID.
    Зачем: дубликаты приводят к утечке при разбиении (один объект
           попадает и в train, и в test).
    """
    r = {
        "дубликаты строк": int(df.duplicated().sum()),
        "дубликаты индекса": int(df.index.duplicated().sum()),
    }
    for c in (id_candidates or []):
        if c in df.columns:
            r[f"дубликаты «{c}»"] = int(df[c].duplicated().sum())
    return pd.Series(r, name="Значение").to_frame()


# ------------------------------------------------------------
# 3.2 Пропуски, бесконечности, невозможные значения
# ------------------------------------------------------------

def check_missing_inf_impossible(df, numeric_cols, max_abs=None):
    """
    Что делает: по каждому столбцу считает пропуски (NaN),
                 бесконечности (±inf), отрицательные и нули;
                 опционально проверяет превышение модуля max_abs.
    Зачем: 0, отрицательные и inf физически невозможны для длин
           и массы — это ошибки ввода или парсинга.
    """
    rows = []
    for c in df.columns:
        nan = int(df[c].isna().sum())
        inf = neg = zero = 0
        if c in numeric_cols:
            n = pd.to_numeric(df[c], errors="coerce")
            inf = int(np.isinf(n).sum())
            neg = int((n < 0).sum())
            zero = int((n == 0).sum())
        row = {"Признак": c, "Пропуски": nan, "Бесконечности": inf,
               "Отрицательных": neg, "Нулей": zero}
        if max_abs is not None and c in numeric_cols:
            row["Превышение |max|"] = bool(abs(df[c].max()) > max_abs)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Признак")


# ------------------------------------------------------------
# 3.3 Согласованность геометрических измерений
# ------------------------------------------------------------

def check_geometric_consistency(df, weight="Weight"):
    """
    Что делает: проверяет анатомическую согласованность:
                 Length1 ≤ Length2 ≤ Length3,
                 Height и Width меньше Length3,
                 Weight > 0,
                 плотность Weight / (Length3·Height·Width) в [0.5, 2.0].
    Зачем: несоответствия сигналят об ошибках ввода или смешении
           единиц измерения (см/мм, г/кг).
    """
    l1, l2, l3 = df["Length1"], df["Length2"], df["Length3"]
    v = l3 * df["Height"] * df["Width"]           # приближённый объём
    density = df[weight] / v                       # приближённая плотность

    checks = pd.DataFrame({
        "L1 > L2": l1 > l2,                        # нарушение порядка длин
        "L2 > L3": l2 > l3,
        "Height > L3": df["Height"] > l3,          # поперечный больше длины
        "Width > L3": df["Width"] > l3,
        "Weight ≤ 0": df[weight] <= 0,             # неположительная масса
        "Плотность вне [0.5, 2.0]": (density < 0.5) | (density > 2.0),
    })
    return checks, checks.sum().to_frame("Нарушений"), density


# ------------------------------------------------------------
# 3.4 Редкие категории и дисбаланс
# ------------------------------------------------------------

def check_categorical_balance(df, col, min_count=10):
    """
    Что делает: для категориального признака считает частоты,
                 доли, помечает редкие классы (< min_count),
                 печатает коэффициент дисбаланса max/min.
    Зачем: дисбаланс классов искажает обучение (модель «любит»
           частые виды) и требует стратификации и macro-метрик.
    """
    vc = df[col].value_counts()
    out = pd.DataFrame({
        "Количество": vc.values,
        "Доля, %": (vc / len(df) * 100).round(2),
        f"Редкая (<{min_count})": vc.values < min_count,
    }, index=vc.index)
    print(f"Классов: {df[col].nunique()} | Дисбаланс: {vc.max()/vc.min():.1f}×")
    return out


# ------------------------------------------------------------
# 3.5 Почти одинаковые строки
# ------------------------------------------------------------

def find_near_duplicates(df, numeric_cols, tol=0.005):
    """
    Что делает: ищет пары строк, у которых нормированное евклидово
                 расстояние по числовым признакам меньше tol.
                 Каждый признак масштабируется к [0, 1] по (x−min)/(max−min).
    Зачем: почти одинаковые строки — потенциальный источник утечки
           (например, повторное измерение) и должны проверяться
           до разбиения на train/test.
    """
    cols = ["i", "j", "расстояние", "Species_i", "Species_j"]
    X = df[numeric_cols].astype(float).values
    rng = X.max(0) - X.min(0)
    rng[rng == 0] = 1.0                            # защита от деления на ноль
    Xn = X / rng

    pairs = []
    for i in range(len(Xn)):
        d = np.linalg.norm(Xn[i+1:] - Xn[i], axis=1)   # расстояние до всех j > i
        for j in np.where(d < tol)[0] + i + 1:
            pairs.append({
                "i": i, "j": int(j),
                "расстояние": round(float(d[j-i-1]), 5),
                "Species_i": df.iloc[i]["Species"],
                "Species_j": df.iloc[j]["Species"],
            })
    # Пустой результат тоже должен содержать колонки.
    return pd.DataFrame(pairs, columns=cols).sort_values("расстояние") \
           if pairs else pd.DataFrame(columns=cols)


# ------------------------------------------------------------
# 3.6 Уникальные поля: ID-кандидаты и источники утечки
# ------------------------------------------------------------

def check_identifier_candidates(df, threshold=0.9):
    """
    Что делает: считает долю уникальных значений по каждому столбцу.
                 Если доля ≥ threshold — помечает как ID-кандидата.
    Зачем: идентификаторы нужно исключать из признаков, иначе модель
           запомнит объекты по номеру строки и «узнает» их в тесте.
    """
    ratio = df.nunique(dropna=False) / len(df)
    return pd.DataFrame({
        "Уникальных": df.nunique(),
        "Доля": ratio.round(3),
        "Похоже на ID": ratio >= threshold,
        "dtype": df.dtypes.astype(str),
    }).sort_values("Доля", ascending=False)


def check_leakage_candidates(df, target_col=TARGET, threshold=0.98):
    """
    Что делает: считает корреляцию Пирсона каждого числового признака
                 с целевой переменной и помечает |r| > threshold.
    Зачем: слишком сильная связь с целевой — сигнал возможной утечки
           (признак получен из целевой или содержит постфактум-информацию).
    """
    corr = df.select_dtypes("number").corr()[target_col].drop(target_col)
    s = corr[corr.abs() > threshold]
    return pd.DataFrame({"Признак": s.index, "r": s.values.round(4)})


# ------------------------------------------------------------
# Прогон всех проверок шага 3
# ------------------------------------------------------------

def run_all_checks(df, expected=None, numeric_cols=NUMERIC_COLS, cat_col="Species"):
    """
    Что делает: последовательно вызывает все проверки шага 3
                 и печатает их результаты в консоль.
    Зачем: одна точка входа для ноутбука и для CI.
    """
    print("=== 3.1 Форма, типы, дубликаты ===")
    print(check_shape_dtypes(df, expected))
    print("\n", check_duplicates(df, [cat_col]))

    print("\n=== 3.2 Пропуски / inf / невозможные ===")
    print(check_missing_inf_impossible(df, numeric_cols, max_abs=5000))

    print("\n=== 3.3 Согласованность измерений ===")
    _, summary, density = check_geometric_consistency(df)
    print(summary)
    print("Плотность:\n", density.describe().round(3))

    print("\n=== 3.4 Дисбаланс категорий ===")
    print(check_categorical_balance(df, cat_col))

    print("\n=== 3.5 Почти одинаковые строки ===")
    near = find_near_duplicates(df, numeric_cols, tol=0.005)
    print(f"Пар (tol=0.5%): {len(near)}")

    print("\n=== 3.6 ID-кандидаты и утечки ===")
    print(check_identifier_candidates(df))
    leak = check_leakage_candidates(df)
    print("\n|r| > 0.98 к Weight:")
    print(leak if len(leak) else "— не обнаружено")