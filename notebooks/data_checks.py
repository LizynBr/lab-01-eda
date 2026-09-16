# ============================================================
# ЛАБОРАТОРНАЯ РАБОТА: EDA датасета Fish Market
# Формат вектора признаков:
#   Species, Weight, Length1, Length2, Length3, Height, Width
#
# Запуск:  python fish_eda.py
# Требуется файл Fish.csv в той же папке.
# ============================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # без GUI; графики сохраняются в файлы
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# -------------------- Глобальные настройки --------------------
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11
pd.set_option("display.width", 140)

RANDOM_STATE = 42
DATA_FILE = "C:\\ktp\\ml\\lab-01-eda\\notebooks\\Fish.csv"
OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

FEATURES = ["Species", "Weight", "Length1", "Length2", "Length3", "Height", "Width"]
NUMERIC_COLS = ["Weight", "Length1", "Length2", "Length3", "Height", "Width"]
TARGET = "Weight"


# ============================================================
# Хелперы вывода
# ============================================================

def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def savefig(name: str) -> None:
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[сохранён график] {path}")


# ============================================================
# ШАГ 1. Загрузка и первичное описание
# ============================================================

def load_data(path: str = DATA_FILE) -> pd.DataFrame:
    if not os.path.exists(path):
        sys.exit(f"Файл {path} не найден. Положите Fish.csv рядом со скриптом.")
    df = pd.read_csv(path)
    missing = set(FEATURES) - set(df.columns)
    if missing:
        sys.exit(f"В таблице отсутствуют ожидаемые признаки: {missing}")
    return df[FEATURES].copy()


def describe_data(df: pd.DataFrame) -> None:
    section("ШАГ 1. ПЕРВИЧНОЕ ОПИСАНИЕ ДАННЫХ")
    print("Форма таблицы:", df.shape)
    print("\nТипы данных:\n", df.dtypes)
    print("\nПервые 5 строк:\n", df.head())
    print("\nОписательные статистики:\n", df.describe(include="all").T)
    print("\nЧисленность по видам:\n", df["Species"].value_counts())


# ============================================================
# ШАГ 3. АВТОМАТИЧЕСКАЯ ПРОВЕРКА КАЧЕСТВА
# ============================================================

def check_shape_dtypes(df: pd.DataFrame, expected_features=None) -> pd.DataFrame:
    info = pd.DataFrame({
        "Признак": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "Непустых": df.notna().sum().values,
        "Уникальных": df.nunique().values,
    })
    print(f"Форма таблицы: {df.shape[0]} строк × {df.shape[1]} столбцов")
    if expected_features is not None:
        missing = set(expected_features) - set(df.columns)
        extra = set(df.columns) - set(expected_features)
        print(f"Отсутствуют ожидаемые признаки: {missing or '—'}")
        print(f"Лишние столбцы:                 {extra or '—'}")
    return info


def check_duplicates(df: pd.DataFrame, subset=None, id_candidates=None) -> pd.Series:
    result = {
        "полные дубликаты": int(df.duplicated(subset=subset).sum()),
        "дубликаты без учёта индекса": int(df.reset_index(drop=True).duplicated().sum()),
    }
    if id_candidates:
        for col in id_candidates:
            if col in df.columns:
                result[f"дубликаты в «{col}»"] = int(df[col].duplicated().sum())
    return pd.Series(result, name="Значение").to_frame()


def check_missing_inf_impossible(df, numeric_cols, non_negative_cols=None,
                                 positive_cols=None, max_abs=None) -> pd.DataFrame:
    non_negative_cols = non_negative_cols or []
    positive_cols = positive_cols or []
    rows = []
    for col in df.columns:
        s = df[col]
        nan_cnt = int(s.isna().sum())
        inf_cnt = neg_cnt = zero_cnt = 0
        if col in numeric_cols:
            num = pd.to_numeric(s, errors="coerce")
            inf_cnt = int(np.isinf(num).sum())
            neg_cnt = int((num < 0).sum())
            zero_cnt = int((num == 0).sum())
        rows.append({
            "Признак": col,
            "Пропуски": nan_cnt,
            "Бесконечности": inf_cnt,
            "Отрицательных": neg_cnt,
            "Нулей": zero_cnt,
            "Нарушение non_negative": col in non_negative_cols and neg_cnt > 0,
            "Нарушение positive": col in positive_cols and (neg_cnt + zero_cnt) > 0,
        })
    report = pd.DataFrame(rows)
    if max_abs is not None:
        report["Превышение |max|"] = [
            bool(abs(df[c].max()) > max_abs) if c in numeric_cols else False
            for c in df.columns
        ]
    return report


def check_geometric_consistency(df,
                                ordered_lengths=("Length1", "Length2", "Length3"),
                                height="Height", width="Width",
                                weight="Weight"):
    checks = pd.DataFrame(index=df.index)
    l1, l2, l3 = [df[c] for c in ordered_lengths]
    checks["L1 > L2"] = l1 > l2
    checks["L2 > L3"] = l2 > l3
    checks["Height > Length3"] = df[height] > l3
    checks["Width > Length3"] = df[width] > l3
    checks["Weight <= 0"] = df[weight] <= 0
    v = l3 * df[height] * df[width]
    density = df[weight] / v
    checks["Плотность вне [0.5, 2.0]"] = (density < 0.5) | (density > 2.0)
    summary = checks.sum().to_frame(name="Нарушений, шт.")
    summary["Строки-нарушители"] = [
        list(checks.index[checks[col]])[:10] for col in checks.columns
    ]
    return checks, summary, density


def check_categorical_balance(df, cat_col, min_count=10):
    vc = df[cat_col].value_counts()
    share = (vc / len(df) * 100).round(2)
    out = pd.DataFrame({
        "Количество": vc.values,
        "Доля, %": share.values,
        f"Редкая (<{min_count})": vc.values < min_count,
    }, index=vc.index)
    imbalance_ratio = vc.max() / vc.min()
    print(f"Классов:                {df[cat_col].nunique()}")
    print(f"Самый частый:           {vc.idxmax()} ({vc.max()})")
    print(f"Самый редкий:           {vc.idxmin()} ({vc.min()})")
    print(f"Коэффициент дисбаланса: {imbalance_ratio:.1f}×")
    return out, imbalance_ratio

def find_near_duplicates(df, numeric_cols, tol=0.005) -> pd.DataFrame:
    """
    Поиск почти одинаковых строк по нормированному евклидову расстоянию.
    Возвращает DataFrame с колонками:
        i, j, расстояние, Species_i, Species_j, Weight_i, Weight_j.
    Если совпадений нет — пустой DataFrame с теми же колонками.
    """
    columns = ["i", "j", "расстояние", "Species_i", "Species_j",
               "Weight_i", "Weight_j"]

    X = df[numeric_cols].astype(float).values
    rng = X.max(axis=0) - X.min(axis=0)
    rng[rng == 0] = 1.0            # защита от деления на ноль
    Xn = X / rng

    pairs = []
    n = len(Xn)
    for i in range(n):
        d = np.linalg.norm(Xn[i + 1:] - Xn[i], axis=1)
        close = np.where(d < tol)[0] + (i + 1)
        for j in close:
            pairs.append({
                "i": i,
                "j": int(j),
                "расстояние": round(float(d[j - (i + 1)]), 5),
                "Species_i": df.iloc[i]["Species"],
                "Species_j": df.iloc[j]["Species"],
                "Weight_i": df.iloc[i]["Weight"],
                "Weight_j": df.iloc[j]["Weight"],
            })

    # КЛЮЧЕВОЕ: пустой результат тоже должен иметь правильные колонки
    if not pairs:
        return pd.DataFrame(columns=columns)

    return (pd.DataFrame(pairs)
              .sort_values("расстояние")
              .reset_index(drop=True))


def check_identifier_candidates(df, id_like_threshold=0.9) -> pd.DataFrame:
    rows = []
    n = len(df)
    for col in df.columns:
        uniq = df[col].nunique(dropna=False)
        ratio = uniq / n
        rows.append({
            "Признак": col,
            "Уникальных": uniq,
            "Доля уникальных": round(ratio, 3),
            "Похоже на ID": ratio >= id_like_threshold,
            "dtype": str(df[col].dtype),
        })
    return pd.DataFrame(rows).sort_values("Доля уникальных", ascending=False)


def check_leakage_candidates(df, target_col) -> pd.DataFrame:
    num = df.select_dtypes(include=np.number)
    if target_col not in num.columns:
        return pd.DataFrame()
    corr = num.corr()[target_col].drop(target_col)
    suspicious = corr[corr.abs() > 0.98]
    return pd.DataFrame({
        "Признак": suspicious.index,
        "r с целевой": suspicious.values.round(4),
        "Риск утечки": "Высокий",
    })


def full_quality_report(df, numeric_cols, cat_cols, target_col) -> dict:
    report = {}
    report["shape"] = df.shape
    report["dtypes"] = df.dtypes.astype(str).to_dict()
    report["full_duplicates"] = int(df.duplicated().sum())

    num = df[numeric_cols]
    report["missing"] = int(num.isna().sum().sum())
    report["inf"] = int(np.isinf(num.astype(float)).sum().sum())
    report["negative"] = int((num < 0).sum().sum())
    report["zero"] = int((num == 0).sum().sum())

    l1, l2, l3 = df["Length1"], df["Length2"], df["Length3"]
    report["L1>L2"] = int((l1 > l2).sum())
    report["L2>L3"] = int((l2 > l3).sum())

    vc = df[cat_cols[0]].value_counts()
    report["imbalance_ratio"] = round(vc.max() / vc.min(), 2)
    report["rare_classes"] = int((vc < 10).sum())

    report["near_duplicates_0.5%"] = len(find_near_duplicates(df, numeric_cols, tol=0.005))

    corr = df[numeric_cols].corr()[target_col].drop(target_col)
    report["features_|r|>0.98"] = list(corr[corr.abs() > 0.98].index)
    return report


def run_quality_checks(df: pd.DataFrame) -> None:
    section("ШАГ 3. АВТОМАТИЧЕСКАЯ ПРОВЕРКА КАЧЕСТВА ДАННЫХ")

    print("\n--- 3.1 Форма, типы, дубликаты ---")
    print(check_shape_dtypes(df, expected_features=FEATURES))
    print("\n", check_duplicates(df, id_candidates=["Species"]))

    print("\n--- 3.2 Пропуски, бесконечности, невозможные значения ---")
    qr = check_missing_inf_impossible(
        df,
        numeric_cols=NUMERIC_COLS,
        non_negative_cols=NUMERIC_COLS,
        positive_cols=NUMERIC_COLS,
        max_abs=5000,
    )
    print(qr)

    print("\n--- 3.3 Согласованность геометрических измерений ---")
    _, geo_summary, density = check_geometric_consistency(df)
    print(geo_summary)
    print("\nСтатистика плотности Weight/(L3·H·W):")
    print(density.describe().round(3))

    print("\n--- 3.4 Редкие категории и дисбаланс видов ---")
    balance_table, imbalance = check_categorical_balance(df, "Species", min_count=10)
    print(balance_table)

    print("\n--- 3.5 Почти одинаковые строки (near-duplicates) ---")
    near_dups = find_near_duplicates(df, numeric_cols=NUMERIC_COLS, tol=0.005)
    print(f"Найдено почти одинаковых пар (tol=0.5%): {len(near_dups)}")
    if len(near_dups):
        print(near_dups.head(15))

    print("\n--- 3.6 Уникальные поля и источники утечки ---")
    print(check_identifier_candidates(df))
    leak = check_leakage_candidates(df, target_col=TARGET)
    print("\nПризнаки с подозрительно высокой корреляцией с Weight:")
    print(leak if len(leak) else "— не обнаружено")

    print("\n--- 3.7 Сводный отчёт ---")
    final = full_quality_report(df, NUMERIC_COLS, ["Species"], TARGET)
    for k, v in final.items():
        print(f"{k:28s}: {v}")


# ============================================================
# ШАГ 2 (графики). Визуальный анализ
# ============================================================

def plot_target_distribution(df: pd.DataFrame) -> None:
    section("ГРАФИК 1. РАСПРЕДЕЛЕНИЕ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df[TARGET], kde=True, bins=25, ax=axes[0], color="steelblue")
    axes[0].set_title("Распределение массы рыбы (Weight)")
    axes[0].set_xlabel("Масса, г")
    axes[0].set_ylabel("Число наблюдений, шт.")

    sns.histplot(np.log1p(df[TARGET]), kde=True, bins=25, ax=axes[1], color="seagreen")
    axes[1].set_title("Распределение log(1 + Weight)")
    axes[1].set_xlabel("log(1 + Масса), безразмерная величина")
    axes[1].set_ylabel("Число наблюдений, шт.")
    plt.tight_layout()
    savefig("01_target_distribution.png")

    print(f"Skew (исходная): {df[TARGET].skew():.3f}")
    print(f"Skew (лог):      {np.log1p(df[TARGET]).skew():.3f}")
    print(f"Shapiro p (исходная): {stats.shapiro(df[TARGET]).pvalue:.4e}")
    print(f"Shapiro p (лог):      {stats.shapiro(np.log1p(df[TARGET])).pvalue:.4e}")
    print("ВЫВОД: исходное распределение скошено вправо; логарифм приближает к нормальному.")


def plot_species_count_and_weight(df: pd.DataFrame) -> None:
    section("ГРАФИК 2. ЧИСЛЕННОСТЬ И РАСПРЕДЕЛЕНИЕ ПО ВИДАМ")
    order = df["Species"].value_counts().index
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.countplot(data=df, x="Species", ax=axes[0], palette="Set2", order=order)
    axes[0].set_title("Численность наблюдений по видам рыб")
    axes[0].set_xlabel("Вид (Species)")
    axes[0].set_ylabel("Количество, шт.")
    axes[0].tick_params(axis="x", rotation=30)
    for p in axes[0].patches:
        axes[0].annotate(int(p.get_height()),
                         (p.get_x() + p.get_width() / 2, p.get_height()),
                         ha="center", va="bottom", fontsize=10)

    sns.boxplot(data=df, x="Species", y="Weight", ax=axes[1],
                palette="Set2", order=order)
    axes[1].set_title("Распределение массы по видам")
    axes[1].set_xlabel("Вид (Species)")
    axes[1].set_ylabel("Масса, г")
    axes[1].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    savefig("02_species_count_weight.png")
    print("ВЫВОД: классы несбалансированы (~9×); нужны стратификация и macro-метрики.")


def plot_pairplot(df: pd.DataFrame) -> None:
    section("ГРАФИК 3. МАТРИЦА ПАРНЫХ ГРАФИКОВ")
    g = sns.pairplot(df[NUMERIC_COLS + ["Species"]], hue="Species",
                     diag_kind="kde", corner=True,
                     plot_kws={"alpha": 0.6, "s": 25})
    g.fig.suptitle("Матрица парных графиков числовых признаков", y=1.02)
    g.savefig(os.path.join(OUT_DIR, "03_pairplot.png"), dpi=120, bbox_inches="tight")
    plt.close(g.fig)
    print(f"[сохранён график] {OUT_DIR}/03_pairplot.png")
    print("ВЫВОД: длины сильно коллинеарны между собой и с массой; связь нелинейна.")


def plot_corr_matrix(df: pd.DataFrame) -> None:
    section("ГРАФИК 4. КОРРЕЛЯЦИОННАЯ МАТРИЦА")
    corr = df[NUMERIC_COLS].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, cbar_kws={"label": "Коэффициент Пирсона"})
    plt.title("Корреляционная матрица (Пирсон)")
    plt.xlabel("Признак")
    plt.ylabel("Признак")
    plt.tight_layout()
    savefig("04_corr_matrix.png")

    corr_s = df[NUMERIC_COLS].corr(method="spearman")
    print("Pearson  Weight–Length3:", round(corr.loc["Weight", "Length3"], 3))
    print("Spearman Weight–Length3:", round(corr_s.loc["Weight", "Length3"], 3))
    print("ВЫВОД: Length1–3 дают r > 0.99 — мультиколлинеарность; Пирсон ловит только линейные связи.")


def plot_missing_map(df: pd.DataFrame) -> None:
    section("ГРАФИК 5. КАРТА ПРОПУСКОВ И КАЧЕСТВО ДАННЫХ")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(df.isna(), cbar=False, cmap="viridis", ax=axes[0])
    axes[0].set_title("Карта пропусков (жёлтый = NaN)")
    axes[0].set_xlabel("Признак")
    axes[0].set_ylabel("Индекс строки")

    df.isna().sum().plot(kind="bar", ax=axes[1], color="indianred")
    axes[1].set_title("Число пропусков по признакам")
    axes[1].set_xlabel("Признак")
    axes[1].set_ylabel("Пропуски, шт.")
    plt.tight_layout()
    savefig("05_missing_map.png")
    print("ВЫВОД: пропусков нет, дубликатов нет, отрицательных значений нет.")


def plot_outliers(df: pd.DataFrame) -> None:
    section("ГРАФИК 6. ВЫБРОСЫ С УКАЗАНИЕМ КОНКРЕТНЫХ СТРОК")

    def outliers_iqr(s):
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return (s < lo) | (s > hi), (lo, hi)

    def outliers_mad(s, threshold=3.5):
        med = s.median()
        mad = (s - med).abs().median()
        z = 0.6745 * (s - med) / mad
        return z.abs() > threshold, z

    mask_iqr, (lo_w, hi_w) = outliers_iqr(df[TARGET])
    mask_mad, z_mad = outliers_mad(df[TARGET])

    print(f"IQR-границы Weight: [{lo_w:.1f}, {hi_w:.1f}]")
    print(f"Выбросов по IQR: {mask_iqr.sum()}")
    print(f"Выбросов по MAD: {mask_mad.sum()}")

    report = pd.DataFrame({
        "index": df.index,
        "Weight": df[TARGET].values,
        "IQR_out": mask_iqr.values,
        "MAD_z": z_mad.round(2).values,
        "MAD_out": mask_mad.values,
    })
    print("\nПодозрительные строки:")
    print(report[report["IQR_out"] | report["MAD_out"]])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(x=df[TARGET], ax=axes[0], color="lightblue")
    axes[0].set_title("Выбросы по массе (IQR): отмечены конкретные строки")
    axes[0].set_xlabel("Масса, г")
    axes[0].set_ylabel("")
    for _, row in report[report["IQR_out"]].iterrows():
        axes[0].annotate(f"idx={int(row['index'])}\n({row['Weight']:.0f} г)",
                         xy=(row["Weight"], 0), xytext=(row["Weight"], 0.25),
                         ha="center", fontsize=8, color="crimson",
                         arrowprops=dict(arrowstyle="->", color="crimson"))

    axes[1].scatter(df.index, df[TARGET], c="steelblue", alpha=0.7, s=30)
    axes[1].scatter(report.loc[mask_iqr.values, "index"],
                    report.loc[mask_iqr.values, "Weight"],
                    c="crimson", s=60, label="Выброс (IQR)")
    axes[1].axhline(lo_w, color="gray", ls="--", label="Границы IQR")
    axes[1].axhline(hi_w, color="gray", ls="--")
    axes[1].set_title("Масса по индексу строки с выделением выбросов")
    axes[1].set_xlabel("Индекс строки")
    axes[1].set_ylabel("Масса, г")
    axes[1].legend()
    plt.tight_layout()
    savefig("06_outliers.png")
    print("ВЫВОД: выброс ≠ ошибка; крупные экземпляры оставляем как ценные наблюдения.")


def plot_hypothesis_volume(df: pd.DataFrame) -> None:
    section("ГРАФИК 7. ГИПОТЕЗА V ≈ L·H·W")
    df = df.copy()
    df["V_approx"] = df["Length3"] * df["Height"] * df["Width"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.scatterplot(data=df, x="V_approx", y="Weight", hue="Species",
                    ax=axes[0], alpha=0.8, s=35)
    axes[0].set_title("Гипотеза: Weight ≈ k · (Length3 · Height · Width)")
    axes[0].set_xlabel("V_approx = Length3 · Height · Width, см³")
    axes[0].set_ylabel("Масса, г")

    for sp, g in df.groupby("Species"):
        axes[1].scatter(np.log(g["V_approx"]), np.log(g["Weight"]),
                        label=sp, alpha=0.8, s=25)
    axes[1].set_title("Лог-лог: линейная связь массы и объёма")
    axes[1].set_xlabel("ln(V_approx)")
    axes[1].set_ylabel("ln(Weight)")
    axes[1].legend(fontsize=8, title="Вид")
    plt.tight_layout()
    savefig("07_hypothesis_volume.png")

    r_log = np.corrcoef(np.log(df["V_approx"]), np.log(df["Weight"]))[0, 1]
    r_sp = stats.spearmanr(df["V_approx"], df["Weight"]).correlation
    print(f"Pearson r (log-log): {r_log:.3f}")
    print(f"Spearman r:          {r_sp:.3f}")
    print("ВЫВОД: масса ∝ V^α, α ≈ 1 — гипотеза подтверждена; признак не использует Weight.")


# ============================================================
# ШАГ 3.5. IQR и MAD — сводная таблица
# ============================================================

def iqr_mad_table(df: pd.DataFrame, cols) -> pd.DataFrame:
    section("ШАГ 3.5. IQR И MAD — СВОДНАЯ ТАБЛИЦА")
    rows = []
    for col in cols:
        s = df[col]
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        med = s.median()
        mad = (s - med).abs().median()
        z = 0.6745 * (s - med) / mad
        rows.append({
            "Признак": col,
            "Q1": round(q1, 2),
            "Q3": round(q3, 2),
            "IQR": round(iqr, 2),
            "Нижняя граница": round(lo, 2),
            "Верхняя граница": round(hi, 2),
            "Выбросов IQR, шт.": int(((s < lo) | (s > hi)).sum()),
            "MAD": round(mad, 3),
            "Выбросов MAD (|z|>3.5), шт.": int((z.abs() > 3.5).sum()),
        })
    return pd.DataFrame(rows)


# ============================================================
# ШАГ 3.6. Гипотезы о признаках
# ============================================================

def feature_hypotheses(df: pd.DataFrame) -> None:
    section("ШАГ 3.6. ГИПОТЕЗЫ О ПРИЗНАКАХ")
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    enc.fit(df[["Species"]])
    print("Категории, известные кодировщику:", list(enc.categories_[0]))
    print("Пример вектора:", enc.transform(df[["Species"]].head(1)).ravel())
    print("\n1) Физическая гипотеза V ≈ L·H·W подтверждена (см. график 7).")
    print("2) Species → One-Hot (нет порядка); handle_unknown='ignore' для новых видов.")
    print("3) Length1–3 мультиколлинеарны (r > 0.99); удалять по одному порогу — только с обоснованием.")


# ============================================================
# ШАГ 3.7. Разбиение и таблица каналов утечки
# ============================================================

def split_and_leakage_table(df: pd.DataFrame) -> None:
    section("ШАГ 3.7. РАЗБИЕНИЕ И КАНАЛЫ УТЕЧКИ")
    train, temp = train_test_split(df, test_size=0.4, stratify=df["Species"],
                                   random_state=RANDOM_STATE)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp["Species"],
                                 random_state=RANDOM_STATE)
    print(f"Train: {len(train)} ({len(train)/len(df):.0%})")
    print(f"Val:   {len(val)}   ({len(val)/len(df):.0%})")
    print(f"Test:  {len(test)}  ({len(test)/len(df):.0%})")

    leakage = pd.DataFrame([
        ("Статистики всей таблицы", "mean/std до split → test виден в train",
         "fit только на train, transform на val/test"),
        ("Повторные строки", "одинаковые записи в train и test",
         "drop_duplicates до split"),
        ("Идентификатор (Id/индекс)", "модель запоминает строку по номеру",
         "удалять ID перед обучением"),
        ("Постфактум-измерение", "признак известен только после события",
         "проверять причинно-временную доступность"),
        ("Ручной подбор по test", "итеративная настройка по test",
         "подбор на val, test — один раз в конце"),
    ], columns=["Канал утечки", "Как проявляется", "Как предотвратить"])
    print(leakage.to_string(index=False))


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    df = load_data(DATA_FILE)
    describe_data(df)
    run_quality_checks(df)

    plot_target_distribution(df)
    plot_species_count_and_weight(df)
    plot_pairplot(df)
    plot_corr_matrix(df)
    plot_missing_map(df)
    plot_outliers(df)
    plot_hypothesis_volume(df)

    print("\n", iqr_mad_table(df, NUMERIC_COLS))
    feature_hypotheses(df)
    split_and_leakage_table(df)

    section("ГОТОВО")
    print(f"Все графики сохранены в папку: {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()