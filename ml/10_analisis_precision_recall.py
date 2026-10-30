from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# CONFIGURACIÓN
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "Train.csv"
OUTPUT_DIR = ROOT / "data" / "processed"

TARGET = "Reached.on.Time_Y.N"
RANDOM_STATE = 2026
MIN_PRECISION = 0.80

THRESHOLDS = [
    round(float(value), 2)
    for value in __import__("numpy").arange(0.20, 0.81, 0.05)
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. CARGAR DATOS
# ============================================================

print("=" * 72)
print("ANÁLISIS PRECISION-RECALL")
print("ECOMMERCE DELIVERY INTELLIGENCE")
print("=" * 72)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"No existe el dataset: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

if TARGET not in df.columns:
    raise ValueError(f"No existe la columna objetivo: {TARGET}")

if df[TARGET].isna().any():
    raise ValueError("La variable objetivo contiene valores ausentes.")

X = df.drop(columns=[TARGET, "ID"], errors="ignore")
y = df[TARGET].astype(int)


# ============================================================
# 2. DIVISIÓN 60/20/20
# ============================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=RANDOM_STATE,
    stratify=y,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp,
)

print(f"\nEntrenamiento: {len(X_train):,}")
print(f"Validación:    {len(X_val):,}")
print(f"Prueba reservada: {len(X_test):,}")


# ============================================================
# 3. PREPROCESAMIENTO
# ============================================================

numeric_features = X_train.select_dtypes(
    include="number"
).columns.tolist()

categorical_features = X_train.select_dtypes(
    exclude="number"
).columns.tolist()

transformers = []

if numeric_features:
    transformers.append(
        (
            "numeric",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median"))
                ]
            ),
            numeric_features,
        )
    )

if categorical_features:
    transformers.append(
        (
            "categorical",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent"),
                    ),
                    (
                        "encoder",
                        OneHotEncoder(handle_unknown="ignore"),
                    ),
                ]
            ),
            categorical_features,
        )
    )


def create_preprocessor():
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


# ============================================================
# 4. MODELOS
# ============================================================

models = {
    "RandomForestClassifier": Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    ),
    "ExtraTreesClassifier": Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "classifier",
                ExtraTreesClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    class_weight=None,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    ),
}


# ============================================================
# 5. ANALIZAR SOLO VALIDACIÓN
# ============================================================

all_results = []
curve_results = {}
recommendations = {}

for model_name, model in models.items():
    print(f"\nEntrenando {model_name}...")

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_val)[:, 1]

    average_precision = average_precision_score(
        y_val,
        probabilities,
    )

    precision_curve, recall_curve, curve_thresholds = (
        precision_recall_curve(y_val, probabilities)
    )

    curve_results[model_name] = {
        "precision": precision_curve.tolist(),
        "recall": recall_curve.tolist(),
        "thresholds": curve_thresholds.tolist(),
        "average_precision": float(average_precision),
    }

    eligible_results = []

    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)

        precision = precision_score(
            y_val,
            predictions,
            zero_division=0,
        )
        recall = recall_score(
            y_val,
            predictions,
            zero_division=0,
        )
        f1 = f1_score(
            y_val,
            predictions,
            zero_division=0,
        )

        true_positive = int(
            ((predictions == 1) & (y_val.to_numpy() == 1)).sum()
        )
        false_positive = int(
            ((predictions == 1) & (y_val.to_numpy() == 0)).sum()
        )
        false_negative = int(
            ((predictions == 0) & (y_val.to_numpy() == 1)).sum()
        )

        row = {
            "modelo": model_name,
            "umbral": threshold,
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "average_precision": float(average_precision),
            "verdaderos_positivos": true_positive,
            "falsos_positivos": false_positive,
            "falsos_negativos": false_negative,
            "total_alertas": int(predictions.sum()),
        }

        all_results.append(row)

        if precision >= MIN_PRECISION and predictions.sum() > 0:
            eligible_results.append(row)

    if eligible_results:
        # Entre los umbrales que cumplen la precisión mínima,
        # priorizar el recall y después el F1.
        best = max(
            eligible_results,
            key=lambda item: (
                item["recall"],
                item["f1_score"],
            ),
        )
        recommendations[model_name] = {
            "cumple_precision_minima": True,
            "criterio": (
                f"Precisión >= {MIN_PRECISION:.0%}; "
                "maximizar recall y después F1."
            ),
            "resultado": best,
        }
    else:
        recommendations[model_name] = {
            "cumple_precision_minima": False,
            "criterio": (
                f"No se encontró un umbral evaluado con precisión "
                f">= {MIN_PRECISION:.0%} y al menos una alerta."
            ),
            "resultado": None,
        }


# ============================================================
# 6. GUARDAR TABLAS Y GRÁFICO
# ============================================================

results_df = pd.DataFrame(all_results)

csv_path = OUTPUT_DIR / "analisis_precision_recall.csv"
json_path = OUTPUT_DIR / "analisis_precision_recall.json"
plot_path = OUTPUT_DIR / "curvas_precision_recall.png"

results_df.to_csv(
    csv_path,
    index=False,
    encoding="utf-8-sig",
)

fig, ax = plt.subplots(figsize=(9, 6))

for model_name, curve in curve_results.items():
    ax.plot(
        curve["recall"],
        curve["precision"],
        label=(
            f"{model_name} "
            f"(AP={curve['average_precision']:.3f})"
        ),
    )

ax.axhline(
    MIN_PRECISION,
    linestyle="--",
    label=f"Precisión objetivo: {MIN_PRECISION:.0%}",
)

ax.set_xlabel("Recall (retrasos detectados)")
ax.set_ylabel("Precision (alertas correctas)")
ax.set_title("Curvas Precision-Recall sobre validación")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(plot_path, dpi=160, bbox_inches="tight")
plt.close(fig)

summary = {
    "objetivo_precision_minima": MIN_PRECISION,
    "division": {
        "random_state": RANDOM_STATE,
        "entrenamiento": len(X_train),
        "validacion": len(X_val),
        "prueba_reservada": len(X_test),
    },
    "resultados_por_umbral": all_results,
    "recomendaciones_por_modelo": recommendations,
    "curvas_precision_recall": curve_results,
    "advertencia": (
        "Los umbrales se analizan únicamente con validación. "
        "El conjunto de prueba no se utiliza para elegirlos. "
        "El dataset ya fue explorado durante experimentos previos, "
        "por lo que no constituye una validación totalmente "
        "independiente de todas las decisiones del proyecto."
    ),
}

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=4, ensure_ascii=False)

print("\n" + "=" * 72)
print("RESULTADOS DE VALIDACIÓN")
print("=" * 72)

for model_name, recommendation in recommendations.items():
    print(f"\nModelo: {model_name}")

    if recommendation["cumple_precision_minima"]:
        result = recommendation["resultado"]
        print(f"Umbral candidato: {result['umbral']:.2f}")
        print(f"Precisión: {result['precision']:.2%}")
        print(f"Recall: {result['recall']:.2%}")
        print(f"F1-score: {result['f1_score']:.4f}")
        print(f"Alertas: {result['total_alertas']}")
        print(f"Falsas alarmas: {result['falsos_positivos']}")
        print(f"Retrasos no detectados: {result['falsos_negativos']}")
    else:
        print(recommendation["criterio"])

print("\nArchivos generados:")
print(f"- {csv_path.relative_to(ROOT)}")
print(f"- {json_path.relative_to(ROOT)}")
print(f"- {plot_path.relative_to(ROOT)}")

print("\nAnálisis terminado. No se modificó el backend.")