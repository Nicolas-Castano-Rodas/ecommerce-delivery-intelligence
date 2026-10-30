from pathlib import Path
import json

import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
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
THRESHOLD = 0.50

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. CARGA Y VALIDACIÓN
# ============================================================

print("=" * 72)
print("ECOMMERCE DELIVERY INTELLIGENCE")
print("COMPARACIÓN CONTROLADA: RANDOM FOREST VS EXTRA TREES")
print("=" * 72)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"No se encontró el dataset: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

if TARGET not in df.columns:
    raise ValueError(
        f"No se encontró la variable objetivo: {TARGET}"
    )

if df[TARGET].isna().any():
    raise ValueError(
        "La variable objetivo contiene valores ausentes."
    )

if not set(df[TARGET].unique()).issubset({0, 1}):
    raise ValueError(
        "La variable objetivo debe contener únicamente 0 y 1."
    )

X = df.drop(columns=[TARGET, "ID"], errors="ignore")
y = df[TARGET].astype(int)

print(f"\nRegistros totales: {len(df):,}")
print(f"Variables predictivas: {X.shape[1]}")


# ============================================================
# 2. DIVISIÓN ESTRATIFICADA 60/20/20
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

print("\nDivisión de datos:")
print(f"Entrenamiento: {len(X_train):,}")
print(f"Validación:    {len(X_val):,}")
print(f"Prueba:        {len(X_test):,}")

# La validación se reserva para decisiones posteriores.
# En esta comparación, ambos modelos se evalúan en la misma
# prueba final y utilizan un umbral fijo de 0.50.


# ============================================================
# 3. PREPROCESAMIENTO COMPARTIDO
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
                    (
                        "imputer",
                        SimpleImputer(strategy="median"),
                    )
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
                        SimpleImputer(
                            strategy="most_frequent"
                        ),
                    ),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        ),
                    ),
                ]
            ),
            categorical_features,
        )
    )


def create_preprocessor():
    """Crea el mismo preprocesamiento para cada modelo."""
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


# ============================================================
# 4. CONFIGURACIÓN DE LOS MODELOS
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
# 5. ENTRENAMIENTO Y EVALUACIÓN COMPARABLE
# ============================================================

results = []
confusion_matrices = {}

for model_name, model in models.items():
    print("\n" + "-" * 72)
    print(f"Entrenando: {model_name}")
    print("-" * 72)

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "modelo": model_name,
        "umbral": THRESHOLD,
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
        "verdaderos_negativos": int(tn),
        "falsos_positivos": int(fp),
        "falsos_negativos": int(fn),
        "verdaderos_positivos": int(tp),
        "alertas_retraso": int(predictions.sum()),
    }

    results.append(metrics)

    confusion_matrices[model_name] = {
        "filas": ["Real: a tiempo", "Real: retrasado"],
        "columnas": [
            "Predicción: a tiempo",
            "Predicción: retrasado",
        ],
        "valores": [
            [int(tn), int(fp)],
            [int(fn), int(tp)],
        ],
    }

    print(f"Accuracy:   {metrics['accuracy']:.4f}")
    print(f"Precision:  {metrics['precision']:.4f}")
    print(f"Recall:     {metrics['recall']:.4f}")
    print(f"F1-score:   {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:    {metrics['roc_auc']:.4f}")
    print(f"Falsos positivos: {fp}")
    print(f"Falsos negativos: {fn}")


# ============================================================
# 6. COMPARACIÓN Y GUARDADO
# ============================================================

results_df = pd.DataFrame(results)

csv_path = OUTPUT_DIR / "comparacion_modelos_justa.csv"
json_path = OUTPUT_DIR / "comparacion_modelos_justa.json"

results_df.to_csv(
    csv_path,
    index=False,
    encoding="utf-8-sig",
)

summary = {
    "experimento": "Comparación controlada de clasificadores",
    "dataset": str(DATA_PATH.relative_to(ROOT)),
    "registros_totales": int(len(df)),
    "variables_predictivas": int(X.shape[1]),
    "random_state_division": RANDOM_STATE,
    "distribucion": {
        "entrenamiento": int(len(X_train)),
        "validacion": int(len(X_val)),
        "prueba": int(len(X_test)),
    },
    "umbral_comun": THRESHOLD,
    "configuracion": {
        "preprocesamiento": (
            "Imputación de medianas y modas, "
            "OneHotEncoder y exclusión de ID."
        ),
        "random_forest": {
            "n_estimators": 300,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "random_state": 42,
        },
        "extra_trees": {
            "n_estimators": 300,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "class_weight": None,
            "random_state": 42,
        },
    },
    "matrices_confusion": confusion_matrices,
    "resultados": results,
    "advertencia": (
        "El dataset ya se utilizó en experimentos anteriores. "
        "Esta comparación usa la misma partición de prueba para "
        "ambos modelos, pero no constituye una validación "
        "completamente independiente de todas las decisiones "
        "previas del proyecto. El preprocesamiento compartido "
        "también implica que los resultados pueden diferir "
        "de los modelos originales guardados."
    ),
}

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(
        summary,
        file,
        indent=4,
        ensure_ascii=False,
    )

print("\n" + "=" * 72)
print("RESUMEN COMPARATIVO")
print("=" * 72)

print(
    results_df[
        [
            "modelo",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "falsos_positivos",
            "falsos_negativos",
        ]
    ].to_string(index=False)
)

print("\nArchivos generados:")
print(f"- {csv_path.relative_to(ROOT)}")
print(f"- {json_path.relative_to(ROOT)}")

print("\nComparación terminada.")
print("No se modificaron los modelos usados por el backend.")