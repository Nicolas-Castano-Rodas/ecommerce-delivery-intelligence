from pathlib import Path
import json

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
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

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "raw" / "Train.csv"
OUTPUT_DIR = ROOT / "data" / "processed"

TARGET = "Reached.on.Time_Y.N"
RANDOM_STATE = 2026

# Umbral elegido exclusivamente con los resultados de validación.
SELECTED_THRESHOLD = 0.50

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CARGA Y VALIDACIÓN DE DATOS
# ============================================================

print("=" * 72)
print("EVALUACIÓN FINAL DEL UMBRAL - ECOMMERCE DELIVERY INTELLIGENCE")
print("=" * 72)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"No se encontró el dataset: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

if TARGET not in df.columns:
    raise ValueError(
        f"No se encontró la columna objetivo: {TARGET}"
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

print(f"\nTotal de registros: {len(df):,}")


# ============================================================
# REPRODUCIR LA DIVISIÓN 60/20/20 DEL SCRIPT 06
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

print("\nDistribución de datos:")
print(f"Entrenamiento: {len(X_train):,}")
print(f"Validación:    {len(X_val):,}")
print(f"Prueba final:  {len(X_test):,}")


# ============================================================
# PREPROCESAMIENTO
# ============================================================

numeric_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    exclude=["number"]
).columns.tolist()

transformers = []

if numeric_features:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    transformers.append(
        ("numeric", numeric_pipeline, numeric_features)
    )

if categorical_features:
    categorical_pipeline = Pipeline(
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
    )

    transformers.append(
        ("categorical", categorical_pipeline, categorical_features)
    )

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop",
)


# ============================================================
# ENTRENAMIENTO DEL MISMO MODELO
# ============================================================

print("\nEntrenando Extra Trees con la configuración anterior...")

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
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
)

model.fit(X_train, y_train)


# ============================================================
# VERIFICACIÓN DEL UMBRAL CON VALIDACIÓN
# ============================================================

# El umbral se fija en 0.50 según el análisis previo.
# No se selecciona utilizando los resultados de prueba.

val_probabilities = model.predict_proba(X_val)[:, 1]
val_predictions = (
    val_probabilities >= SELECTED_THRESHOLD
).astype(int)

val_metrics = {
    "precision": float(
        precision_score(y_val, val_predictions, zero_division=0)
    ),
    "recall": float(
        recall_score(y_val, val_predictions, zero_division=0)
    ),
    "f1": float(
        f1_score(y_val, val_predictions, zero_division=0)
    ),
}

print("\nResultados de validación con umbral 0.50:")
for name, value in val_metrics.items():
    print(f"{name.upper():<12}: {value:.4f} ({value:.2%})")


# ============================================================
# EVALUACIÓN FINAL SOBRE PRUEBA
# ============================================================

print("\nEvaluando el conjunto de prueba con umbral fijo 0.50...")

test_probabilities = model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= SELECTED_THRESHOLD
).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1],
).ravel()

test_metrics = {
    "accuracy": float(
        accuracy_score(y_test, test_predictions)
    ),
    "precision": float(
        precision_score(
            y_test,
            test_predictions,
            zero_division=0,
        )
    ),
    "recall": float(
        recall_score(
            y_test,
            test_predictions,
            zero_division=0,
        )
    ),
    "f1": float(
        f1_score(
            y_test,
            test_predictions,
            zero_division=0,
        )
    ),
    "roc_auc": float(
        roc_auc_score(y_test, test_probabilities)
    ),
}

print("\n" + "=" * 72)
print("RESULTADOS DEL CONJUNTO DE PRUEBA")
print("=" * 72)

print(f"Umbral utilizado: {SELECTED_THRESHOLD:.2f}")

for name, value in test_metrics.items():
    print(f"{name.upper():<12}: {value:.4f} ({value:.2%})")

print("\nMatriz de confusión:")
print(
    confusion_matrix(
        y_test,
        test_predictions,
        labels=[0, 1],
    )
)

print("\nDetalle por clase:")
print(
    classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=["A tiempo", "Retrasado"],
        zero_division=0,
    )
)

print("\nResumen de predicciones:")
print(f"Verdaderos negativos: {tn}")
print(f"Falsos positivos:     {fp}")
print(f"Falsos negativos:     {fn}")
print(f"Verdaderos positivos: {tp}")

print(f"\nEntregas marcadas como tardías: {fp + tp}")
print(f"Entregas marcadas como a tiempo: {tn + fn}")


# ============================================================
# GUARDAR RESULTADOS
# ============================================================

json_path = OUTPUT_DIR / "evaluacion_umbral_050.json"

summary = {
    "modelo": "ExtraTreesClassifier",
    "umbral_evaluado": SELECTED_THRESHOLD,
    "random_state_division": RANDOM_STATE,
    "distribucion": {
        "entrenamiento": len(X_train),
        "validacion": len(X_val),
        "prueba": len(X_test),
    },
    "metricas_validacion": val_metrics,
    "metricas_prueba": test_metrics,
    "matriz_confusion": {
        "verdaderos_negativos": int(tn),
        "falsos_positivos": int(fp),
        "falsos_negativos": int(fn),
        "verdaderos_positivos": int(tp),
    },
    "advertencia": (
        "El conjunto de datos ya se había explorado en experimentos "
        "anteriores. Esta evaluación no debe considerarse completamente "
        "independiente de todas las decisiones previas del proyecto."
    ),
}

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(
        summary,
        file,
        indent=4,
        ensure_ascii=False,
    )

print(f"\nResultados guardados en: {json_path.relative_to(ROOT)}")
print("\nEvaluación terminada.")
print("El modelo actual del backend no ha sido modificado.")