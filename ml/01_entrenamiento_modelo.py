from pathlib import Path
import json

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# CONFIGURACIÓN
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "raw" / "Train.csv"
MODEL_DIR = ROOT_DIR / "ml" / "artifacts"
RESULTS_DIR = ROOT_DIR / "data" / "processed"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Reached.on.Time_Y.N"
RANDOM_STATE = 42


# ============================================================
# 1. CARGA Y VALIDACIÓN DE LOS DATOS
# ============================================================

print("\n" + "=" * 65)
print("ECOMMERCE DELIVERY INTELLIGENCE")
print("ENTRENAMIENTO DEL MODELO DE MACHINE LEARNING")
print("=" * 65)

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

print(f"\nRegistros disponibles: {len(df):,}")
print(f"Columnas originales: {len(df.columns)}")


# ============================================================
# 2. SEPARACIÓN DE VARIABLES Y OBJETIVO
# ============================================================

# ID es un identificador, no una característica predictiva.
X = df.drop(columns=[TARGET, "ID"], errors="ignore")
y = df[TARGET].astype(int)

numeric_features = X.select_dtypes(
    include="number"
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude="number"
).columns.tolist()

print("\nVariables numéricas:")
print(numeric_features)

print("\nVariables categóricas:")
print(categorical_features)

print("\nDistribución de la variable objetivo:")
print(
    y.value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
    .rename("porcentaje")
    .to_string()
)


# ============================================================
# 3. DIVISIÓN ENTRE ENTRENAMIENTO Y PRUEBA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\nDivisión de datos:")
print(f"Entrenamiento: {len(X_train):,} registros")
print(f"Prueba: {len(X_test):,} registros")


# ============================================================
# 4. MODELO DE REFERENCIA
# ============================================================

# Predice siempre la clase más frecuente.
# Sirve para comprobar si el modelo avanzado mejora
# una estrategia extremadamente sencilla.

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_test)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions,
)

baseline_f1 = f1_score(
    y_test,
    baseline_predictions,
    zero_division=0,
)

print("\n" + "-" * 65)
print("MODELO DE REFERENCIA")
print("-" * 65)
print(f"Accuracy: {baseline_accuracy:.4f}")
print(f"F1-score: {baseline_f1:.4f}")


# ============================================================
# 5. PREPARACIÓN Y ENTRENAMIENTO DEL MODELO
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
        (
            "numeric",
            "passthrough",
            numeric_features,
        ),
    ],
)

random_forest = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", random_forest),
    ]
)

print("\nEntrenando Random Forest...")
model.fit(X_train, y_train)

print("Entrenamiento finalizado.")


# ============================================================
# 6. EVALUACIÓN
# ============================================================

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(
    y_test,
    predictions,
    zero_division=0,
)
recall = recall_score(
    y_test,
    predictions,
    zero_division=0,
)
f1 = f1_score(
    y_test,
    predictions,
    zero_division=0,
)
roc_auc = roc_auc_score(y_test, probabilities)

metrics = {
    "model": "RandomForestClassifier",
    "dataset_rows": int(len(df)),
    "training_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "random_state": RANDOM_STATE,
    "baseline_accuracy": float(baseline_accuracy),
    "baseline_f1": float(baseline_f1),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc),
}

print("\n" + "=" * 65)
print("RESULTADOS DEL MODELO")
print("=" * 65)

for name, value in metrics.items():
    if isinstance(value, float):
        print(f"{name}: {value:.4f}")
    else:
        print(f"{name}: {value}")

print("\nInforme de clasificación:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Llegó a tiempo (0)",
            "No llegó a tiempo (1)",
        ],
        zero_division=0,
    )
)


# ============================================================
# 7. GUARDAR MÉTRICAS E INFORMES
# ============================================================

metrics_path = RESULTS_DIR / "metricas_modelo.json"

with open(metrics_path, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4, ensure_ascii=False)

report = classification_report(
    y_test,
    predictions,
    target_names=[
        "Llegó a tiempo (0)",
        "No llegó a tiempo (1)",
    ],
    output_dict=True,
    zero_division=0,
)

report_df = pd.DataFrame(report).transpose()
report_df.to_csv(
    RESULTS_DIR / "informe_clasificacion.csv",
    encoding="utf-8-sig",
)


# ============================================================
# 8. MATRIZ DE CONFUSIÓN
# ============================================================

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1],
)

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=[
        "Llegó a tiempo",
        "No llegó a tiempo",
    ],
)

fig, ax = plt.subplots(figsize=(8, 6))

display.plot(
    ax=ax,
    values_format="d",
    colorbar=False,
)

ax.set_title("Matriz de confusión - Random Forest")
fig.tight_layout()

confusion_path = RESULTS_DIR / "matriz_confusion.png"

fig.savefig(
    confusion_path,
    dpi=150,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 9. IMPORTANCIA DE LAS VARIABLES
# ============================================================

fitted_preprocessor = model.named_steps["preprocessor"]
fitted_classifier = model.named_steps["classifier"]

feature_names = (
    fitted_preprocessor.get_feature_names_out()
)

importances = pd.DataFrame(
    {
        "variable": feature_names,
        "importancia": fitted_classifier.feature_importances_,
    }
).sort_values(
    "importancia",
    ascending=False,
)

importance_path = RESULTS_DIR / "importancia_variables.csv"

importances.to_csv(
    importance_path,
    index=False,
    encoding="utf-8-sig",
)

top_features = importances.head(15).sort_values(
    "importancia",
    ascending=True,
)

fig, ax = plt.subplots(figsize=(10, 7))

ax.barh(
    top_features["variable"],
    top_features["importancia"],
)

ax.set_title("15 variables más importantes para el modelo")
ax.set_xlabel("Importancia relativa")
ax.set_ylabel("Variable")

fig.tight_layout()

importance_plot_path = (
    RESULTS_DIR / "importancia_variables.png"
)

fig.savefig(
    importance_plot_path,
    dpi=150,
    bbox_inches="tight",
)

plt.close(fig)

print("\n15 variables más importantes:")
print(importances.head(15).to_string(index=False))


# ============================================================
# 10. GUARDAR EL MODELO ENTRENADO
# ============================================================

model_path = MODEL_DIR / "delivery_delay_model.joblib"

joblib.dump(model, model_path)

print("\n" + "=" * 65)
print("ARCHIVOS GENERADOS")
print("=" * 65)

for path in [
    model_path,
    metrics_path,
    RESULTS_DIR / "informe_clasificacion.csv",
    confusion_path,
    importance_path,
    importance_plot_path,
]:
    print(f"- {path.relative_to(ROOT_DIR)}")

print("\nModelo entrenado y guardado correctamente.")