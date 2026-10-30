from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURACIÓN
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "raw" / "Train.csv"
MODEL_PATH = ROOT / "ml" / "artifacts" / "extra_trees_candidate.joblib"
OUTPUT_DIR = ROOT / "data" / "processed"

TARGET = "Reached.on.Time_Y.N"
RANDOM_STATE = 42

# Umbrales que vamos a comparar.
THRESHOLDS = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CARGA DE DATOS
# ============================================================

print("=" * 72)
print("ANÁLISIS DE UMBRALES DE PREDICCIÓN")
print("ECOMMERCE DELIVERY INTELLIGENCE")
print("=" * 72)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"No se encontró el dataset: {DATA_PATH}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH}")

df = pd.read_csv(DATA_PATH)

if TARGET not in df.columns:
    raise ValueError(f"No se encontró la columna objetivo: {TARGET}")

if df[TARGET].isna().any():
    raise ValueError("La variable objetivo contiene valores ausentes.")

X = df.drop(columns=[TARGET, "ID"], errors="ignore")
y = df[TARGET].astype(int)


# ============================================================
# RECREAR LA PARTICIÓN ORIGINAL
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

# Importante: el modelo ya fue entrenado con X_train.
# En este experimento utilizamos las probabilidades de X_test
# para estudiar el comportamiento de distintos umbrales.
model = joblib.load(MODEL_PATH)

probabilities = model.predict_proba(X_test)[:, 1]


# ============================================================
# COMPARAR UMBRALES
# ============================================================

results = []

print(f"\nPedidos evaluados: {len(X_test):,}")
print(f"Umbrales analizados: {len(THRESHOLDS)}")
print("\nResultados por umbral:\n")

for threshold in THRESHOLDS:
    # La clase 1 representa un pedido retrasado.
    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

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

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    # El ROC-AUC no depende del umbral.
    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    results.append(
        {
            "umbral": threshold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "verdaderos_negativos": int(tn),
            "falsos_positivos": int(fp),
            "falsos_negativos": int(fn),
            "verdaderos_positivos": int(tp),
            "alertas_generadas": int(tp + fp),
        }
    )

    print(
        f"Umbral={threshold:.2f} | "
        f"Precision={precision:.4f} | "
        f"Recall={recall:.4f} | "
        f"F1={f1:.4f} | "
        f"Accuracy={accuracy:.4f} | "
        f"Alertas={tp + fp}"
    )


# ============================================================
# GUARDAR Y MOSTRAR RESULTADOS
# ============================================================

results_df = pd.DataFrame(results)

csv_path = OUTPUT_DIR / "analisis_umbrales.csv"
json_path = OUTPUT_DIR / "analisis_umbrales.json"

results_df.to_csv(csv_path, index=False)

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(
        results,
        file,
        indent=4,
        ensure_ascii=False,
    )

print("\n" + "=" * 72)
print("TABLA COMPARATIVA")
print("=" * 72)

print(
    results_df[
        [
            "umbral",
            "precision",
            "recall",
            "f1",
            "accuracy",
            "falsos_positivos",
            "falsos_negativos",
            "alertas_generadas",
        ]
    ].to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

# Referencia descriptiva: mejor F1 entre los umbrales probados.
best_f1_row = results_df.loc[results_df["f1"].idxmax()]

print("\nUmbral con mayor F1 en este análisis:")
print(f"Umbral: {best_f1_row['umbral']:.2f}")
print(f"Precision: {best_f1_row['precision']:.4f}")
print(f"Recall: {best_f1_row['recall']:.4f}")
print(f"F1: {best_f1_row['f1']:.4f}")
print(
    "Falsas alarmas: "
    f"{int(best_f1_row['falsos_positivos'])}"
)
print(
    "Retrasos no detectados: "
    f"{int(best_f1_row['falsos_negativos'])}"
)

print(f"\nCSV guardado: {csv_path.relative_to(ROOT)}")
print(f"JSON guardado: {json_path.relative_to(ROOT)}")

print("\nAnálisis finalizado.")
print(
    "Nota: estos resultados sirven para explorar umbrales; "
    "no constituyen una nueva evaluación independiente."
)