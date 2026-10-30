from pathlib import Path
import json

import joblib
import pandas as pd

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


# ============================================================
# CONFIGURACIÓN
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "raw" / "Train.csv"
MODEL_PATH = ROOT / "ml" / "artifacts" / "extra_trees_candidate.joblib"
OUTPUT_DIR = ROOT / "data" / "processed"

TARGET = "Reached.on.Time_Y.N"
RANDOM_STATE = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# VALIDACIONES
# ============================================================

print("=" * 70)
print("EVALUACIÓN FINAL DEL MODELO")
print("ECOMMERCE DELIVERY INTELLIGENCE")
print("=" * 70)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"No se encontró el dataset: {DATA_PATH}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH}")

df = pd.read_csv(DATA_PATH)

if TARGET not in df.columns:
    raise ValueError(f"No se encontró la columna {TARGET}")

X = df.drop(columns=[TARGET, "ID"], errors="ignore")
y = df[TARGET].astype(int)


# ============================================================
# RECREAR LA MISMA PARTICIÓN
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"\nRegistros de prueba: {len(X_test):,}")

# El modelo candidato fue entrenado exclusivamente con X_train.
model = joblib.load(MODEL_PATH)

# Primera evaluación sobre el conjunto de prueba reservado.
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# ============================================================
# MÉTRICAS
# ============================================================

metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred, zero_division=0)),
    "recall": float(recall_score(y_test, y_pred, zero_division=0)),
    "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, y_prob)),
}

matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])

tn, fp, fn, tp = matrix.ravel()

print("\nRESULTADOS SOBRE EL CONJUNTO DE PRUEBA")
print("-" * 70)

for metric, value in metrics.items():
    print(f"{metric.upper():<12}: {value:.4f} ({value:.2%})")

print("\nMATRIZ DE CONFUSIÓN")
print("Filas = valor real; columnas = predicción")
print("Orden de clases: 0 = a tiempo, 1 = retrasado")
print()
print(matrix)

print("\nDETALLE DE CLASIFICACIÓN")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["A tiempo", "Retrasado"],
        zero_division=0,
    )
)

print("\nINTERPRETACIÓN DE LA MATRIZ")
print(f"Pedidos a tiempo correctamente identificados: {tn}")
print(f"Pedidos a tiempo marcados como retrasados: {fp}")
print(f"Retrasos que el modelo no detectó: {fn}")
print(f"Retrasos correctamente detectados: {tp}")


# ============================================================
# GUARDAR RESULTADOS
# ============================================================

results = {
    "modelo": "ExtraTreesClassifier",
    "conjunto": "test reservado",
    "registros_prueba": int(len(X_test)),
    "metricas": metrics,
    "matriz_confusion": {
        "verdaderos_negativos": int(tn),
        "falsos_positivos": int(fp),
        "falsos_negativos": int(fn),
        "verdaderos_positivos": int(tp),
    },
    "advertencia": (
        "Resultados calculados sobre una partición reservada. "
        "No se deben ajustar los parámetros repetidamente "
        "usando estos resultados."
    ),
}

output_path = OUTPUT_DIR / "evaluacion_final_extra_trees.json"

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4, ensure_ascii=False)

print(f"\nResultados guardados en: {output_path.relative_to(ROOT)}")
print("\nEvaluación finalizada.")