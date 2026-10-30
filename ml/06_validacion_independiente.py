from pathlib import Path
import json

import joblib
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
ARTIFACTS_DIR = ROOT / "ml" / "artifacts"

TARGET = "Reached.on.Time_Y.N"
RANDOM_STATE = 2026

# Umbrales candidatos que se seleccionarán usando validación.
THRESHOLDS = [
    0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70,
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CARGA DE DATOS
# ============================================================

print("=" * 72)
print("VALIDACIÓN DEL MODELO - ECOMMERCE DELIVERY INTELLIGENCE")
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
# DIVISIÓN 60/20/20
# ============================================================

# 60 % entrenamiento; 40 % temporal.
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=RANDOM_STATE,
    stratify=y,
)

# El 40 % temporal se divide por igual:
# 20 % validación y 20 % prueba final.
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp,
)

print("\nDistribución de datos:")
print(f"Entrenamiento: {len(X_train):,} ({len(X_train) / len(df):.1%})")
print(f"Validación:    {len(X_val):,} ({len(X_val) / len(df):.1%})")
print(f"Prueba final:  {len(X_test):,} ({len(X_test) / len(df):.1%})")


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
# ENTRENAMIENTO
# ============================================================

print("\nEntrenando Extra Trees...")

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

# Las probabilidades de validación se utilizan para elegir
# el umbral. El conjunto de prueba aún no se consulta.
val_probabilities = model.predict_proba(X_val)[:, 1]


# ============================================================
# SELECCIÓN DEL UMBRAL CON VALIDACIÓN
# ============================================================

validation_results = []

print("\nComparación de umbrales sobre validación:")

for threshold in THRESHOLDS:
    val_predictions = (
        val_probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        val_predictions,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        y_val,
        val_predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_val,
        val_predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_val,
        val_predictions,
        zero_division=0,
    )

    validation_results.append(
        {
            "umbral": threshold,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "falsos_positivos": int(fp),
            "falsos_negativos": int(fn),
        }
    )

    print(
        f"Umbral={threshold:.2f} | "
        f"Precision={precision:.4f} | "
        f"Recall={recall:.4f} | "
        f"F1={f1:.4f}"
    )

# Elegimos el umbral con mejor F1 en validación.
# En caso de empate, preferimos el umbral más cercano a 0.40.
best_validation = max(
    validation_results,
    key=lambda row: (
        row["f1"],
        -abs(row["umbral"] - 0.40),
    ),
)

selected_threshold = best_validation["umbral"]

print("\nUmbral seleccionado usando validación:")
print(f"Umbral: {selected_threshold:.2f}")
print(f"F1 de validación: {best_validation['f1']:.4f}")
print(f"Precision: {best_validation['precision']:.4f}")
print(f"Recall: {best_validation['recall']:.4f}")


# ============================================================
# EVALUACIÓN FINAL SOBRE EL CONJUNTO DE PRUEBA
# ============================================================

print("\nEvaluando el conjunto de prueba...")

# Solo ahora obtenemos las probabilidades de prueba.
test_probabilities = model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= selected_threshold
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
print("RESULTADOS FINALES")
print("=" * 72)

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

print("\nResumen de errores:")
print(f"Verdaderos negativos: {tn}")
print(f"Falsos positivos:     {fp}")
print(f"Falsos negativos:     {fn}")
print(f"Verdaderos positivos: {tp}")


# ============================================================
# GUARDAR ARTEFACTOS Y RESULTADOS
# ============================================================

model_path = (
    ARTIFACTS_DIR
    / "extra_trees_validated_candidate.joblib"
)

csv_path = (
    OUTPUT_DIR
    / "validacion_umbrales_60_20_20.csv"
)

json_path = (
    OUTPUT_DIR
    / "evaluacion_60_20_20.json"
)

pd.DataFrame(validation_results).to_csv(
    csv_path,
    index=False,
)

joblib.dump(model, model_path)

summary = {
    "modelo": "ExtraTreesClassifier",
    "random_state_division": RANDOM_STATE,
    "distribucion": {
        "entrenamiento": len(X_train),
        "validacion": len(X_val),
        "prueba": len(X_test),
    },
    "umbral_seleccionado": selected_threshold,
    "metricas_validacion": best_validation,
    "metricas_prueba": test_metrics,
    "matriz_confusion": {
        "verdaderos_negativos": int(tn),
        "falsos_positivos": int(fp),
        "falsos_negativos": int(fn),
        "verdaderos_positivos": int(tp),
    },
    "modelo_guardado": str(model_path.relative_to(ROOT)),
    "advertencia": (
        "La división utiliza un nuevo random_state, pero el dataset "
        "ya fue explorado en experimentos anteriores. Por ello, esta "
        "prueba no debe considerarse completamente independiente "
        "de todas las decisiones previas del proyecto."
    ),
}

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(
        summary,
        file,
        indent=4,
        ensure_ascii=False,
    )

print(f"\nModelo candidato: {model_path.relative_to(ROOT)}")
print(f"Tabla de validación: {csv_path.relative_to(ROOT)}")
print(f"Resumen: {json_path.relative_to(ROOT)}")

print("\nProceso terminado.")
print("El modelo actual del backend no ha sido modificado.")