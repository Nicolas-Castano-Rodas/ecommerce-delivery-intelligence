from pathlib import Path
import json
import time

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# CONFIGURACIÓN
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "raw" / "Train.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACTS_DIR = ROOT / "ml" / "artifacts"

TARGET = "Reached.on.Time_Y.N"
RANDOM_STATE = 42

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CARGA Y VALIDACIÓN DE DATOS
# ============================================================

print("=" * 72)
print("OPTIMIZACIÓN DE EXTRA TREES")
print("ECOMMERCE DELIVERY INTELLIGENCE")
print("=" * 72)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"No se encontró el dataset: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

if TARGET not in df.columns:
    raise ValueError(
        f"No existe la columna objetivo: {TARGET}"
    )

if df[TARGET].isna().any():
    raise ValueError(
        "La columna objetivo contiene valores ausentes."
    )

if not set(df[TARGET].unique()).issubset({0, 1}):
    raise ValueError(
        "La variable objetivo debe contener únicamente 0 y 1."
    )

X = df.drop(columns=[TARGET, "ID"], errors="ignore")
y = df[TARGET].astype(int)

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

print(f"\nRegistros: {len(df):,}")
print(f"Características numéricas: {len(numeric_features)}")
print(f"Características categóricas: {len(categorical_features)}")
print(f"Proporción de retrasos: {y.mean():.2%}")


# ============================================================
# SEPARAR ENTRENAMIENTO Y PRUEBA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"\nDatos de entrenamiento: {len(X_train):,}")
print(f"Datos reservados para prueba: {len(X_test):,}")


# ============================================================
# PREPROCESAMIENTO
# ============================================================

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
# CONFIGURACIONES EXPERIMENTALES
# ============================================================

configurations = [
    {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced",
    },
    {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 2,
        "max_features": 0.8,
        "class_weight": "balanced",
    },
    {
        "n_estimators": 300,
        "max_depth": 20,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced",
    },
    {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 4,
        "max_features": "sqrt",
        "class_weight": "balanced",
    },
    {
        "n_estimators": 400,
        "max_depth": None,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
    },
    {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": None,
    },
]

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

results = []

best_f1 = -1.0
best_pipeline = None
best_configuration = None

print("\nIniciando validación cruzada de 5 particiones...")
print("El conjunto de prueba permanecerá sin utilizar.\n")


# ============================================================
# EVALUACIÓN
# ============================================================

for index, params in enumerate(configurations, start=1):
    start = time.time()

    model = ExtraTreesClassifier(
        **params,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        error_score="raise",
    )

    row = {
        "configuracion": index,
        **params,
        "accuracy_media": scores["test_accuracy"].mean(),
        "precision_media": scores["test_precision"].mean(),
        "recall_media": scores["test_recall"].mean(),
        "f1_media": scores["test_f1"].mean(),
        "roc_auc_media": scores["test_roc_auc"].mean(),
        "f1_desviacion": scores["test_f1"].std(),
        "tiempo_segundos": round(time.time() - start, 2),
    }

    results.append(row)

    print(
        f"Configuración {index}/{len(configurations)} | "
        f"F1={row['f1_media']:.4f} | "
        f"Recall={row['recall_media']:.4f} | "
        f"Precision={row['precision_media']:.4f} | "
        f"ROC-AUC={row['roc_auc_media']:.4f} | "
        f"{row['tiempo_segundos']:.1f} s"
    )

    if row["f1_media"] > best_f1:
        best_f1 = row["f1_media"]
        best_configuration = params.copy()

        # Entrenar el candidato con todos los datos de entrenamiento.
        best_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    ExtraTreesClassifier(
                        **best_configuration,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        best_pipeline.fit(X_train, y_train)


# ============================================================
# GUARDAR RESULTADOS
# ============================================================

results_df = pd.DataFrame(results).sort_values(
    by="f1_media",
    ascending=False,
)

csv_path = PROCESSED_DIR / "optimizacion_extra_trees.csv"
json_path = PROCESSED_DIR / "optimizacion_extra_trees.json"
candidate_path = ARTIFACTS_DIR / "extra_trees_candidate.joblib"

results_df.to_csv(csv_path, index=False)

summary = {
    "modelo": "ExtraTreesClassifier",
    "criterio_seleccion": "F1 medio en validación cruzada",
    "particiones_cv": 5,
    "random_state": RANDOM_STATE,
    "mejor_configuracion": best_configuration,
    "mejores_metricas_cv": {
        "accuracy": float(
            results_df.iloc[0]["accuracy_media"]
        ),
        "precision": float(
            results_df.iloc[0]["precision_media"]
        ),
        "recall": float(
            results_df.iloc[0]["recall_media"]
        ),
        "f1": float(
            results_df.iloc[0]["f1_media"]
        ),
        "roc_auc": float(
            results_df.iloc[0]["roc_auc_media"]
        ),
        "f1_desviacion": float(
            results_df.iloc[0]["f1_desviacion"]
        ),
    },
    "nota": (
        "El conjunto de prueba reservado no se utilizó "
        "para seleccionar la configuración."
    ),
}

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=4, ensure_ascii=False)

# Guardar como candidato independiente.
# No sobrescribe el modelo actualmente utilizado por la aplicación.
joblib.dump(best_pipeline, candidate_path)


# ============================================================
# RESUMEN
# ============================================================

print("\n" + "=" * 72)
print("RESULTADOS DE OPTIMIZACIÓN")
print("=" * 72)

columns = [
    "configuracion",
    "n_estimators",
    "max_depth",
    "min_samples_leaf",
    "max_features",
    "class_weight",
    "precision_media",
    "recall_media",
    "f1_media",
    "roc_auc_media",
]

print(
    results_df[columns].to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

print("\nMejor configuración:")
print(best_configuration)

print(f"\nCSV guardado: {csv_path.relative_to(ROOT)}")
print(f"JSON guardado: {json_path.relative_to(ROOT)}")
print(f"Candidato guardado: {candidate_path.relative_to(ROOT)}")

print("\nIMPORTANTE:")
print("El modelo de producción no se ha modificado.")
print("El conjunto de prueba todavía no se ha evaluado.")

print("\nOptimización finalizada.")