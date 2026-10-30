from pathlib import Path
import json

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "Train.csv"
OUTPUT_DIR = ROOT / "data" / "processed"

TARGET = "Reached.on.Time_Y.N"
RANDOM_STATE = 2026

# Umbrales elegidos previamente usando exclusivamente validación.
THRESHOLDS = {
    "RandomForestClassifier": 0.50,
    "ExtraTreesClassifier": 0.60,
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 1. Cargar datos
df = pd.read_csv(DATA_PATH)

if TARGET not in df.columns or df[TARGET].isna().any():
    raise ValueError("La variable objetivo no existe o contiene nulos.")

X = df.drop(columns=[TARGET, "ID"], errors="ignore")
y = df[TARGET].astype(int)


# 2. Reproducir exactamente la división 60/20/20
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


# 3. Preprocesamiento compartido
numeric_features = X_train.select_dtypes(
    include="number"
).columns.tolist()

categorical_features = X_train.select_dtypes(
    exclude="number"
).columns.tolist()


def create_preprocessor():
    transformers = []

    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
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
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
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

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


# 4. Configurar los modelos
models = {
    "RandomForestClassifier": Pipeline(
        [
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
        [
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


# 5. Entrenar y evaluar en prueba una sola vez
results = []

print("=" * 72)
print("EVALUACIÓN FINAL COMPARATIVA")
print("=" * 72)
print(f"Entrenamiento: {len(X_train)}")
print(f"Validación: {len(X_val)}")
print(f"Prueba reservada: {len(X_test)}")

for name, model in models.items():
    threshold = THRESHOLDS[name]

    print(f"\nEvaluando {name} con umbral {threshold:.2f}...")

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    results.append(
        {
            "modelo": name,
            "umbral": threshold,
            "accuracy": float(
                accuracy_score(y_test, predictions)
            ),
            "precision": float(
                precision_score(
                    y_test, predictions, zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    y_test, predictions, zero_division=0
                )
            ),
            "f1_score": float(
                f1_score(
                    y_test, predictions, zero_division=0
                )
            ),
            "roc_auc": float(
                roc_auc_score(y_test, probabilities)
            ),
            "verdaderos_negativos": int(tn),
            "falsos_positivos": int(fp),
            "falsos_negativos": int(fn),
            "verdaderos_positivos": int(tp),
            "total_alertas": int(predictions.sum()),
        }
    )


# 6. Guardar resultados
results_df = pd.DataFrame(results)

csv_path = OUTPUT_DIR / "evaluacion_final_comparativa.csv"
json_path = OUTPUT_DIR / "evaluacion_final_comparativa.json"

results_df.to_csv(
    csv_path,
    index=False,
    encoding="utf-8-sig",
)

summary = {
    "experimento": "Evaluación comparativa con umbrales fijados",
    "random_state": RANDOM_STATE,
    "registros_prueba": int(len(X_test)),
    "umbrales_seleccionados_en_validacion": THRESHOLDS,
    "resultados": results,
    "advertencia": (
        "El dataset ya fue explorado durante experimentos previos. "
        "Aunque estos umbrales se fijaron con validación y aquí se "
        "evalúan sobre la partición de prueba, esta evaluación no "
        "es completamente independiente de todas las decisiones "
        "anteriores del proyecto."
    ),
}

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=4, ensure_ascii=False)

print("\n" + "=" * 72)
print("RESULTADOS EN EL CONJUNTO DE PRUEBA")
print("=" * 72)

print(
    results_df[
        [
            "modelo",
            "umbral",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "falsos_positivos",
            "falsos_negativos",
            "total_alertas",
        ]
    ].to_string(index=False)
)

print(f"\nCSV guardado: {csv_path.relative_to(ROOT)}")
print(f"JSON guardado: {json_path.relative_to(ROOT)}")
print("\nEvaluación terminada. No se modificó el backend.")