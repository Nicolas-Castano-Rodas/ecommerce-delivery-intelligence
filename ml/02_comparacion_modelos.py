
from pathlib import Path
import json

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.compose import make_column_selector


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "raw" / "Train.csv"
RESULTS_DIR = ROOT_DIR / "data" / "processed"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Reached.on.Time_Y.N"
RANDOM_STATE = 42

df = pd.read_csv(DATA_PATH)

if TARGET not in df.columns:
    raise ValueError(f"No existe la variable objetivo: {TARGET}")

if df[TARGET].isna().any():
    raise ValueError("La variable objetivo contiene valores ausentes.")

if not set(df[TARGET].unique()).issubset({0, 1}):
    raise ValueError("La variable objetivo debe contener solo 0 y 1.")

X = df.drop(columns=[TARGET, "ID"], errors="ignore")
y = df[TARGET].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

numeric_features = X.select_dtypes(include="number").columns.tolist()
categorical_features = X.select_dtypes(exclude="number").columns.tolist()


def make_preprocessor(scale_numeric=False):
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])


models = {
    "Dummy": (
        DummyClassifier(strategy="most_frequent"),
        False,
    ),
    "LogisticRegression": (
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        True,
    ),
    "RandomForest": (
        RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        False,
    ),
    "ExtraTrees": (
        ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        False,
    ),
}

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

print("=" * 70)
print("COMPARACIÓN DE MODELOS - ECOMMERCE DELIVERY INTELLIGENCE")
print("=" * 70)
print(f"Registros: {len(df):,}")
print(f"Entrenamiento: {len(X_train):,}")
print(f"Prueba reservada: {len(X_test):,}")
print("\nEvaluación cruzada de 5 particiones sobre entrenamiento...\n")

for name, (classifier, scale_numeric) in models.items():
    pipeline = Pipeline([
        ("preprocessor", make_preprocessor(scale_numeric)),
        ("classifier", classifier),
    ])

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        error_score="raise",
    )

    results.append({
        "modelo": name,
        "accuracy_media": scores["test_accuracy"].mean(),
        "precision_media": scores["test_precision"].mean(),
        "recall_media": scores["test_recall"].mean(),
        "f1_media": scores["test_f1"].mean(),
        "roc_auc_media": scores["test_roc_auc"].mean(),
    })

    print(
        f"{name:22} "
        f"Recall={scores['test_recall'].mean():.4f} | "
        f"F1={scores['test_f1'].mean():.4f} | "
        f"ROC-AUC={scores['test_roc_auc'].mean():.4f}"
    )

results_df = pd.DataFrame(results).sort_values(
    by="f1_media",
    ascending=False,
)

csv_path = RESULTS_DIR / "comparacion_modelos_cv.csv"
json_path = RESULTS_DIR / "comparacion_modelos_cv.json"

results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
results_df.to_json(
    json_path,
    orient="records",
    indent=2,
    force_ascii=False,
)

print("\n" + "=" * 70)
print("RESULTADOS DE VALIDACIÓN CRUZADA")
print("=" * 70)
print(results_df.round(4).to_string(index=False))
print(f"\nCSV guardado: {csv_path.relative_to(ROOT_DIR)}")
print(f"JSON guardado: {json_path.relative_to(ROOT_DIR)}")
print("\nEl conjunto de prueba se ha reservado para la evaluación final.")
