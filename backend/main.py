
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# CONFIGURACIÓN
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT_DIR / "ml" / "artifacts" / "delivery_delay_model.joblib"
DATASET_PATH = ROOT_DIR / "data" / "raw" / "Train.csv"

CANDIDATE_MODEL_PATH = (
    ROOT_DIR
    / "ml"
    / "artifacts"
    / "extra_trees_validated_candidate.joblib"
)

CANDIDATE_THRESHOLD = 0.50

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"No se encontró el modelo entrenado: {MODEL_PATH}"
    )

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"No se encontró el dataset: {DATASET_PATH}"
    )

model = joblib.load(MODEL_PATH)

candidate_model = None

if CANDIDATE_MODEL_PATH.exists():
    candidate_model = joblib.load(CANDIDATE_MODEL_PATH)
    print("Modelo candidato Extra Trees cargado correctamente.")
else:
    print("Modelo candidato no encontrado. Se conserva Random Forest.")

# Dataset histórico utilizado para las estadísticas del dashboard.
dataset = pd.read_csv(DATASET_PATH)

REQUIRED_COLUMNS = [
    "Warehouse_block",
    "Mode_of_Shipment",
    "Product_importance",
    "Customer_rating",
    "Reached.on.Time_Y.N",
]

missing_columns = [
    column for column in REQUIRED_COLUMNS
    if column not in dataset.columns
]

if missing_columns:
    raise ValueError(
        f"Faltan columnas requeridas en Train.csv: {missing_columns}"
    )


# ============================================================
# APLICACIÓN FASTAPI
# ============================================================

app = FastAPI(
    title="Ecommerce Delivery Intelligence API",
    description=(
        "API de análisis histórico y predicción de retrasos "
        "en entregas utilizando Machine Learning."
    ),
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ============================================================
# ESQUEMA DE ENTRADA
# ============================================================

class DeliveryInput(BaseModel):
    Warehouse_block: Literal["A", "B", "C", "D", "F"]
    Mode_of_Shipment: Literal["Flight", "Road", "Ship"]
    Customer_care_calls: int = Field(ge=0)
    Customer_rating: int = Field(ge=1, le=5)
    Cost_of_the_Product: float = Field(ge=0)
    Prior_purchases: int = Field(ge=0)
    Product_importance: Literal["low", "medium", "high"]
    Gender: Literal["F", "M"]
    Discount_offered: float = Field(ge=0)
    Weight_in_gms: float = Field(ge=0)


# ============================================================
# ENDPOINTS GENERALES
# ============================================================

@app.get("/")
def root():
    return {
        "project": "Ecommerce Delivery Intelligence",
        "status": "online",
        "model": "RandomForestClassifier",
        "docs": "/docs",
        "stats": "/stats",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "dataset_loaded": not dataset.empty,
    }


# ============================================================
# ESTADÍSTICAS HISTÓRICAS PARA EL DASHBOARD
# ============================================================

@app.get("/stats")
def get_stats():
    try:
        total = len(dataset)

        # En este dataset, 1 significa que el envío NO llegó a tiempo.
        delayed = int(
            (dataset["Reached.on.Time_Y.N"] == 1).sum()
        )
        on_time = int(
            (dataset["Reached.on.Time_Y.N"] == 0).sum()
        )

        if total == 0:
            raise HTTPException(
                status_code=500,
                detail="El dataset no contiene registros."
            )

        def distribution_by(column: str):
            grouped = (
                dataset.groupby(column, dropna=False)
                ["Reached.on.Time_Y.N"]
                .agg(["count", "sum"])
                .reset_index()
            )

            grouped.columns = [
                "category",
                "total",
                "delayed",
            ]

            grouped["on_time"] = (
                grouped["total"] - grouped["delayed"]
            )

            grouped["delay_percent"] = (
                grouped["delayed"] / grouped["total"] * 100
            ).round(2)

            grouped["on_time_percent"] = (
                grouped["on_time"] / grouped["total"] * 100
            ).round(2)

            return grouped.to_dict(orient="records")

        return {
            "source": "data/raw/Train.csv",
            "data_type": "historical",
            "total_shipments": total,
            "delayed_shipments": delayed,
            "on_time_shipments": on_time,
            "delay_percent": round(delayed / total * 100, 2),
            "on_time_percent": round(on_time / total * 100, 2),
            "by_shipment": distribution_by("Mode_of_Shipment"),
            "by_warehouse": distribution_by("Warehouse_block"),
            "by_importance": distribution_by("Product_importance"),
            "by_rating": distribution_by("Customer_rating"),
        }

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="No fue posible calcular las estadísticas históricas.",
        ) from error


# ============================================================
# PREDICCIÓN CON MACHINE LEARNING
# ============================================================

@app.post("/predict")
def predict_delivery(data: DeliveryInput):
    try:
        input_data = pd.DataFrame([data.model_dump()])

        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]

        class_index = list(model.classes_).index(1)
        probability_delay = float(probabilities[class_index])

        return {
            "prediction": prediction,
            "prediction_label": (
                "No llegó a tiempo"
                if prediction == 1
                else "Llegó a tiempo"
            ),
            "probability_delay": round(probability_delay, 4),
            "probability_delay_percent": round(
                probability_delay * 100, 2
            ),
            "model": "RandomForestClassifier",
            "interpretation": (
                "El modelo estima un riesgo de retraso "
                "relativamente alto."
                if probability_delay >= 0.5
                else
                "El modelo estima una probabilidad de "
                "retraso inferior al 50 %."
            ),
            "disclaimer": (
                "Estimación del modelo, no una garantía "
                "del resultado real de la entrega."
            ),
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="No fue posible generar la predicción.",
        ) from error


# ============================================================
# PREDICCIÓN CON EL MODELO CANDIDATO EXTRA TREES
# ============================================================

@app.post("/predict-candidate")
def predict_delivery_candidate(data: DeliveryInput):
    if candidate_model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "El modelo candidato no está disponible. "
                "Verifica el archivo extra_trees_validated_candidate.joblib."
            ),
        )

    try:
        input_data = pd.DataFrame([data.model_dump()])

        probabilities = candidate_model.predict_proba(input_data)[0]

        class_index = list(candidate_model.classes_).index(1)
        probability_delay = float(probabilities[class_index])

        # Aplicamos el umbral elegido durante la validación.
        prediction = int(
            probability_delay >= CANDIDATE_THRESHOLD
        )

        return {
            "prediction": prediction,
            "prediction_label": (
                "No llegó a tiempo"
                if prediction == 1
                else "Llegó a tiempo"
            ),
            "probability_delay": round(probability_delay, 4),
            "probability_delay_percent": round(
                probability_delay * 100, 2
            ),
            "threshold": CANDIDATE_THRESHOLD,
            "model": "ExtraTreesClassifier",
            "interpretation": (
                "El modelo estima un riesgo de retraso "
                "igual o superior al umbral establecido."
                if prediction == 1
                else
                "El modelo estima un riesgo de retraso "
                "inferior al umbral establecido."
            ),
            "disclaimer": (
                "Es una estimación del modelo, no una garantía "
                "del resultado real de la entrega."
            ),
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="No fue posible generar la predicción del candidato.",
        ) from error