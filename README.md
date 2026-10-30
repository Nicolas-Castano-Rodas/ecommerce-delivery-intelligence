# Ecommerce Delivery Intelligence

**Machine Learning · Data Science · Next.js · FastAPI**

An end-to-end data science project that analyzes e-commerce shipping data and predicts the risk of delivery delays. It combines a modern web dashboard, a REST API, and machine learning models to support logistics analysis and delivery-risk assessment.

## Overview

Ecommerce Delivery Intelligence explores historical shipment data to identify delivery patterns and estimate whether a shipment is likely to arrive late.

The application includes:

- Historical delivery statistics and visual analytics.
- A responsive web interface built with Next.js.
- A REST API powered by FastAPI.
- A Random Forest classifier for delivery-delay prediction.
- An Extra Trees candidate model for comparative evaluation.
- Model evaluation, threshold analysis, and precision-recall studies.
- Validation of incoming prediction data using Pydantic.

> **Important:** Predictions are estimates produced by machine learning models. They do not guarantee the actual outcome of a delivery.

## Technology Stack

| Area | Technologies |
|---|---|
| Frontend | Next.js, React, TypeScript |
| Backend | Python, FastAPI, Uvicorn |
| Data processing | pandas, NumPy |
| Machine Learning | scikit-learn, Random Forest, Extra Trees |
| Model persistence | joblib |
| Data visualization | Matplotlib |
| Version control | Git, GitHub |

## Project Structure

```text
ecommerce-delivery-intelligence/
├── backend/
│   ├── main.py
│   └── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── frontend/
│   └── src/
│       ├── app/
│       └── components/
├── ml/
│   ├── artifacts/
│   ├── 01_entrenamiento_modelo.py
│   ├── 02_comparacion_modelos.py
│   ├── 03_optimizar_modelo.py
│   ├── 04_evaluacion_final.py
│   ├── 05_analisis_umbrales.py
│   ├── 06_validacion_independiente.py
│   ├── 07_seleccion_umbral.py
│   ├── 08_evaluacion_umbral_final.py
│   ├── 09_comparacion_justa.py
│   ├── 10_analisis_precision_recall.py
│   ├── 11_evaluacion_final_comparativa.py
│   ├── README_EVALUACION.md
│   └── requirements.txt
├── notebooks/
├── package.json
├── README.md
└── .gitignore
```

## Dataset

The project uses the **E-Commerce Shipping Data** dataset, which contains shipment characteristics and the target variable `Reached.on.Time_Y.N`.

The target is interpreted as follows:

- `1`: The shipment did not arrive on time.
- `0`: The shipment arrived on time.

The dataset used in the current experiments contains **10,999 records**.

The raw dataset is intentionally excluded from the Git repository. To reproduce the current application, obtain the dataset and place it at:

```text
data/raw/Train.csv
```

## Machine Learning Evaluation

The project compares Random Forest and Extra Trees using the same train-validation-test split and preprocessing approach for the reported final comparison.

The following results were obtained on the final test split:

| Metric | Random Forest | Extra Trees |
|---|---:|---:|
| Accuracy | 69.50% | 68.50% |
| Precision | 88.12% | 88.46% |
| Recall | 56.51% | 54.30% |
| F1-score | 68.86% | 67.30% |
| ROC-AUC | 75.68% | 75.87% |
| False positives | 100 | 93 |
| False negatives | 571 | 600 |

### Interpretation

Random Forest achieved higher accuracy, recall, and F1-score in this comparison. Extra Trees achieved slightly higher precision and ROC-AUC.

Random Forest is retained as the main model for the current application. The Extra Trees model is available as a candidate for further evaluation.

These results describe the evaluated dataset and split. They should not be interpreted as a guarantee of performance on future shipments or unseen operational environments.

For additional details, see [`ml/README_EVALUACION.md`](ml/README_EVALUACION.md).

## Prerequisites

Install the following tools:

- Node.js and npm.
- Python with `venv` support.
- Git.

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/Nicolas-Castano-Rodas/ecommerce-delivery-intelligence.git
cd ecommerce-delivery-intelligence
```

### 2. Install frontend dependencies

```bash
npm install --prefix frontend
```

### 3. Create and activate a Python virtual environment

On Windows CMD:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

### 4. Install backend dependencies

```cmd
python -m pip install -r backend\requirements.txt
```

### 5. Prepare the dataset and model artifacts

The raw dataset and trained model files are excluded from this repository.

The backend expects these files at the following paths:

```text
data/raw/Train.csv
ml/artifacts/delivery_delay_model.joblib
```

The optional Extra Trees endpoint also requires:

```text
ml/artifacts/extra_trees_validated_candidate.joblib
```

Place the required files in those locations before starting the backend. The model files must be compatible with the installed Python and scikit-learn versions.

### 6. Start the backend

From the repository root:

```cmd
python -m uvicorn backend.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/health
```

Historical statistics:

```text
http://127.0.0.1:8000/stats
```

### 7. Start the frontend

Open a second terminal at the repository root:

```cmd
npm run dev
```

Open:

```text
http://localhost:3000
```

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API information |
| GET | `/health` | Service health check |
| GET | `/stats` | Historical shipment statistics |
| POST | `/predict` | Prediction using Random Forest |
| POST | `/predict-candidate` | Prediction using the optional Extra Trees candidate |

The prediction endpoints validate shipment attributes and return a prediction, estimated delay probability, and interpretation.

## Evaluation and Reproducibility

The `ml/` directory contains the training and evaluation scripts used throughout the project.

The `data/processed/` directory contains selected evaluation results, tables, and visualizations.

To reproduce the reported results, use the documented data split, preprocessing, model configuration, and thresholds. The final test results should remain separate from decisions made during model selection.

## Current Limitations

- The models and raw dataset are not included in the repository.
- The reported metrics come from the available dataset and evaluation procedure; performance on new operational data may differ.
- The prediction is a risk estimate, not a guarantee of a delivery outcome.
- Further validation on independently collected data would be necessary before using the predictions for real logistics decisions.

## Author

**Nicolás Castaño Rodas**

Full Stack Developer | Python, Machine Learning, Next.js & FastAPI

GitHub: [Nicolas-Castano-Rodas](https://github.com/Nicolas-Castano-Rodas)