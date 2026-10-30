from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = (
    ROOT
    / "data"
    / "processed"
    / "validacion_umbrales_60_20_20.csv"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "umbral_recomendado.json"
)

MIN_PRECISION = 0.70


print("=" * 72)
print("SELECCIÓN DEL UMBRAL")
print("ECOMMERCE DELIVERY INTELLIGENCE")
print("=" * 72)

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"No se encontró el archivo: {CSV_PATH}"
    )

df = pd.read_csv(CSV_PATH)

required_columns = {
    "umbral",
    "precision",
    "recall",
    "f1",
    "falsos_positivos",
    "falsos_negativos",
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(
        f"Faltan columnas necesarias: {sorted(missing)}"
    )

print(f"\nPrecisión mínima exigida: {MIN_PRECISION:.0%}")

eligible = df[df["precision"] >= MIN_PRECISION].copy()

if eligible.empty:
    print(
        "\nNingún umbral cumple la precisión mínima. "
        "Será necesario revisar el criterio."
    )

    result = {
        "estado": "sin_umbral_elegible",
        "precision_minima": MIN_PRECISION,
        "umbral": None,
    }

else:
    # Entre los umbrales que cumplen la precisión mínima,
    # seleccionar el de mayor recall.
    # Si hay empate, preferir mayor F1.
    best = eligible.sort_values(
        by=["recall", "f1", "precision"],
        ascending=[False, False, False],
    ).iloc[0]

    result = {
        "estado": "candidato_seleccionado",
        "criterio": (
            "Mayor recall entre los umbrales con "
            "precision igual o superior al mínimo"
        ),
        "precision_minima": MIN_PRECISION,
        "umbral": float(best["umbral"]),
        "precision_validacion": float(best["precision"]),
        "recall_validacion": float(best["recall"]),
        "f1_validacion": float(best["f1"]),
        "falsos_positivos_validacion": int(
            best["falsos_positivos"]
        ),
        "falsos_negativos_validacion": int(
            best["falsos_negativos"]
        ),
        "nota": (
            "Este umbral se seleccionó con validación. "
            "No se ha realizado una nueva evaluación "
            "sobre el conjunto de prueba."
        ),
    }

    print("\nUmbrales que cumplen la condición:")
    print(
        eligible[
            [
                "umbral",
                "precision",
                "recall",
                "f1",
                "falsos_positivos",
                "falsos_negativos",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nUmbral candidato recomendado:")
    print(f"Umbral: {result['umbral']:.2f}")
    print(
        "Precision en validación: "
        f"{result['precision_validacion']:.2%}"
    )
    print(
        "Recall en validación: "
        f"{result['recall_validacion']:.2%}"
    )
    print(
        "F1 en validación: "
        f"{result['f1_validacion']:.4f}"
    )

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    json.dump(
        result,
        file,
        indent=4,
        ensure_ascii=False,
    )

print(
    "\nResultado guardado en: "
    f"{OUTPUT_PATH.relative_to(ROOT)}"
)