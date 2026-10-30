from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


# --------------------------------------------------
# 1. Configuración de rutas
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT_DIR / "data" / "raw" / "Train.csv"
OUTPUT_DIR = ROOT_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Reached.on.Time_Y.N"


# --------------------------------------------------
# 2. Cargar los datos
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"No se encontró el dataset en: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

if TARGET not in df.columns:
    raise ValueError(
        f"No se encontró la variable objetivo: {TARGET}"
    )

print("\n" + "=" * 60)
print("ECOMMERCE DELIVERY INTELLIGENCE")
print("ANÁLISIS EXPLORATORIO DE DATOS")
print("=" * 60)


# --------------------------------------------------
# 3. Dimensiones y tipos de datos
# --------------------------------------------------

print("\n1. DIMENSIONES DEL DATASET")
print(f"Filas: {df.shape[0]:,}")
print(f"Columnas: {df.shape[1]}")

print("\n2. TIPOS DE DATOS")
print(df.dtypes.to_string())


# --------------------------------------------------
# 4. Valores ausentes y duplicados
# --------------------------------------------------

print("\n3. VALORES AUSENTES")

missing = df.isna().sum()

print(
    missing[missing > 0].to_string()
    if missing.any()
    else "No se encontraron valores ausentes."
)

print("\n4. REGISTROS DUPLICADOS")

duplicate_count = df.duplicated().sum()

print(f"Filas duplicadas: {duplicate_count}")

print("\n5. IDENTIFICADORES DUPLICADOS")

if "ID" in df.columns:
    print(f"ID duplicados: {df['ID'].duplicated().sum()}")


# --------------------------------------------------
# 5. Distribución de la variable objetivo
# --------------------------------------------------

print("\n6. DISTRIBUCIÓN DE LOS RESULTADOS")

target_counts = df[TARGET].value_counts().sort_index()

target_percentages = (
    df[TARGET]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
)

target_summary = pd.DataFrame({
    "cantidad": target_counts,
    "porcentaje": target_percentages.round(2),
})

target_summary.index.name = TARGET

print(target_summary.to_string())

target_summary.to_csv(
    OUTPUT_DIR / "distribucion_entregas.csv",
    encoding="utf-8-sig",
)


# --------------------------------------------------
# 6. Correlaciones numéricas iniciales
# --------------------------------------------------

print("\n7. CORRELACIÓN CON LA VARIABLE OBJETIVO")

numeric_df = df.select_dtypes(include="number").drop(
    columns=["ID"],
    errors="ignore",
)

correlations = (
    numeric_df.corr()[TARGET]
    .drop(TARGET)
    .sort_values(key=abs, ascending=False)
)

print(correlations.round(3).to_string())

correlations.rename("correlacion").to_csv(
    OUTPUT_DIR / "correlaciones_iniciales.csv",
    encoding="utf-8-sig",
)


# --------------------------------------------------
# 7. Tasas de retraso por categorías
# --------------------------------------------------

print("\n8. TASA DE RETRASO POR CATEGORÍA")

categorical_columns = [
    "Warehouse_block",
    "Mode_of_Shipment",
    "Product_importance",
    "Gender",
]

category_summaries = []

for column in categorical_columns:
    if column not in df.columns:
        continue

    summary = (
        df.groupby(column, dropna=False)[TARGET]
        .agg(["count", "mean"])
        .rename(columns={
            "count": "cantidad",
            "mean": "proporcion_retraso",
        })
    )

    summary["porcentaje_retraso"] = (
        summary["proporcion_retraso"] * 100
    ).round(2)

    summary["variable"] = column
    summary["categoria"] = summary.index.astype(str)

    summary = summary.reset_index(drop=True)

    category_summaries.append(summary)

    print(f"\n{column}")
    print(
        summary[
            ["categoria", "cantidad", "porcentaje_retraso"]
        ].to_string(index=False)
    )

if category_summaries:
    category_results = pd.concat(
        category_summaries,
        ignore_index=True,
    )

    category_results.to_csv(
        OUTPUT_DIR / "tasas_retraso_por_categoria.csv",
        index=False,
        encoding="utf-8-sig",
    )


# --------------------------------------------------
# 8. Gráfico de distribución de entregas
# --------------------------------------------------

labels = {
    0: "Llegó a tiempo",
    1: "No llegó a tiempo",
}

plot_labels = [
    labels.get(value, str(value))
    for value in target_counts.index
]

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(
    plot_labels,
    target_counts.values,
)

ax.set_title("Distribución de resultados de entrega")
ax.set_ylabel("Número de registros")
ax.set_xlabel("Resultado")

ax.bar_label(bars, padding=3)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "distribucion_entregas.png",
    dpi=150,
    bbox_inches="tight",
)

plt.close(fig)


# --------------------------------------------------
# 9. Gráfico de tasa de retraso por transporte
# --------------------------------------------------

if "Mode_of_Shipment" in df.columns:
    shipment_rates = (
        df.groupby("Mode_of_Shipment")[TARGET]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        shipment_rates.index.astype(str),
        shipment_rates.values,
    )

    ax.set_title("Porcentaje de envíos que no llegaron a tiempo")
    ax.set_xlabel("Medio de transporte")
    ax.set_ylabel("Porcentaje de retraso")
    ax.set_ylim(0, 100)

    ax.bar_label(
        bars,
        fmt="%.1f%%",
        padding=3,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "retrasos_por_transporte.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# --------------------------------------------------
# 10. Resumen final
# --------------------------------------------------

print("\n9. ARCHIVOS GENERADOS")

for file_path in sorted(OUTPUT_DIR.iterdir()):
    if file_path.is_file():
        print(f"- {file_path.name}")

print("\nAnálisis exploratorio terminado correctamente.")