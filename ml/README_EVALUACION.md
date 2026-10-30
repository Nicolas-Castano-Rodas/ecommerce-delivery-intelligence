# Evaluación comparativa de modelos

## Objetivo

Comparar Random Forest y Extra Trees para predecir si un envío llegará a tiempo, utilizando el mismo conjunto de prueba y los umbrales definidos previamente en validación.

## Configuración

- Dataset: `data/raw/Train.csv`
- Registros: 10.999
- Variable objetivo: `Reached.on.Time_Y.N`
- División: entrenamiento, validación y prueba
- Semilla de división: `2026`
- Registros de entrenamiento: 6.599
- Registros de validación: 2.200
- Registros de prueba: 2.200
- Umbral Random Forest: `0.50`
- Umbral Extra Trees: `0.60`

## Resultados en prueba

| Métrica | Random Forest | Extra Trees |
|---|---:|---:|
| Accuracy | 0.6950 | 0.6850 |
| Precision | 0.8812 | 0.8846 |
| Recall | 0.5651 | 0.5430 |
| F1-score | 0.6886 | 0.6730 |
| ROC-AUC | 0.7568 | 0.7587 |
| Falsos positivos | 100 | 93 |
| Falsos negativos | 571 | 600 |
| Alertas | 842 | 806 |

## Decisión provisional

Random Forest se selecciona como modelo principal provisional porque obtiene mejor exactitud, recall y F1-score. Extra Trees obtiene una precisión ligeramente superior y siete falsos positivos menos, pero registra 29 falsos negativos adicionales.

La elección definitiva debe considerar el coste operativo de las alertas incorrectas y de los retrasos no detectados.

## Limitaciones

El dataset se ha utilizado en experimentos anteriores, por lo que el conjunto de prueba no debe considerarse completamente independiente de todas las decisiones de desarrollo. Los resultados son una evaluación comparativa del experimento actual, no una garantía del rendimiento en producción.

Además, deben revisarse la disponibilidad real de las variables en el momento de la predicción, el posible sesgo del dataset y el rendimiento con datos nuevos antes de desplegar el modelo.
