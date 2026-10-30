
"use client";

import { useEffect, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BrainCircuit,
  LoaderCircle,
  Wifi,
  WifiOff,
} from "lucide-react";

const API_URL = "http://127.0.0.1:8000";

type DeliveryInput = {
  Warehouse_block: string;
  Mode_of_Shipment: string;
  Customer_care_calls: number;
  Customer_rating: number;
  Cost_of_the_Product: number;
  Prior_purchases: number;
  Product_importance: string;
  Gender: string;
  Discount_offered: number;
  Weight_in_gms: number;
};

type PredictionResponse = {
  prediction: number;
  prediction_label: string;
  probability_delay: number;
  probability_delay_percent: number;
  model: string;
  interpretation: string;
  disclaimer: string;
};

const initialValues: DeliveryInput = {
  Warehouse_block: "A",
  Mode_of_Shipment: "Ship",
  Customer_care_calls: 3,
  Customer_rating: 4,
  Cost_of_the_Product: 200,
  Prior_purchases: 3,
  Product_importance: "medium",
  Gender: "F",
  Discount_offered: 10,
  Weight_in_gms: 3500,
};

const inputClass =
  "mt-2 w-full rounded-xl border border-white/10 bg-[#0b1020] px-3 py-3 text-sm text-slate-100 outline-none transition focus:border-cyan-400/60";

export default function PredictionPanel() {
  const [form, setForm] = useState<DeliveryInput>(initialValues);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;

    async function checkApi() {
      try {
        const response = await fetch(`${API_URL}/health`);
        if (!response.ok) throw new Error("API no disponible");
        if (active) setApiOnline(true);
      } catch {
        if (active) setApiOnline(false);
      }
    }

    void checkApi();

    return () => {
      active = false;
    };
  }, []);

  function updateField<K extends keyof DeliveryInput>(
    field: K,
    value: DeliveryInput[K],
  ) {
    setForm((previous) => ({ ...previous, [field]: value }));
  }

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail
            ? JSON.stringify(data.detail)
            : "No fue posible generar la predicción.",
        );
      }

      setResult(data as PredictionResponse);
      setApiOnline(true);
    } catch (err) {
      setApiOnline(false);
      setError(
        err instanceof Error
          ? `${err.message}. Verifica que FastAPI esté ejecutándose en el puerto 8000 y que CORS permita localhost:3000.`
          : "Error inesperado al conectar con la API.",
      );
    } finally {
      setLoading(false);
    }
  }

  const highRisk = result?.prediction === 1;
  const probability = Math.min(
    100,
    Math.max(0, result?.probability_delay_percent ?? 0),
  );

  return (
    <section className="overflow-hidden rounded-2xl border border-cyan-400/20 bg-[#101626]">
      <div className="flex flex-col justify-between gap-4 border-b border-white/[0.07] p-5 sm:flex-row sm:items-center sm:p-6">
        <div className="flex items-start gap-3">
          <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-cyan-400/10 text-cyan-300">
            <BrainCircuit size={23} />
          </div>
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="text-lg font-semibold">
                Predictor de retrasos
              </h2>
              <span className="rounded-md border border-cyan-400/20 bg-cyan-400/10 px-2 py-1 text-[10px] font-bold uppercase tracking-wider text-cyan-300">
                Machine Learning
              </span>
            </div>
            <p className="mt-1 text-sm text-slate-400">
              Evalúa un envío con el modelo Random Forest entrenado en Python.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 text-xs">
          {apiOnline === true ? (
            <>
              <Wifi size={15} className="text-emerald-300" />
              <span className="text-emerald-300">API conectada</span>
            </>
          ) : apiOnline === false ? (
            <>
              <WifiOff size={15} className="text-rose-300" />
              <span className="text-rose-300">API desconectada</span>
            </>
          ) : (
            <>
              <Activity size={15} className="text-amber-300" />
              <span className="text-amber-300">Comprobando API...</span>
            </>
          )}
        </div>
      </div>

      <div className="grid gap-6 p-5 sm:p-6 xl:grid-cols-[1.2fr_0.8fr]">
        <form onSubmit={handleSubmit} className="space-y-5">
          <div>
            <h3 className="text-sm font-semibold text-slate-200">
              Características del envío
            </h3>
            <p className="mt-1 text-xs text-slate-500">
              Introduce los valores para estimar el riesgo de retraso.
            </p>
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <label className="text-xs font-medium text-slate-400">
              Bloque del almacén
              <select
                className={inputClass}
                value={form.Warehouse_block}
                onChange={(e) =>
                  updateField("Warehouse_block", e.target.value)
                }
              >
                {["A", "B", "C", "D", "F"].map((value) => (
                  <option key={value} value={value}>
                    Bloque {value}
                  </option>
                ))}
              </select>
            </label>

            <label className="text-xs font-medium text-slate-400">
              Medio de transporte
              <select
                className={inputClass}
                value={form.Mode_of_Shipment}
                onChange={(e) =>
                  updateField("Mode_of_Shipment", e.target.value)
                }
              >
                <option value="Ship">Marítimo</option>
                <option value="Flight">Avión</option>
                <option value="Road">Terrestre</option>
              </select>
            </label>

            <label className="text-xs font-medium text-slate-400">
              Llamadas de atención al cliente
              <input
                className={inputClass}
                type="number"
                min="0"
                max="100"
                required
                value={form.Customer_care_calls}
                onChange={(e) =>
                  updateField("Customer_care_calls", Number(e.target.value))
                }
              />
            </label>

            <label className="text-xs font-medium text-slate-400">
              Calificación del cliente (1–5)
              <select
                className={inputClass}
                value={form.Customer_rating}
                onChange={(e) =>
                  updateField("Customer_rating", Number(e.target.value))
                }
              >
                {[1, 2, 3, 4, 5].map((value) => (
                  <option key={value} value={value}>
                    {value} / 5
                  </option>
                ))}
              </select>
            </label>

            <label className="text-xs font-medium text-slate-400">
              Costo del producto
              <input
                className={inputClass}
                type="number"
                min="0"
                required
                value={form.Cost_of_the_Product}
                onChange={(e) =>
                  updateField(
                    "Cost_of_the_Product",
                    Number(e.target.value),
                  )
                }
              />
            </label>

            <label className="text-xs font-medium text-slate-400">
              Compras anteriores
              <input
                className={inputClass}
                type="number"
                min="0"
                required
                value={form.Prior_purchases}
                onChange={(e) =>
                  updateField("Prior_purchases", Number(e.target.value))
                }
              />
            </label>

            <label className="text-xs font-medium text-slate-400">
              Importancia del producto
              <select
                className={inputClass}
                value={form.Product_importance}
                onChange={(e) =>
                  updateField("Product_importance", e.target.value)
                }
              >
                <option value="low">Baja</option>
                <option value="medium">Media</option>
                <option value="high">Alta</option>
              </select>
            </label>

            <label className="text-xs font-medium text-slate-400">
              Género registrado en el dataset
              <select
                className={inputClass}
                value={form.Gender}
                onChange={(e) => updateField("Gender", e.target.value)}
              >
                <option value="F">F</option>
                <option value="M">M</option>
              </select>
            </label>

            <label className="text-xs font-medium text-slate-400">
              Descuento ofrecido
              <input
                className={inputClass}
                type="number"
                min="0"
                required
                value={form.Discount_offered}
                onChange={(e) =>
                  updateField("Discount_offered", Number(e.target.value))
                }
              />
            </label>

            <label className="text-xs font-medium text-slate-400">
              Peso del producto (gramos)
              <input
                className={inputClass}
                type="number"
                min="0"
                required
                value={form.Weight_in_gms}
                onChange={(e) =>
                  updateField("Weight_in_gms", Number(e.target.value))
                }
              />
            </label>
          </div>

          {error && (
            <div
              role="alert"
              className="rounded-xl border border-rose-400/20 bg-rose-400/[0.06] p-4 text-sm leading-6 text-rose-200"
            >
              <div className="mb-1 flex items-center gap-2 font-semibold">
                <AlertTriangle size={16} />
                No se pudo generar la predicción
              </div>
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-cyan-400 px-4 py-3 text-sm font-bold text-slate-950 transition hover:bg-cyan-300 disabled:cursor-not-allowed disabled:opacity-60 sm:w-auto"
          >
            {loading ? (
              <LoaderCircle size={17} className="animate-spin" />
            ) : (
              <BrainCircuit size={17} />
            )}
            {loading ? "Analizando envío..." : "Predecir retraso"}
          </button>
        </form>

        <aside className="flex min-h-[300px] flex-col rounded-2xl border border-white/[0.07] bg-[#0b1020] p-5">
          <div className="flex items-center gap-2 text-sm font-semibold">
            <Activity size={17} className="text-cyan-300" />
            Resultado del modelo
          </div>

          {!result && !loading && (
            <div className="flex flex-1 flex-col items-center justify-center py-10 text-center">
              <div className="flex h-16 w-16 items-center justify-center rounded-2xl border border-white/[0.07] bg-white/[0.025] text-slate-500">
                <BrainCircuit size={30} />
              </div>
              <p className="mt-4 font-medium text-slate-300">
                Esperando datos
              </p>
              <p className="mt-2 max-w-xs text-xs leading-5 text-slate-500">
                Completa las características y ejecuta una predicción para
                consultar el resultado de tu modelo.
              </p>
            </div>
          )}

          {loading && (
            <div className="flex flex-1 flex-col items-center justify-center py-10 text-center">
              <LoaderCircle
                size={34}
                className="animate-spin text-cyan-300"
              />
              <p className="mt-4 font-medium text-slate-300">
                Procesando características...
              </p>
              <p className="mt-2 text-xs text-slate-500">
                Esperando la respuesta de FastAPI.
              </p>
            </div>
          )}

          {result && !loading && (
            <div className="mt-6 space-y-5">
              <div
                className={`rounded-xl border p-4 ${
                  highRisk
                    ? "border-rose-400/20 bg-rose-400/[0.06]"
                    : "border-emerald-400/20 bg-emerald-400/[0.06]"
                }`}
              >
                <p className="text-xs text-slate-400">
                  Clasificación estimada
                </p>
                <p
                  className={`mt-2 text-xl font-bold ${
                    highRisk ? "text-rose-300" : "text-emerald-300"
                  }`}
                >
                  {result.prediction_label}
                </p>
                <p className="mt-2 text-xs leading-5 text-slate-400">
                  {result.interpretation}
                </p>
              </div>

              <div>
                <div className="flex items-end justify-between gap-3">
                  <span className="text-xs text-slate-400">
                    Probabilidad estimada de retraso
                  </span>
                  <span className="text-2xl font-bold tabular-nums">
                    {probability.toFixed(2)}%
                  </span>
                </div>

                <div
                  className="mt-3 h-2.5 overflow-hidden rounded-full bg-white/[0.07]"
                  role="progressbar"
                  aria-label="Probabilidad estimada de retraso"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={probability}
                >
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      highRisk ? "bg-rose-400" : "bg-emerald-400"
                    }`}
                    style={{ width: `${probability}%` }}
                  />
                </div>
              </div>

              <div className="rounded-xl border border-white/[0.06] p-3">
                <p className="text-xs text-slate-500">Modelo utilizado</p>
                <p className="mt-1 text-sm font-medium text-slate-200">
                  {result.model}
                </p>
              </div>

              <p className="text-[11px] leading-5 text-slate-500">
                {result.disclaimer}
              </p>
            </div>
          )}
        </aside>
      </div>

      <div className="border-t border-white/[0.06] px-5 py-3 sm:px-6">
        <p className="text-[11px] leading-5 text-slate-500">
          La predicción es experimental. El rendimiento del modelo debe
          validarse antes de utilizarse para tomar decisiones operativas.
        </p>
      </div>
    </section>
  );
}
