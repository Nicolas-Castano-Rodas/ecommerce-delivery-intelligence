"use client";

import { useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  Bell,
  CalendarDays,
  CheckCircle2,
  ChevronDown,
  CircleHelp,
  Clock3,
  Download,
  LayoutDashboard,
  Menu,
  Package,
  Search,
  Settings,
  ShieldCheck,
  Truck,
  Users,
  X,
  Zap,
} from "lucide-react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type Order = {
  id: string;
  customer: string;
  destination: string;
  segment: string;
  scheduled: string;
  status: "Entregado" | "En tránsito" | "Retrasado";
  risk: "Bajo" | "Medio" | "Alto";
  days: number;
};

const orders: Order[] = [
  { id: "ORD-10482", customer: "María González", destination: "Medellín, CO", segment: "Consumidor", scheduled: "24 Oct, 2026", status: "Entregado", risk: "Bajo", days: 2 },
  { id: "ORD-10481", customer: "Carlos Ramírez", destination: "Bogotá, CO", segment: "Corporativo", scheduled: "24 Oct, 2026", status: "En tránsito", risk: "Medio", days: 4 },
  { id: "ORD-10480", customer: "Laura Martínez", destination: "Cali, CO", segment: "Consumidor", scheduled: "23 Oct, 2026", status: "Retrasado", risk: "Alto", days: 6 },
  { id: "ORD-10479", customer: "Andrés Torres", destination: "Barranquilla, CO", segment: "Pequeña empresa", scheduled: "23 Oct, 2026", status: "En tránsito", risk: "Alto", days: 5 },
  { id: "ORD-10478", customer: "Sofía Herrera", destination: "Cartagena, CO", segment: "Corporativo", scheduled: "22 Oct, 2026", status: "Entregado", risk: "Bajo", days: 2 },
  { id: "ORD-10477", customer: "Daniel Castro", destination: "Pereira, CO", segment: "Consumidor", scheduled: "22 Oct, 2026", status: "En tránsito", risk: "Medio", days: 3 },
  { id: "ORD-10476", customer: "Valentina Rojas", destination: "Manizales, CO", segment: "Pequeña empresa", scheduled: "21 Oct, 2026", status: "Retrasado", risk: "Alto", days: 7 },
];

const deliveryTrend = [
  { day: "Lun", onTime: 78, delayed: 22 },
  { day: "Mar", onTime: 84, delayed: 16 },
  { day: "Mié", onTime: 81, delayed: 19 },
  { day: "Jue", onTime: 89, delayed: 11 },
  { day: "Vie", onTime: 86, delayed: 14 },
  { day: "Sáb", onTime: 92, delayed: 8 },
  { day: "Dom", onTime: 88, delayed: 12 },
];

const riskDistribution = [
  { name: "Riesgo bajo", value: 58, color: "#34d399" },
  { name: "Riesgo medio", value: 27, color: "#fbbf24" },
  { name: "Riesgo alto", value: 15, color: "#fb7185" },
];

const navItems = [
  { label: "Dashboard", icon: LayoutDashboard },
  { label: "Pedidos", icon: Package },
  { label: "Entregas", icon: Truck },
  { label: "Clientes", icon: Users },
  { label: "Configuración", icon: Settings },
];

function StatusBadge({ status }: { status: Order["status"] }) {
  const styles = {
    Entregado: "bg-emerald-400/10 text-emerald-300",
    "En tránsito": "bg-sky-400/10 text-sky-300",
    Retrasado: "bg-rose-400/10 text-rose-300",
  };

  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${styles[status]}`}>
      <span className="h-1.5 w-1.5 rounded-full bg-current" />
      {status}
    </span>
  );
}

function RiskBadge({ risk }: { risk: Order["risk"] }) {
  const styles = {
    Bajo: "text-emerald-300 bg-emerald-400/10",
    Medio: "text-amber-300 bg-amber-400/10",
    Alto: "text-rose-300 bg-rose-400/10",
  };

  return (
    <span className={`rounded-md px-2 py-1 text-xs font-medium ${styles[risk]}`}>
      {risk}
    </span>
  );
}

export default function Home() {
  const [query, setQuery] = useState("");
  const [riskFilter, setRiskFilter] = useState("Todos");
  const [activeNav, setActiveNav] = useState("Dashboard");
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [notificationOpen, setNotificationOpen] = useState(false);

  const filteredOrders = useMemo(() => {
    return orders.filter((order) => {
      const matchesQuery =
        `${order.id} ${order.customer} ${order.destination}`
          .toLowerCase()
          .includes(query.toLowerCase());

      const matchesRisk = riskFilter === "Todos" || order.risk === riskFilter;

      return matchesQuery && matchesRisk;
    });
  }, [query, riskFilter]);

  function exportCSV() {
    const headers = ["Pedido", "Cliente", "Destino", "Estado", "Riesgo", "Dias estimados"];
    const rows = filteredOrders.map((order) => [
      order.id,
      order.customer,
      order.destination,
      order.status,
      order.risk,
      String(order.days),
    ]);

    const csv = [headers, ...rows]
      .map((row) => row.map((value) => `"${value.replaceAll('"', '""')}"`).join(","))
      .join("\n");

    const blob = new Blob(["\uFEFF" + csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "deliveryiq-pedidos.csv";
    link.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="min-h-screen bg-[#090d18] text-slate-100">
      <div className="flex min-h-screen">
        {mobileMenuOpen && (
          <button
            aria-label="Cerrar menú"
            onClick={() => setMobileMenuOpen(false)}
            className="fixed inset-0 z-40 bg-black/60 lg:hidden"
          />
        )}

        <aside className={`fixed inset-y-0 left-0 z-50 flex w-64 flex-col border-r border-white/[0.07] bg-[#0c1120] transition-transform lg:static lg:translate-x-0 ${mobileMenuOpen ? "translate-x-0" : "-translate-x-full"}`}>
          <div className="flex h-[76px] items-center gap-3 border-b border-white/[0.07] px-6">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-cyan-400 text-slate-950 shadow-lg shadow-cyan-500/20">
              <Activity size={22} strokeWidth={2.5} />
            </div>
            <div>
              <p className="text-lg font-bold tracking-tight">Delivery<span className="text-cyan-400">IQ</span></p>
              <p className="text-[10px] font-semibold uppercase tracking-[0.22em] text-slate-500">Logistics intelligence</p>
            </div>
            <button className="ml-auto text-slate-500 lg:hidden" onClick={() => setMobileMenuOpen(false)} aria-label="Cerrar navegación">
              <X size={19} />
            </button>
          </div>

          <div className="px-4 pt-7">
            <p className="mb-3 px-3 text-[10px] font-bold uppercase tracking-[0.2em] text-slate-600">Workspace</p>
            <nav className="space-y-1">
              {navItems.map((item) => {
                const Icon = item.icon;
                return (
                  <button
                    key={item.label}
                    onClick={() => {
                      setActiveNav(item.label);
                      setMobileMenuOpen(false);
                    }}
                    className={`flex w-full items-center gap-3 rounded-xl px-3 py-3 text-left text-sm transition ${activeNav === item.label ? "bg-cyan-400/10 font-semibold text-cyan-300 ring-1 ring-cyan-400/15" : "text-slate-400 hover:bg-white/[0.04] hover:text-white"}`}
                  >
                    <Icon size={18} />
                    {item.label}
                    {item.label === "Pedidos" && <span className="ml-auto rounded-md bg-white/[0.06] px-2 py-0.5 text-[10px] text-slate-400">248</span>}
                  </button>
                );
              })}
            </nav>
          </div>

          <div className="mt-8 px-4">
            <p className="mb-3 px-3 text-[10px] font-bold uppercase tracking-[0.2em] text-slate-600">Intelligence</p>
            <div className="rounded-xl border border-cyan-400/15 bg-gradient-to-br from-cyan-400/[0.08] to-blue-500/[0.03] p-4">
              <div className="mb-3 flex items-center gap-2">
                <Zap size={16} className="text-cyan-300" />
                <span className="text-xs font-semibold text-slate-200">Delivery AI</span>
                <span className="ml-auto rounded bg-cyan-400/10 px-1.5 py-0.5 text-[9px] font-bold text-cyan-300">BETA</span>
              </div>
              <p className="text-xs leading-5 text-slate-400">Predicción de retrasos basada en datos y aprendizaje automático.</p>
              <div className="mt-4 h-1.5 overflow-hidden rounded-full bg-white/[0.07]">
                <div className="h-full w-[72%] rounded-full bg-cyan-400" />
              </div>
              <p className="mt-2 text-[10px] text-slate-500">Módulo de demostración</p>
            </div>
          </div>

          <div className="mt-auto border-t border-white/[0.07] p-4">
            <div className="flex items-center gap-3 rounded-xl p-2">
              <div className="flex h-9 w-9 items-center justify-center rounded-full bg-gradient-to-br from-violet-400 to-cyan-400 text-xs font-bold text-slate-950">DA</div>
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium">Data Analyst</p>
                <p className="truncate text-xs text-slate-500">Administrador</p>
              </div>
              <button aria-label="Ayuda" className="text-slate-500 hover:text-white"><CircleHelp size={17} /></button>
            </div>
          </div>
        </aside>

        <main className="min-w-0 flex-1">
          <header className="sticky top-0 z-30 flex h-[76px] items-center justify-between border-b border-white/[0.07] bg-[#090d18]/90 px-4 backdrop-blur-xl sm:px-7 lg:px-10">
            <div className="flex items-center gap-3">
              <button onClick={() => setMobileMenuOpen(true)} className="text-slate-400 lg:hidden" aria-label="Abrir menú">
                <Menu size={22} />
              </button>
              <div>
                <div className="flex items-center gap-2 text-xs text-slate-500">
                  <span>Workspace</span><span>/</span><span className="text-slate-300">{activeNav}</span>
                </div>
                <h1 className="mt-1 text-lg font-semibold tracking-tight sm:text-xl">{activeNav}</h1>
              </div>
            </div>

            <div className="flex items-center gap-2 sm:gap-4">
              <div className="hidden items-center gap-2 rounded-lg border border-white/[0.08] bg-white/[0.02] px-3 py-2 text-xs text-slate-400 md:flex">
                <CalendarDays size={14} />
                <span>Oct 18 – Oct 24, 2026</span>
                <ChevronDown size={14} />
              </div>
              <div className="relative">
                <button
                  onClick={() => setNotificationOpen(!notificationOpen)}
                  aria-label="Notificaciones"
                  className="relative flex h-10 w-10 items-center justify-center rounded-xl border border-white/[0.08] text-slate-400 hover:bg-white/[0.04] hover:text-white"
                >
                  <Bell size={18} />
                  <span className="absolute right-2 top-2 h-2 w-2 rounded-full border-2 border-[#090d18] bg-rose-400" />
                </button>
                {notificationOpen && (
                  <div className="absolute right-0 top-12 z-50 w-72 rounded-xl border border-white/10 bg-[#111827] p-4 shadow-2xl">
                    <p className="font-semibold">Notificaciones</p>
                    <p className="mt-2 text-sm text-slate-400">Hay pedidos con riesgo alto que requieren revisión.</p>
                  </div>
                )}
              </div>
              <div className="hidden h-9 w-9 items-center justify-center rounded-full border border-white/10 bg-violet-400/10 text-xs font-bold text-violet-200 sm:flex">DA</div>
            </div>
          </header>

          <div className="mx-auto max-w-[1600px] space-y-7 p-4 sm:p-7 lg:p-10">
            <section className="flex flex-col justify-between gap-4 sm:flex-row sm:items-end">
              <div>
                <div className="mb-2 inline-flex items-center gap-2 rounded-full border border-emerald-400/15 bg-emerald-400/[0.06] px-3 py-1 text-[11px] font-medium text-emerald-300">
                  <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-400" />
                  Sistema operativo
                </div>
                <h2 className="text-2xl font-bold tracking-tight sm:text-3xl">Logistics overview<span className="text-cyan-400">.</span></h2>
                <p className="mt-2 max-w-xl text-sm leading-6 text-slate-400">Monitorea el rendimiento de las entregas, detecta riesgos y convierte datos logísticos en decisiones.</p>
              </div>
              <button onClick={exportCSV} className="inline-flex items-center justify-center gap-2 rounded-xl bg-cyan-400 px-4 py-2.5 text-sm font-bold text-slate-950 transition hover:bg-cyan-300">
                <Download size={16} /> Exportar CSV
              </button>
            </section>

            <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
              {[
                { title: "Total de pedidos", value: "12.486", change: "+12,8%", icon: Package, note: "vs. periodo anterior", positive: true },
                { title: "Entregas a tiempo", value: "86,4%", change: "+4,2%", icon: CheckCircle2, note: "vs. periodo anterior", positive: true },
                { title: "Pedidos retrasados", value: "1.204", change: "-2,4%", icon: Clock3, note: "vs. periodo anterior", positive: true },
                { title: "Riesgo alto", value: "15,2%", change: "+1,8%", icon: AlertTriangle, note: "requiere atención", positive: false },
              ].map((metric) => {
                const Icon = metric.icon;
                return (
                  <article key={metric.title} className="rounded-2xl border border-white/[0.07] bg-[#101626] p-5 transition hover:border-cyan-400/20">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-slate-400">{metric.title}</span>
                      <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-white/[0.04] text-cyan-300"><Icon size={18} /></div>
                    </div>
                    <div className="mt-5 flex items-end justify-between gap-2">
                      <p className="text-3xl font-bold tracking-tight">{metric.value}</p>
                      <span className={`mb-1 inline-flex items-center gap-1 text-xs font-semibold ${metric.positive ? "text-emerald-300" : "text-rose-300"}`}>
                        {metric.positive ? <ArrowUpRight size={14} /> : <ArrowDownRight size={14} />}
                        {metric.change}
                      </span>
                    </div>
                    <p className="mt-2 text-xs text-slate-500">{metric.note}</p>
                  </article>
                );
              })}
            </section>

            <section className="grid gap-5 xl:grid-cols-[1.65fr_1fr]">
              <article className="rounded-2xl border border-white/[0.07] bg-[#101626] p-5 sm:p-6">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <h3 className="font-semibold">Rendimiento de entregas</h3>
                    <p className="mt-1 text-xs text-slate-500">Porcentaje de entregas a tiempo durante la semana</p>
                  </div>
                  <span className="rounded-lg border border-white/[0.08] px-3 py-1.5 text-xs text-slate-400">Últimos 7 días <ChevronDown className="ml-1 inline" size={13} /></span>
                </div>
                <div className="mt-7 h-[270px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={deliveryTrend} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
                      <defs>
                        <linearGradient id="onTimeGradient" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#22d3ee" stopOpacity={0.25} />
                          <stop offset="95%" stopColor="#22d3ee" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid stroke="#ffffff0b" vertical={false} />
                      <XAxis dataKey="day" axisLine={false} tickLine={false} tick={{ fill: "#64748b", fontSize: 12 }} dy={10} />
                      <YAxis axisLine={false} tickLine={false} tick={{ fill: "#64748b", fontSize: 12 }} domain={[0, 100]} tickFormatter={(value) => `${value}%`} />
                      <Tooltip contentStyle={{ background: "#111827", border: "1px solid #263244", borderRadius: 12, color: "#e2e8f0" }} />
                      <Area type="monotone" dataKey="onTime" name="A tiempo" stroke="#22d3ee" strokeWidth={2.5} fill="url(#onTimeGradient)" />
                      <Area type="monotone" dataKey="delayed" name="Retrasadas" stroke="#fb7185" strokeWidth={2} fill="transparent" strokeDasharray="5 5" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
                <div className="mt-3 flex flex-wrap gap-5 text-xs text-slate-400">
                  <span className="flex items-center gap-2"><span className="h-2 w-2 rounded-full bg-cyan-400" />Entregas a tiempo</span>
                  <span className="flex items-center gap-2"><span className="h-2 w-2 rounded-full bg-rose-400" />Entregas retrasadas</span>
                </div>
              </article>

              <article className="rounded-2xl border border-white/[0.07] bg-[#101626] p-5 sm:p-6">
                <div>
                  <h3 className="font-semibold">Distribución del riesgo</h3>
                  <p className="mt-1 text-xs text-slate-500">Clasificación de pedidos por nivel de riesgo</p>
                </div>
                <div className="relative mt-3 h-[235px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie data={riskDistribution} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={66} outerRadius={91} paddingAngle={4} stroke="none">
                        {riskDistribution.map((entry) => <Cell key={entry.name} fill={entry.color} />)}
                      </Pie>
                      <Tooltip contentStyle={{ background: "#111827", border: "1px solid #263244", borderRadius: 12, color: "#e2e8f0" }} formatter={(value) => [`${value}%`, "Pedidos"]} />
                    </PieChart>
                  </ResponsiveContainer>
                  <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
                    <span className="text-3xl font-bold">100%</span>
                    <span className="mt-1 text-xs text-slate-500">Pedidos</span>
                  </div>
                </div>
                <div className="space-y-3">
                  {riskDistribution.map((risk) => (
                    <div key={risk.name} className="flex items-center justify-between text-sm">
                      <span className="flex items-center gap-2 text-slate-400"><span className="h-2.5 w-2.5 rounded-full" style={{ background: risk.color }} />{risk.name}</span>
                      <span className="font-semibold">{risk.value}%</span>
                    </div>
                  ))}
                </div>
              </article>
            </section>

            <section className="overflow-hidden rounded-2xl border border-white/[0.07] bg-[#101626]">
              <div className="flex flex-col justify-between gap-4 border-b border-white/[0.07] p-5 sm:flex-row sm:items-center sm:p-6">
                <div>
                  <h3 className="font-semibold">Pedidos recientes</h3>
                  <p className="mt-1 text-xs text-slate-500">Consulta los pedidos y revisa sus niveles de riesgo.</p>
                </div>
                <div className="flex flex-col gap-2 sm:flex-row">
                  <div className="relative">
                    <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
                    <input
                      value={query}
                      onChange={(event) => setQuery(event.target.value)}
                      placeholder="Buscar pedido o cliente..."
                      className="w-full rounded-xl border border-white/[0.08] bg-[#0b1020] py-2.5 pl-9 pr-3 text-sm outline-none placeholder:text-slate-600 focus:border-cyan-400/50 sm:w-64"
                    />
                  </div>
                  <select
                    value={riskFilter}
                    onChange={(event) => setRiskFilter(event.target.value)}
                    aria-label="Filtrar por riesgo"
                    className="rounded-xl border border-white/[0.08] bg-[#0b1020] px-3 py-2.5 text-sm text-slate-300 outline-none focus:border-cyan-400/50"
                  >
                    <option>Todos</option>
                    <option>Bajo</option>
                    <option>Medio</option>
                    <option>Alto</option>
                  </select>
                </div>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full min-w-[850px] text-left text-sm">
                  <thead className="bg-white/[0.015] text-[11px] uppercase tracking-wider text-slate-500">
                    <tr>
                      <th className="px-6 py-4 font-semibold">Pedido / Cliente</th>
                      <th className="px-6 py-4 font-semibold">Destino</th>
                      <th className="px-6 py-4 font-semibold">Fecha programada</th>
                      <th className="px-6 py-4 font-semibold">Estado</th>
                      <th className="px-6 py-4 font-semibold">Riesgo IA</th>
                      <th className="px-6 py-4 font-semibold">Días estimados</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/[0.045]">
                    {filteredOrders.map((order) => (
                      <tr key={order.id} className="transition hover:bg-white/[0.02]">
                        <td className="px-6 py-4">
                          <p className="font-semibold text-slate-200">{order.id}</p>
                          <p className="mt-1 text-xs text-slate-500">{order.customer}</p>
                        </td>
                        <td className="px-6 py-4 text-slate-400">{order.destination}</td>
                        <td className="px-6 py-4 text-slate-400">{order.scheduled}</td>
                        <td className="px-6 py-4"><StatusBadge status={order.status} /></td>
                        <td className="px-6 py-4"><RiskBadge risk={order.risk} /></td>
                        <td className="px-6 py-4 text-slate-300">{order.days} días</td>
                      </tr>
                    ))}
                    {filteredOrders.length === 0 && (
                      <tr><td colSpan={6} className="px-6 py-12 text-center text-slate-500">No se encontraron pedidos con esos filtros.</td></tr>
                    )}
                  </tbody>
                </table>
              </div>

              <div className="flex flex-col gap-2 border-t border-white/[0.07] px-6 py-4 text-xs text-slate-500 sm:flex-row sm:items-center sm:justify-between">
                <span>Mostrando {filteredOrders.length} de {orders.length} pedidos de demostración</span>
                <button onClick={exportCSV} className="inline-flex items-center gap-2 self-start font-semibold text-cyan-300 hover:text-cyan-200 sm:self-auto"><Download size={14} /> Descargar resultados</button>
              </div>
            </section>

            <section className="grid gap-4 md:grid-cols-2">
              <article className="flex items-start gap-4 rounded-2xl border border-rose-400/15 bg-rose-400/[0.035] p-5">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-rose-400/10 text-rose-300"><AlertTriangle size={20} /></div>
                <div>
                  <p className="font-semibold">Atención: pedidos con riesgo alto</p>
                  <p className="mt-1 text-sm leading-6 text-slate-400">Revisa las entregas marcadas con riesgo alto para priorizar el seguimiento y anticipar posibles retrasos.</p>
                  <button onClick={() => { setRiskFilter("Alto"); document.getElementById("pedidos")?.scrollIntoView({ behavior: "smooth" }); }} className="mt-3 text-xs font-bold text-rose-300 hover:text-rose-200">Revisar riesgo alto →</button>
                </div>
              </article>
              <article className="flex items-start gap-4 rounded-2xl border border-emerald-400/15 bg-emerald-400/[0.035] p-5">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-emerald-400/10 text-emerald-300"><ShieldCheck size={20} /></div>
                <div>
                  <p className="font-semibold">Centro de inteligencia</p>
                  <p className="mt-1 text-sm leading-6 text-slate-400">La siguiente fase conectará el panel con datos reales y un modelo de Machine Learning desarrollado en Python.</p>
                  <span className="mt-3 inline-flex items-center gap-1.5 text-xs font-semibold text-emerald-300"><Activity size={14} /> Backend pendiente de conexión</span>
                </div>
              </article>
            </section>

            <footer className="flex flex-col gap-2 border-t border-white/[0.06] pt-5 text-xs text-slate-600 sm:flex-row sm:items-center sm:justify-between">
              <p>© 2026 DeliveryIQ · Proyecto de analítica logística</p>
              <p className="flex items-center gap-2"><span className="h-1.5 w-1.5 rounded-full bg-emerald-400" /> Interfaz de demostración · Datos ficticios</p>
            </footer>
          </div>
        </main>
      </div>
    </div>
  );
}