import React, { useState, useMemo, useCallback } from "react";
import {
  Layers, Pin, PinOff, Play, ChevronRight, Info, Database,
  Eye, GitCompare, Sparkles, AlertTriangle,
} from "lucide-react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  ReferenceArea, CartesianGrid,
} from "recharts";

/* ============================================================================
   miStudio — J-Space Readout Panel  (demo / fixture data)

   WIRE FORMAT: the structures below mirror Neuronpedia's lens stream exactly
   (LensMetaMessage / LensTokenMessage / LensTypeSlice from
   apps/webapp/lib/utils/lens.ts). Swapping this demo onto a live miLLM or
   Neuronpedia inference stream means replacing buildFixture() with the socket
   handler — no component changes.

     meta   = { model, types, layers_by_type, top_n, prompt_len }
     token  = { position, token, is_generated, results: LensTypeSlice[] }
     slice  = { type: 'JACOBIAN_LENS'|'LOGIT_LENS', top_tokens[layer][k],
                top_probs[layer][k] }

   All readouts here are SYNTHETIC. See the provenance strip at the foot of the
   panel — it is wired to the same fields BR-007 requires on a real artifact.
   ========================================================================== */

const LAYERS = Array.from({ length: 21 }, (_, i) => i * 5); // reindexed 0–100
const BAND = { workspaceStart: 40, motorStart: 90 };
const TOP_N = 8;

const NOISE = [
  "Biserica", "Freguesias", "ambamb", "prid", "фев", "хол", "\u23CE\u23CE",
  "pecul", "fric", "ної", "Dom", "welsh", "niet", "storing", "ijij", "१",
  "moz", "carson", "hust", "barnes", "\u0645\u0627", "gaz", "yor", "silon",
];

/* ---------------------------------------------------------------- fixtures */
// Each concept: token, the positions it is active over, the layer it peaks at,
// how broadly it spreads, and its strength. This is the *shape* the paper
// describes — concepts absent early, rising through the workspace band in
// computation order, giving way to the imminent output in the motor layers.

const FIXTURES = {
  multihop: {
    id: "multihop",
    label: "Multi-hop recall",
    note: "The bridging entity never appears in the prompt or the output.",
    tokens: ["Fact", ":", " The", " number", " of", " legs", " on", " the",
      " animal", " that", " spins", " webs", " is"],
    completion: " 8",
    concepts: [
      { t: "spider", from: 8, to: 12, peak: 65, spread: 16, s: 1.0 },
      { t: "spiders", from: 9, to: 12, peak: 63, spread: 14, s: 0.72 },
      { t: "web", from: 10, to: 12, peak: 52, spread: 15, s: 0.66 },
      { t: "arachn", from: 9, to: 12, peak: 60, spread: 11, s: 0.5 },
      { t: "legs", from: 5, to: 12, peak: 80, spread: 13, s: 0.92 },
      { t: "eight", from: 11, to: 12, peak: 86, spread: 10, s: 0.86 },
      { t: "count", from: 3, to: 6, peak: 55, spread: 16, s: 0.55 },
      { t: "insect", from: 8, to: 11, peak: 58, spread: 12, s: 0.42 },
    ],
    motor: { 12: "8", 11: " is", 10: " webs", 9: " spins", 8: " that" },
  },
  arithmetic: {
    id: "arithmetic",
    label: "Mental arithmetic",
    note: "Intermediates surface in the order the computation requires.",
    tokens: ["calc", ":", " (", " 4", " +", " 17", " )", " *", " 2", " +",
      " 7", " ="],
    completion: " 49",
    concepts: [
      { t: "arithmetic", from: 0, to: 11, peak: 48, spread: 14, s: 0.7 },
      { t: "Math", from: 0, to: 11, peak: 52, spread: 15, s: 0.62 },
      { t: "21", from: 6, to: 11, peak: 66, spread: 11, s: 0.95 },
      { t: "42", from: 8, to: 11, peak: 76, spread: 10, s: 0.9 },
      { t: "49", from: 10, to: 11, peak: 88, spread: 9, s: 1.0 },
      { t: "equals", from: 9, to: 11, peak: 72, spread: 13, s: 0.5 },
      { t: "answer", from: 9, to: 11, peak: 70, spread: 12, s: 0.46 },
    ],
    motor: { 11: "49", 10: " =", 9: " 7", 8: " +" },
  },
  modulation: {
    id: "modulation",
    label: "Directed modulation",
    note: "Held concept and the act of holding it, while the surface text is unrelated.",
    tokens: ["Write", " \"The", " old", " painting", " hung", " cro", "ok",
      "edly", " on", " the", " wall", ".\"", " Concentrate", " on", " citrus",
      " fruits", "."],
    completion: " The old painting hung crookedly…",
    concepts: [
      { t: "orange", from: 4, to: 16, peak: 74, spread: 16, s: 1.0 },
      { t: "lemon", from: 5, to: 16, peak: 72, spread: 13, s: 0.66 },
      { t: "fruit", from: 3, to: 16, peak: 56, spread: 15, s: 0.78 },
      { t: "citrus", from: 13, to: 16, peak: 60, spread: 12, s: 0.7 },
      { t: "thinking", from: 4, to: 16, peak: 62, spread: 18, s: 0.6 },
      { t: "imagine", from: 4, to: 14, peak: 58, spread: 14, s: 0.5 },
      { t: "focused", from: 6, to: 16, peak: 64, spread: 12, s: 0.44 },
    ],
    motor: { 5: "ok", 6: "edly", 7: " on", 8: " the", 9: " wall", 4: " cro" },
  },
};

/* ------------------------------------------------------- fixture generator */
function gauss(x, mu, sigma) {
  const d = (x - mu) / sigma;
  return Math.exp(-0.5 * d * d);
}
// Deterministic hash → [0,1); keeps the demo stable across renders.
function rnd(seed) {
  const x = Math.sin(seed * 12.9898) * 43758.5453;
  return x - Math.floor(x);
}

function scoreAt(fx, pos, layer, lensType) {
  const out = [];
  const degrade = lensType === "LOGIT_LENS";

  for (const c of fx.concepts) {
    if (pos < c.from || pos > c.to) continue;
    let v = c.s * gauss(layer, c.peak, c.spread);
    // The logit lens recovers the same content in later layers but degrades
    // earlier — the paper's central comparison, made visible in DIFF mode.
    if (degrade) v *= layer < 70 ? Math.max(0, (layer - 30) / 60) : 0.94;
    if (v > 0.02) out.push({ token: c.t, score: v });
  }

  // Motor band: the imminent output takes over.
  const nt = fx.motor[pos];
  if (nt) {
    const m = gauss(layer, 100, 9) * 1.25;
    if (m > 0.02) out.push({ token: nt, score: m });
  }

  // Sensory band: uninterpretable readouts, not a null result.
  const noiseW = Math.max(0, 1 - Math.max(0, layer - 12) / 26);
  const nNoise = noiseW > 0.05 ? 6 : 3;
  for (let i = 0; i < nNoise; i++) {
    const s = rnd(pos * 131 + layer * 17 + i * 7 + (degrade ? 991 : 0));
    out.push({
      token: NOISE[Math.floor(s * NOISE.length)],
      score: 0.05 + s * 0.28 * (0.25 + noiseW),
    });
  }

  out.sort((a, b) => b.score - a.score);
  const top = out.slice(0, TOP_N);
  const sum = top.reduce((a, b) => a + b.score, 0) || 1;
  return {
    top_tokens: top.map((t) => t.token),
    top_probs: top.map((t) => t.score / sum),
  };
}

function buildFixture(fx) {
  const types = ["JACOBIAN_LENS", "LOGIT_LENS"];
  const tokens = fx.tokens.map((tok, pos) => ({
    kind: "token",
    position: pos,
    token: tok,
    id: 1000 + pos,
    is_generated: false,
    results: types.map((type) => {
      const per = LAYERS.map((L) => scoreAt(fx, pos, L, type));
      return {
        type,
        top_tokens: per.map((p) => p.top_tokens),
        top_probs: per.map((p) => p.top_probs),
      };
    }),
  }));
  return {
    meta: {
      kind: "meta",
      model: "gemma-2-2b",
      types,
      layers_by_type: { JACOBIAN_LENS: LAYERS, LOGIT_LENS: LAYERS },
      top_n: TOP_N,
      prompt_len: tokens.length,
    },
    tokens,
  };
}

/* ------------------------------------------------------------------ helpers */
const bandOf = (L) =>
  L < BAND.workspaceStart ? "sensory" : L < BAND.motorStart ? "workspace" : "motor";

const BAND_META = {
  sensory: { label: "Sensory", hint: "low-level parsing; readouts expected to be uninterpretable" },
  workspace: { label: "Workspace", hint: "reportable, reusable content" },
  motor: { label: "Motor", hint: "committing to the next token" },
};

function rankOf(slice, layerIdx, token) {
  const row = slice.top_tokens[layerIdx];
  const i = row.indexOf(token);
  return i === -1 ? null : i + 1;
}

// Emerald ramp keyed to rank; miStudio's accent, not a rainbow scale.
function rankColor(rank) {
  if (rank == null) return "transparent";
  const a = Math.max(0.06, 1 - Math.log(rank) / Math.log(TOP_N + 6));
  return `rgba(52, 211, 153, ${a.toFixed(3)})`;
}

const displayTok = (t) =>
  t.replace(/^ /, "\u00B7").replace(/\n/g, "\u23CE").replace(/\t/g, "\u2192");

/* ============================================================== main panel */
export default function JSpacePanel() {
  const [fixtureId, setFixtureId] = useState("multihop");
  const [lensMode, setLensMode] = useState("JACOBIAN_LENS"); // | LOGIT_LENS | DIFF
  const [selPos, setSelPos] = useState(11);
  const [selLayer, setSelLayer] = useState(65);
  const [pinned, setPinned] = useState(["spider", "legs", "eight"]);
  const [hover, setHover] = useState(null);

  const fx = FIXTURES[fixtureId];
  const data = useMemo(() => buildFixture(fx), [fx]);

  const sliceFor = useCallback(
    (tokenMsg, type) => tokenMsg.results.find((r) => r.type === type),
    []
  );

  const readLens = lensMode === "DIFF" ? "JACOBIAN_LENS" : lensMode;
  const layerIdx = LAYERS.indexOf(selLayer);

  const pick = (fixId) => {
    const f = FIXTURES[fixId];
    setFixtureId(fixId);
    setSelPos(f.tokens.length - 1);
    setPinned(f.concepts.slice(0, 3).map((c) => c.t));
    setHover(null);
  };

  const togglePin = (t) =>
    setPinned((p) => (p.includes(t) ? p.filter((x) => x !== t) : [...p, t].slice(0, 6)));

  /* trajectory data for the pinned tokens at the selected position */
  const traj = useMemo(() => {
    const tk = data.tokens[selPos];
    if (!tk) return [];
    const s = sliceFor(tk, readLens);
    return LAYERS.map((L, i) => {
      const row = { layer: L };
      pinned.forEach((p) => {
        const r = rankOf(s, i, p);
        row[p] = r == null ? null : r;
      });
      return row;
    });
  }, [data, selPos, pinned, readLens, sliceFor]);

  const PIN_COLORS = ["#34d399", "#60a5fa", "#f59e0b", "#f472b6", "#a78bfa", "#2dd4bf"];

  return (
    <div className="min-h-screen w-full bg-slate-900 text-slate-200 font-sans">
      <div className="mx-auto max-w-[1400px] px-5 py-5">

        {/* ---------------------------------------------------------- header */}
        <header className="mb-4 flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2">
            <Layers className="h-5 w-5 text-emerald-400" />
            <h1 className="text-lg font-semibold tracking-tight text-slate-100">
              J-Space Readout
            </h1>
          </div>
          <span className="rounded border border-slate-700 bg-slate-800 px-2 py-0.5 font-mono text-xs text-slate-300">
            {data.meta.model}
          </span>
          <span className="rounded border border-emerald-800 bg-emerald-950 px-2 py-0.5 text-xs text-emerald-300">
            fixture data
          </span>

          <div className="ml-auto flex items-center gap-2">
            {[
              { id: "JACOBIAN_LENS", label: "Jacobian", Icon: Sparkles },
              { id: "LOGIT_LENS", label: "Logit", Icon: Eye },
              { id: "DIFF", label: "Diff", Icon: GitCompare },
            ].map(({ id, label, Icon }) => (
              <button
                key={id}
                onClick={() => setLensMode(id)}
                className={`flex items-center gap-1.5 rounded border px-2.5 py-1 text-xs font-medium transition ${
                  lensMode === id
                    ? "border-emerald-600 bg-emerald-900 text-emerald-200"
                    : "border-slate-700 bg-slate-800 text-slate-400 hover:text-slate-200"
                }`}
              >
                <Icon className="h-3.5 w-3.5" />
                {label}
              </button>
            ))}
          </div>
        </header>

        {/* ------------------------------------------------------- fixtures */}
        <div className="mb-4 flex flex-wrap items-center gap-2">
          {Object.values(FIXTURES).map((f) => (
            <button
              key={f.id}
              onClick={() => pick(f.id)}
              className={`flex items-center gap-1.5 rounded border px-3 py-1.5 text-xs transition ${
                fixtureId === f.id
                  ? "border-slate-500 bg-slate-700 text-slate-100"
                  : "border-slate-700 bg-slate-800 text-slate-400 hover:text-slate-200"
              }`}
            >
              <Play className="h-3 w-3" />
              {f.label}
            </button>
          ))}
          <p className="ml-1 text-xs text-slate-500">{fx.note}</p>
        </div>

        <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1fr_320px]">
          {/* ============================================ left: grid + prompt */}
          <div className="min-w-0 space-y-4">

            {/* prompt strip */}
            <section className="rounded-lg border border-slate-700 bg-slate-800 p-3">
              <div className="mb-2 flex items-center gap-2 text-xs font-medium text-slate-400">
                Prompt
                <span className="text-slate-600">·</span>
                <span className="font-normal text-slate-500">click a token to inspect its position</span>
              </div>
              <div className="flex flex-wrap gap-1">
                {data.tokens.map((t) => (
                  <button
                    key={t.position}
                    onClick={() => setSelPos(t.position)}
                    className={`rounded border px-1.5 py-1 font-mono text-xs transition ${
                      selPos === t.position
                        ? "border-emerald-500 bg-emerald-900 text-emerald-100"
                        : "border-slate-700 bg-slate-900 text-slate-300 hover:border-slate-500"
                    }`}
                  >
                    {displayTok(t.token)}
                  </button>
                ))}
                <span className="ml-2 flex items-center gap-1 font-mono text-xs text-slate-500">
                  <ChevronRight className="h-3 w-3" />
                  {fx.completion}
                </span>
              </div>
            </section>

            {/* ------------------------------------------------- signature grid */}
            <section className="rounded-lg border border-slate-700 bg-slate-800 p-3">
              <div className="mb-2 flex items-center justify-between">
                <div className="text-xs font-medium text-slate-400">
                  {pinned.length
                    ? `Rank of pinned tokens \u00B7 layer \u00D7 position`
                    : `Top readout \u00B7 layer \u00D7 position`}
                </div>
                <div className="flex items-center gap-3 text-[10px] text-slate-500">
                  {Object.entries(BAND_META).map(([k, v]) => (
                    <span key={k} className="flex items-center gap-1">
                      <span
                        className={`inline-block h-2 w-2 rounded-sm ${
                          k === "workspace"
                            ? "bg-emerald-500"
                            : k === "motor"
                            ? "bg-amber-500"
                            : "bg-slate-600"
                        }`}
                      />
                      {v.label}
                    </span>
                  ))}
                </div>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full border-separate border-spacing-0">
                  <tbody>
                    {[...LAYERS].reverse().map((L) => {
                      const i = LAYERS.indexOf(L);
                      const band = bandOf(L);
                      return (
                        <tr key={L}>
                          <td
                            onClick={() => setSelLayer(L)}
                            className={`sticky left-0 z-10 cursor-pointer border-r bg-slate-800 pr-2 text-right font-mono text-[10px] ${
                              selLayer === L
                                ? "border-emerald-500 text-emerald-300"
                                : band === "workspace"
                                ? "border-slate-600 text-slate-400"
                                : "border-slate-700 text-slate-600"
                            }`}
                          >
                            L{L}
                          </td>
                          {data.tokens.map((tk) => {
                            const s = sliceFor(tk, readLens);
                            const sL = sliceFor(tk, "LOGIT_LENS");
                            const sJ = sliceFor(tk, "JACOBIAN_LENS");

                            let cellTok = s.top_tokens[i][0];
                            let bg = "transparent";
                            let dim = false;

                            if (pinned.length) {
                              // heatmap: best rank among pinned tokens
                              let best = null, which = null;
                              pinned.forEach((p) => {
                                const r = rankOf(s, i, p);
                                if (r != null && (best == null || r < best)) {
                                  best = r; which = p;
                                }
                              });
                              bg = rankColor(best);
                              cellTok = which ?? "";
                              dim = best == null;
                            } else if (lensMode === "DIFF") {
                              const jr = sJ.top_tokens[i][0];
                              const agree = sL.top_tokens[i][0] === jr;
                              bg = agree ? "rgba(100,116,139,.18)" : "rgba(245,158,11,.28)";
                            } else {
                              bg = band === "workspace"
                                ? "rgba(52,211,153,.10)"
                                : "rgba(100,116,139,.10)";
                            }

                            const isSel = selPos === tk.position && selLayer === L;
                            return (
                              <td
                                key={tk.position}
                                onMouseEnter={() =>
                                  setHover({ pos: tk.position, layerIdx: i, layer: L })
                                }
                                onMouseLeave={() => setHover(null)}
                                onClick={() => { setSelPos(tk.position); setSelLayer(L); }}
                                style={{ background: bg }}
                                className={`cursor-pointer border-b border-r px-1 py-[3px] text-center font-mono text-[10px] leading-tight ${
                                  isSel ? "border-emerald-400" : "border-slate-800"
                                } ${dim ? "text-slate-700" : "text-slate-200"}`}
                                title={`L${L} \u00B7 pos ${tk.position}`}
                              >
                                <span className="block max-w-[64px] truncate">
                                  {displayTok(cellTok)}
                                </span>
                              </td>
                            );
                          })}
                        </tr>
                      );
                    })}
                    <tr>
                      <td className="sticky left-0 bg-slate-800" />
                      {data.tokens.map((tk) => (
                        <td
                          key={tk.position}
                          className="max-w-[64px] truncate px-1 pt-1 text-center font-mono text-[10px] text-slate-500"
                        >
                          {displayTok(tk.token)}
                        </td>
                      ))}
                    </tr>
                  </tbody>
                </table>
              </div>

              {/* hover detail */}
              <div className="mt-2 min-h-[54px] rounded border border-slate-700 bg-slate-900 p-2">
                {hover ? (
                  <>
                    <div className="mb-1 font-mono text-[10px] text-slate-500">
                      L{hover.layer} · pos {hover.pos} ·{" "}
                      <span className="text-slate-400">
                        {BAND_META[bandOf(hover.layer)].label}
                      </span>
                    </div>
                    <div className="flex flex-wrap gap-1">
                      {sliceFor(data.tokens[hover.pos], readLens).top_tokens[
                        hover.layerIdx
                      ].map((t, k) => (
                        <button
                          key={k}
                          onClick={() => togglePin(t)}
                          className={`rounded px-1.5 py-0.5 font-mono text-[10px] ${
                            pinned.includes(t)
                              ? "bg-emerald-800 text-emerald-100"
                              : "bg-slate-800 text-slate-300 hover:bg-slate-700"
                          }`}
                        >
                          {displayTok(t)}
                        </button>
                      ))}
                    </div>
                  </>
                ) : (
                  <p className="text-[11px] text-slate-600">
                    Hover a cell for its full top-{TOP_N} readout. Click a readout token to pin it.
                  </p>
                )}
              </div>
            </section>

            {/* -------------------------------------------------- trajectory */}
            {pinned.length > 0 && (
              <section className="rounded-lg border border-slate-700 bg-slate-800 p-3">
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-xs font-medium text-slate-400">
                    Rank across layers · position {selPos}
                  </span>
                  <span className="text-[10px] text-slate-500">lower is stronger</span>
                </div>
                <div className="h-52">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={traj} margin={{ top: 4, right: 8, bottom: 4, left: -18 }}>
                      <CartesianGrid stroke="#334155" strokeDasharray="2 4" />
                      <ReferenceArea
                        x1={BAND.workspaceStart}
                        x2={BAND.motorStart}
                        fill="#34d399"
                        fillOpacity={0.07}
                      />
                      <XAxis
                        dataKey="layer"
                        stroke="#64748b"
                        tick={{ fontSize: 10 }}
                        label={{ value: "layer (reindexed)", position: "insideBottom",
                                 offset: -2, fill: "#64748b", fontSize: 10 }}
                      />
                      <YAxis
                        reversed
                        domain={[1, TOP_N]}
                        stroke="#64748b"
                        tick={{ fontSize: 10 }}
                        allowDecimals={false}
                      />
                      <Tooltip
                        contentStyle={{
                          background: "#0f172a", border: "1px solid #334155",
                          borderRadius: 6, fontSize: 11,
                        }}
                        labelStyle={{ color: "#94a3b8" }}
                      />
                      {pinned.map((p, i) => (
                        <Line
                          key={p}
                          type="monotone"
                          dataKey={p}
                          stroke={PIN_COLORS[i % PIN_COLORS.length]}
                          strokeWidth={2}
                          dot={false}
                          connectNulls={false}
                          isAnimationActive={false}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </section>
            )}
          </div>

          {/* ================================================= right rail */}
          <div className="space-y-4">

            {/* pinned */}
            <section className="rounded-lg border border-slate-700 bg-slate-800 p-3">
              <div className="mb-2 flex items-center gap-1.5 text-xs font-medium text-slate-400">
                <Pin className="h-3.5 w-3.5" /> Pinned tokens
              </div>
              {pinned.length === 0 ? (
                <p className="text-[11px] text-slate-600">
                  Nothing pinned. Pin a token to turn the grid into a rank heatmap.
                </p>
              ) : (
                <div className="flex flex-wrap gap-1.5">
                  {pinned.map((p, i) => (
                    <button
                      key={p}
                      onClick={() => togglePin(p)}
                      className="group flex items-center gap-1 rounded border border-slate-600 bg-slate-900 px-2 py-1 font-mono text-[11px] text-slate-200 hover:border-slate-500"
                    >
                      <span
                        className="inline-block h-2 w-2 rounded-sm"
                        style={{ background: PIN_COLORS[i % PIN_COLORS.length] }}
                      />
                      {p}
                      <PinOff className="h-3 w-3 text-slate-600 group-hover:text-slate-300" />
                    </button>
                  ))}
                </div>
              )}
            </section>

            {/* by layer at selected position */}
            <section className="rounded-lg border border-slate-700 bg-slate-800 p-3">
              <div className="mb-2 text-xs font-medium text-slate-400">
                By layer · position {selPos}{" "}
                <span className="font-mono text-slate-500">
                  {displayTok(data.tokens[selPos]?.token ?? "")}
                </span>
              </div>
              <div className="max-h-72 space-y-0.5 overflow-y-auto pr-1">
                {[...LAYERS].reverse().map((L) => {
                  const i = LAYERS.indexOf(L);
                  const s = sliceFor(data.tokens[selPos], readLens);
                  const band = bandOf(L);
                  return (
                    <button
                      key={L}
                      onClick={() => setSelLayer(L)}
                      className={`flex w-full items-start gap-2 rounded px-1.5 py-1 text-left ${
                        selLayer === L ? "bg-slate-700" : "hover:bg-slate-700"
                      }`}
                    >
                      <span
                        className={`mt-px w-7 shrink-0 font-mono text-[10px] ${
                          band === "workspace" ? "text-emerald-400"
                          : band === "motor" ? "text-amber-500" : "text-slate-600"
                        }`}
                      >
                        L{L}
                      </span>
                      <span className="flex flex-wrap gap-1">
                        {s.top_tokens[i].slice(0, 4).map((t, k) => (
                          <span
                            key={k}
                            className={`font-mono text-[10px] ${
                              pinned.includes(t)
                                ? "rounded bg-emerald-900 px-1 text-emerald-200"
                                : k === 0 ? "text-slate-200" : "text-slate-500"
                            }`}
                          >
                            {displayTok(t)}
                          </span>
                        ))}
                      </span>
                    </button>
                  );
                })}
              </div>
            </section>

            {/* evidence rung */}
            <section className="rounded-lg border border-slate-700 bg-slate-800 p-3">
              <div className="mb-1.5 flex items-center gap-1.5 text-xs font-medium text-slate-400">
                <AlertTriangle className="h-3.5 w-3.5 text-amber-500" /> Evidence rung
              </div>
              <div className="rounded border border-amber-900 bg-amber-950 px-2 py-1.5">
                <div className="text-[11px] font-medium text-amber-200">Rung 0 · Readout</div>
                <p className="mt-0.5 text-[10px] leading-snug text-amber-200/70">
                  A concept appearing in a readout is not a causal claim. Run a
                  coordinate swap with a matched control to raise the rung.
                </p>
              </div>
            </section>
          </div>
        </div>

        {/* -------------------------------------------------- provenance strip */}
        <footer className="mt-4 rounded-lg border border-slate-700 bg-slate-800 px-3 py-2">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 font-mono text-[10px] text-slate-500">
            <span className="flex items-center gap-1 text-slate-400">
              <Database className="h-3 w-3" /> artifact
            </span>
            <span>gemma-2-2b_jacobian_lens.pt</span>
            <span>target=penultimate</span>
            <span>qk=full</span>
            <span>positions=all-subsequent</span>
            <span>agg=mean</span>
            <span>corpus=wikitext-103-raw-v1</span>
            <span>n_prompts=1000</span>
            <span>seq_len=128</span>
            <span>dtype=fp16</span>
            <span className="text-emerald-500">band L40–L90 (rederived)</span>
          </div>
          <div className="mt-1 flex items-start gap-1.5 text-[10px] leading-snug text-slate-600">
            <Info className="mt-px h-3 w-3 shrink-0" />
            <span>
              Readouts are limited to concepts with single-token names, and some
              workspace-layer readouts resist interpretation. Absence of a signal
              is not evidence that the underlying computation did not occur.
            </span>
          </div>
        </footer>
      </div>
    </div>
  );
}
