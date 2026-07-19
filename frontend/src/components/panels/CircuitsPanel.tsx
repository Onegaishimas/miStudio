/**
 * Circuits panel (Feature 018) — the review/promotion surface for
 * cross-layer circuits. Every circuit and edge shows its evidence rung via
 * SERVER-rendered language (IDL-35); promotion is a badge, not a gate (and
 * reversible). List rows are slim summaries; full evidence loads on expand.
 * Discovery/validation tabs (016/017) join this panel as they land.
 */

import { useCallback, useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  GitBranch, ArrowUpRight, ArrowDownRight, Trash2, Download, Layers,
  Pencil, Check, X as XIcon, FileDown, Upload,
} from 'lucide-react';
import { useRef } from 'react';
import { circuitsApi } from '../../api/circuits';
import type { Circuit, CircuitSummary, CircuitEdge } from '../../types/circuits';
import { RungChip } from '../circuits/RungChip';
import { COMPONENTS } from '../../config/brand';

function EdgeRow({ edge }: { edge: CircuitEdge }) {
  const label = (n: CircuitEdge['up']) =>
    n.kind === 'cluster' ? `cluster:${n.cluster_profile_id}` : `#${n.feature_idx}`;
  return (
    <tr className={`text-xs ${edge.type === 'persistence' ? 'opacity-60' : ''}`}>
      <td className="py-1 pr-3 font-mono text-slate-300 whitespace-nowrap">
        L{edge.up.layer} {label(edge.up)} → L{edge.down.layer} {label(edge.down)}
      </td>
      <td className="py-1 pr-3">
        <span className={`rounded px-1 py-0.5 text-[10px] ${
          edge.type === 'persistence' ? 'bg-amber-500/10 text-amber-300'
          : edge.type === 'attention_mediated' ? 'bg-violet-500/10 text-violet-300'
          : 'bg-slate-700/60 text-slate-300'}`}>
          {edge.type}
        </span>
      </td>
      <td className="py-1 pr-3 font-mono text-slate-400">R{edge.rung}</td>
      <td className="py-1 pr-3 text-slate-400">
        {[
          edge.coactivation?.pmi != null ? `PMI ${edge.coactivation.pmi.toFixed(2)}` : null,
          edge.coactivation?.support != null ? `n=${edge.coactivation.support}` : null,
          edge.attribution?.score != null ? `attr ${edge.attribution.score.toFixed(2)}` : null,
          edge.weight_prior != null ? `prior ${edge.weight_prior.toFixed(2)}` : null,
          edge.effect_size != null ? `ES ${edge.effect_size.toFixed(2)}` : null,
        ].filter(Boolean).join(' · ')}
      </td>
      <td className="py-1 text-slate-500">
        {edge.validation_manifest_ref && (
          <span className="font-mono text-[10px]" title="Validation manifest">
            {edge.validation_manifest_ref}
          </span>
        )}
      </td>
    </tr>
  );
}

function CircuitDetail({ id, onChanged }: { id: string; onChanged: () => void }) {
  const [circuit, setCircuit] = useState<Circuit | null>(null);
  const [error, setError] = useState<string | null>(null);       // load failures only
  const [saveError, setSaveError] = useState<string | null>(null); // keeps the draft alive (R2 B4)
  const [showPersistence, setShowPersistence] = useState(false);
  const [editing, setEditing] = useState(false);
  const [draftName, setDraftName] = useState('');
  const [draftNarrative, setDraftNarrative] = useState('');

  useEffect(() => {
    circuitsApi.get(id).then(setCircuit).catch((e) => setError(String(e.message ?? e)));
  }, [id]);

  if (error && !circuit) return <p className="text-xs text-red-300 mt-2">{error}</p>;
  if (!circuit) return <p className="text-xs text-slate-500 mt-2">Loading evidence…</p>;

  const edges = circuit.edges.filter((e) => showPersistence || e.type !== 'persistence');
  const hiddenCount = circuit.edges.length - edges.length;

  const saveEdit = () => {
    setSaveError(null);
    circuitsApi.update(circuit.id, { name: draftName, narrative: draftNarrative })
      .then((updated) => { setCircuit(updated); setEditing(false); onChanged(); })
      .catch((e) => setSaveError(String(e.message ?? e))); // draft survives (R2 B4)
  };

  return (
    <div className="mt-3 border-t border-slate-700/60 pt-3 space-y-3">
      {editing ? (
        <div className="space-y-2">
          <input
            className="w-full rounded bg-slate-800 border border-slate-600 px-2 py-1 text-sm text-slate-100"
            value={draftName}
            onChange={(e) => setDraftName(e.target.value)}
            aria-label="Circuit name"
          />
          <textarea
            className="w-full rounded bg-slate-800 border border-slate-600 px-2 py-1 text-sm text-slate-100"
            rows={4}
            value={draftNarrative}
            onChange={(e) => setDraftNarrative(e.target.value)}
            placeholder="Narrative (markdown)"
            aria-label="Circuit narrative"
          />
          {saveError && (
            <p className="text-xs text-red-300">{saveError}</p>
          )}
          <div className="flex gap-2">
            <button onClick={saveEdit}
              className="flex items-center gap-1 rounded bg-emerald-600 px-2 py-1 text-xs text-white">
              <Check className="w-3 h-3" /> Save
            </button>
            <button onClick={() => setEditing(false)}
              className="flex items-center gap-1 rounded bg-slate-700 px-2 py-1 text-xs text-slate-200">
              <XIcon className="w-3 h-3" /> Cancel
            </button>
          </div>
        </div>
      ) : (
        <div className="flex items-start gap-2">
          <div className="flex-1 min-w-0">
            {circuit.narrative ? (
              <div className="text-sm text-slate-400">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    // House dark-theme markdown mapping (no typography plugin — R2 B10)
                    p: ({ children }) => <p className="mb-2 leading-relaxed">{children}</p>,
                    strong: ({ children }) => <strong className="font-semibold text-slate-200">{children}</strong>,
                    ul: ({ children }) => <ul className="list-disc list-inside mb-2 space-y-0.5">{children}</ul>,
                    ol: ({ children }) => <ol className="list-decimal list-inside mb-2 space-y-0.5">{children}</ol>,
                    li: ({ children }) => <li className="text-slate-300">{children}</li>,
                    code: ({ children }) => <code className="px-1 py-0.5 rounded bg-slate-900 text-slate-200 font-mono text-xs">{children}</code>,
                    table: ({ children }) => (
                      <div className="my-2 overflow-x-auto">
                        <table className="text-xs border border-slate-700 border-collapse">{children}</table>
                      </div>
                    ),
                    th: ({ children }) => <th className="px-2 py-1 border border-slate-700 text-slate-200">{children}</th>,
                    td: ({ children }) => <td className="px-2 py-1 border border-slate-800 text-slate-300">{children}</td>,
                  }}
                >
                  {circuit.narrative}
                </ReactMarkdown>
              </div>
            ) : (
              <p className="text-xs text-slate-500 italic">No narrative yet.</p>
            )}
          </div>
          <button
            onClick={() => { setDraftName(circuit.name); setDraftNarrative(circuit.narrative ?? ''); setEditing(true); }}
            className="p-1 rounded hover:bg-white/10 text-slate-400 shrink-0"
            title="Edit name & narrative"
          >
            <Pencil className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      <div>
        <h4 className="text-xs font-medium text-slate-400 mb-1">Members by layer</h4>
        {[...new Set(circuit.members.map((m) => m.layer))].sort((a, b) => a - b).map((layer) => {
          const layerMembers = circuit.members.filter((m) => m.layer === layer);
          const count = layerMembers.reduce(
            (n, m) => n + (m.member_kind === 'cluster_ref' ? (m.expanded_members?.length ?? 0) : 1), 0);
          return (
            <div key={layer} className="text-xs text-slate-300 mb-1">
              <span className="font-mono text-slate-500 mr-2">L{layer}</span>
              <span className="text-slate-500 mr-2">({count}/20)</span>
              {layerMembers.map((m, i) => (
                <span key={i} className="mr-2">
                  {m.member_kind === 'cluster_ref'
                    ? `[cluster ${m.cluster_name ?? m.cluster_profile_id}]`
                    : `#${m.feature?.feature_idx}${m.feature?.label ? ` ${m.feature.label}` : ''}`}
                </span>
              ))}
            </div>
          );
        })}
      </div>

      {circuit.edges.length > 0 && (
        <div className="overflow-x-auto">
          <div className="flex items-center gap-3 mb-1">
            <h4 className="text-xs font-medium text-slate-400">Edges</h4>
            {hiddenCount > 0 && !showPersistence && (
              <button onClick={() => setShowPersistence(true)}
                className="text-[10px] text-amber-300/80 hover:text-amber-300">
                show {hiddenCount} persistence edge{hiddenCount !== 1 ? 's' : ''}
              </button>
            )}
            {showPersistence && (
              <button onClick={() => setShowPersistence(false)}
                className="text-[10px] text-slate-500 hover:text-slate-300">
                hide persistence
              </button>
            )}
          </div>
          <table className="w-full">
            <tbody>{edges.map((e, i) => <EdgeRow key={i} edge={e} />)}</tbody>
          </table>
        </div>
      )}

      {circuit.faithfulness && (
        <p className="text-xs text-slate-400">
          Faithfulness: necessity {circuit.faithfulness.necessity?.toFixed(2) ?? '—'}
          {circuit.faithfulness.sufficiency != null &&
            ` · sufficiency ${circuit.faithfulness.sufficiency.toFixed(2)}`}
        </p>
      )}
    </div>
  );
}

export function CircuitsPanel() {
  const [rows, setRows] = useState<CircuitSummary[]>([]);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(() => {
    setLoading(true);
    circuitsApi.list()
      .then((r) => { setRows(r.circuits); setError(null); })
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const setPromoted = (id: string, promoted: boolean) =>
    circuitsApi.setPromoted(id, promoted)
      .then((updated) =>
        setRows((rs) => rs.map((r) => (r.id === id ? { ...r, promoted: updated.promoted } : r))))
      .catch((e) => setError(String(e.message ?? e)));

  const remove = (id: string) => {
    if (window.confirm('Delete this circuit? Its validation manifests survive.')) {
      circuitsApi.remove(id).then(refresh).catch((e) => setError(String(e.message ?? e)));
    }
  };

  const downloadSlices = (id: string, name: string) =>
    circuitsApi.exportSlices(id).then((r) => {
      const blob = new Blob([JSON.stringify(r.slices, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      const safe = name.toLowerCase().replace(/[^a-z0-9-_]+/g, '-').replace(/^-+|-+$/g, '') || 'circuit';
      a.download = `${safe}.slices.json`;
      a.click();
      URL.revokeObjectURL(url);
    }).catch((e) => setError(String(e.message ?? e)));

  const fileInput = useRef<HTMLInputElement>(null);
  const onImportFile = (file: File) => {
    file.text()
      .then((text) => circuitsApi.importDefinition(JSON.parse(text)))
      .then(refresh)
      .catch((e) => setError(`Import failed: ${String(e.message ?? e)}`));
  };

  return (
    <div className="max-w-5xl mx-auto px-6 py-6 space-y-4">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-slate-100 flex items-center gap-2">
            <GitBranch className="w-5 h-5 text-violet-400" />
            Circuits
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            Cross-layer feature circuits with graded evidence. Every artifact shows its rung —
            discovery results are Tier-1 (same-token) associations until validated.
          </p>
        </div>
        <input
          ref={fileInput}
          type="file"
          accept=".json,application/json"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) onImportFile(f);
            e.target.value = '';
          }}
        />
        <button
          onClick={() => fileInput.current?.click()}
          className="flex items-center gap-1.5 rounded bg-slate-700 hover:bg-slate-600 px-3 py-1.5 text-xs text-slate-200 shrink-0"
          title="Import a mistudio.circuit-definition/v1 file"
        >
          <Upload className="w-3.5 h-3.5" /> Import
        </button>
      </div>

      {error && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-2 text-sm text-red-300">
          {error}
        </div>
      )}

      {loading ? (
        <p className="text-slate-500 text-sm">Loading…</p>
      ) : rows.length === 0 ? (
        <div className={`${COMPONENTS.card.base} p-8 text-center`}>
          <Layers className="w-10 h-10 text-slate-600 mx-auto mb-3" />
          <h3 className="text-slate-300 font-medium mb-1">No circuits yet</h3>
          <p className="text-slate-500 text-sm">
            Circuits arrive from discovery runs (coming with the mining feature), via the
            MCP <span className="font-mono">create_circuit</span> /{' '}
            <span className="font-mono">import_circuit_definition</span> tools, or by
            importing a <span className="font-mono">.circuit.json</span> file.
          </p>
        </div>
      ) : (
        rows.map((c) => (
          <div key={c.id} className={`${COMPONENTS.card.base} p-4`}>
            <div className="flex items-center gap-3">
              <button
                className="text-left flex-1 min-w-0"
                onClick={() => setExpanded(expanded === c.id ? null : c.id)}
              >
                <div className="flex items-center gap-2 min-w-0">
                  <span className="text-slate-100 font-medium truncate">{c.name}</span>
                  <RungChip rung={c.rung} language={c.rung_language} nextStep={c.rung_next_step} />
                  {c.promoted && (
                    <span className="rounded bg-emerald-500/10 text-emerald-300 px-1.5 py-0.5 text-[10px]">
                      promoted
                    </span>
                  )}
                </div>
                <div className="text-xs text-slate-500 mt-0.5">
                  {c.member_count} members · {c.edge_count} edges ·{' '}
                  {c.layers.map((l) => `L${l}`).join('+')}
                </div>
              </button>
              {c.promoted ? (
                <button
                  onClick={() => setPromoted(c.id, false)}
                  className="flex items-center gap-1 rounded bg-slate-700 hover:bg-slate-600 px-2.5 py-1.5 text-xs text-slate-200"
                  title="Unpromote (removes the loadable-profile badge)"
                >
                  <ArrowDownRight className="w-3.5 h-3.5" /> Unpromote
                </button>
              ) : (
                <button
                  onClick={() => setPromoted(c.id, true)}
                  className="flex items-center gap-1 rounded bg-violet-600 hover:bg-violet-500 px-2.5 py-1.5 text-xs text-white"
                  title="Promote to a loadable multi-layer steering profile (badge, not gate)"
                >
                  <ArrowUpRight className="w-3.5 h-3.5" /> Promote
                </button>
              )}
              <a
                href={circuitsApi.exportUrl(c.id)}
                className="p-1.5 rounded hover:bg-white/10 text-slate-400"
                title="Export circuit-definition/v1"
              >
                <Download className="w-4 h-4" />
              </a>
              <button
                onClick={() => downloadSlices(c.id, c.name)}
                className="p-1.5 rounded hover:bg-white/10 text-slate-400"
                title="Export per-layer v1 slices (partial renderings for single-SAE consumers)"
              >
                <FileDown className="w-4 h-4" />
              </button>
              <button
                onClick={() => remove(c.id)}
                className="p-1.5 rounded hover:bg-white/10 text-slate-400"
                title="Delete circuit"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>

            {expanded === c.id && <CircuitDetail id={c.id} onChanged={refresh} />}
          </div>
        ))
      )}
    </div>
  );
}

export default CircuitsPanel;
