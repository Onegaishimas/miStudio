/**
 * Circuits panel (Feature 018) — the review/promotion surface for
 * cross-layer circuits. Every circuit and edge shows its evidence rung via
 * SERVER-rendered language (IDL-35); promotion is a badge, not a gate.
 * Discovery/validation tabs (016/017) join this panel as they land.
 */

import { useCallback, useEffect, useState } from 'react';
import { GitBranch, ArrowUpRight, Trash2, Download, Layers } from 'lucide-react';
import { circuitsApi } from '../../api/circuits';
import type { Circuit, CircuitEdge } from '../../types/circuits';
import { RungChip } from '../circuits/RungChip';
import { COMPONENTS } from '../../config/brand';

function EdgeRow({ edge }: { edge: CircuitEdge }) {
  const label = (n: CircuitEdge['up']) =>
    n.kind === 'cluster' ? `cluster:${n.cluster_profile_id}` : `#${n.feature_idx}`;
  return (
    <tr className={`text-xs ${edge.type === 'persistence' ? 'opacity-60' : ''}`}>
      <td className="py-1 pr-3 font-mono text-slate-300">
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
        {edge.coactivation?.pmi != null && `PMI ${edge.coactivation.pmi.toFixed(2)} `}
        {edge.coactivation?.support != null && `· n=${edge.coactivation.support} `}
        {edge.attribution?.score != null && `· attr ${edge.attribution.score.toFixed(2)}`}
        {edge.weight_prior != null && ` · prior ${edge.weight_prior.toFixed(2)}`}
      </td>
    </tr>
  );
}

export function CircuitsPanel() {
  const [circuits, setCircuits] = useState<Circuit[]>([]);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(() => {
    setLoading(true);
    circuitsApi.list()
      .then((r) => { setCircuits(r.circuits); setError(null); })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const promote = (id: string) =>
    circuitsApi.promote(id).then(refresh).catch((e) => setError(e.message));
  const remove = (id: string) => {
    if (window.confirm('Delete this circuit?')) {
      circuitsApi.remove(id).then(refresh).catch((e) => setError(e.message));
    }
  };

  return (
    <div className="max-w-5xl mx-auto px-6 py-6 space-y-4">
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

      {error && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-2 text-sm text-red-300">
          {error}
        </div>
      )}

      {loading ? (
        <p className="text-slate-500 text-sm">Loading…</p>
      ) : circuits.length === 0 ? (
        <div className={`${COMPONENTS.card.base} p-8 text-center`}>
          <Layers className="w-10 h-10 text-slate-600 mx-auto mb-3" />
          <h3 className="text-slate-300 font-medium mb-1">No circuits yet</h3>
          <p className="text-slate-500 text-sm">
            Circuits arrive from discovery runs (coming with the mining feature) or via the
            MCP <span className="font-mono">create_circuit</span> tool.
          </p>
        </div>
      ) : (
        circuits.map((c) => (
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
                  {c.members.length} members · {c.edges.length} edges ·{' '}
                  {[...new Set(c.members.map((m) => m.layer))].sort((a, b) => a - b)
                    .map((l) => `L${l}`).join('+')}
                </div>
              </button>
              {!c.promoted && (
                <button
                  onClick={() => promote(c.id)}
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
                onClick={() => remove(c.id)}
                className="p-1.5 rounded hover:bg-white/10 text-slate-400"
                title="Delete circuit"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>

            {expanded === c.id && (
              <div className="mt-3 border-t border-slate-700/60 pt-3 space-y-3">
                {c.narrative && (
                  <p className="text-sm text-slate-400 whitespace-pre-wrap">{c.narrative}</p>
                )}
                <div>
                  <h4 className="text-xs font-medium text-slate-400 mb-1">Members by layer</h4>
                  {[...new Set(c.members.map((m) => m.layer))].sort((a, b) => a - b).map((layer) => (
                    <div key={layer} className="text-xs text-slate-300 mb-1">
                      <span className="font-mono text-slate-500 mr-2">L{layer}</span>
                      {c.members.filter((m) => m.layer === layer).map((m, i) => (
                        <span key={i} className="mr-2">
                          {m.member_kind === 'cluster_ref'
                            ? `[cluster ${m.cluster_name ?? m.cluster_profile_id}]`
                            : `#${m.feature?.feature_idx}${m.feature?.label ? ` ${m.feature.label}` : ''}`}
                        </span>
                      ))}
                    </div>
                  ))}
                </div>
                {c.edges.length > 0 && (
                  <div className="overflow-x-auto">
                    <h4 className="text-xs font-medium text-slate-400 mb-1">Edges</h4>
                    <table className="w-full">
                      <tbody>
                        {c.edges.map((e, i) => <EdgeRow key={i} edge={e} />)}
                      </tbody>
                    </table>
                  </div>
                )}
                {c.faithfulness && (
                  <p className="text-xs text-slate-400">
                    Faithfulness: necessity {c.faithfulness.necessity?.toFixed(2) ?? '—'}
                    {c.faithfulness.sufficiency != null &&
                      ` · sufficiency ${c.faithfulness.sufficiency.toFixed(2)}`}
                  </p>
                )}
              </div>
            )}
          </div>
        ))
      )}
    </div>
  );
}

export default CircuitsPanel;
