/**
 * ClusterBudgetBar (Feature 013) — shows the cluster's computed strength
 * budget, its consumption by the current member strengths, the resultant-norm
 * gain G, and any coherence flags. Rendered only while a cluster allocation
 * governs the selection.
 */

import { AlertTriangle, Gauge } from 'lucide-react';
import { useSteeringStore } from '../../stores/steeringStore';

const FLAG_COPY: Record<string, string> = {
  cancellation: 'Members partially cancel each other — check the flagged pair',
  default_budget: 'No activation frequencies known — conservative default budget',
  cap_bound: 'Budget capped at the sum of solo optima',
  approximate: 'Decoder unavailable — constant-budget approximation (G=1)',
};

export function ClusterBudgetBar() {
  const { clusterBudget, clusterNotice, selectedFeatures } = useSteeringStore();

  // Gated cluster (e.g. low cohesion): no governing budget, just the notice.
  if (!clusterBudget) {
    if (!clusterNotice) return null;
    return (
      <div className="px-4 pb-2">
        <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 px-3 py-2">
          <p className="flex items-center gap-1 text-[10px] text-amber-400/90">
            <AlertTriangle className="w-2.5 h-2.5 shrink-0" />
            {clusterNotice}
          </p>
        </div>
      </div>
    );
  }

  const used = selectedFeatures.reduce((s, f) => s + Math.abs(f.strength), 0);
  const over = used > clusterBudget.B + 0.05;
  const pct = clusterBudget.B > 0 ? Math.min(100, (used / clusterBudget.B) * 100) : 0;
  const warnings = clusterBudget.flags.filter((f) => FLAG_COPY[f]);

  return (
    <div className="px-4 pb-2">
      <div className="rounded-lg border border-slate-700/60 bg-slate-900/60 px-3 py-2">
        <div className="flex items-center justify-between text-[11px] mb-1">
          <span className="flex items-center gap-1 text-slate-400">
            <Gauge className="w-3 h-3 text-cyan-400" />
            Cluster budget
          </span>
          <span className={`font-mono ${over ? 'text-amber-400' : 'text-slate-400'}`}>
            {used.toFixed(1)} / {clusterBudget.B.toFixed(1)}
            <span className="text-slate-600 ml-2" title="Resultant-norm gain of the injected direction">
              G {clusterBudget.G.toFixed(2)}
            </span>
          </span>
        </div>
        <div className="h-1.5 w-full rounded-full bg-slate-800 overflow-hidden">
          <div
            className={`h-full rounded-full transition-all ${over ? 'bg-amber-500' : 'bg-cyan-500'}`}
            style={{ width: `${pct}%` }}
          />
        </div>
        {warnings.length > 0 && (
          <div className="mt-1.5 space-y-0.5">
            {warnings.map((f) => (
              <p key={f} className="flex items-center gap-1 text-[10px] text-amber-400/90">
                <AlertTriangle className="w-2.5 h-2.5 shrink-0" />
                {FLAG_COPY[f]}
              </p>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
