/**
 * AppliedFeaturesSummary — trust-by-inspection surface for Blended results
 * (Feature 012).
 *
 * Renders the server-returned `features_applied` list from a combined run so
 * users can verify that EVERY cluster member contributed its assigned
 * strength — answering the "is it really combining all of them?" doubt with
 * server truth rather than request state. Collapsed by default.
 */

import { useState } from 'react';
import { ChevronDown, ChevronRight, CheckCircle2 } from 'lucide-react';
import { CombinedFeatureApplied, FEATURE_COLORS } from '../../types/steering';

interface AppliedFeaturesSummaryProps {
  applied: CombinedFeatureApplied[];
}

export function AppliedFeaturesSummary({ applied }: AppliedFeaturesSummaryProps) {
  const [open, setOpen] = useState(false);

  if (!applied || applied.length === 0) return null;

  return (
    <div className="mt-2">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1 text-[11px] text-slate-500 hover:text-slate-300 transition-colors"
        title="Every feature the server applied in this blended generation"
      >
        {open ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
        <CheckCircle2 className="w-3 h-3 text-emerald-500" />
        Applied features ({applied.length})
      </button>

      {open && (
        <div className="mt-1.5 flex flex-wrap gap-1.5 pl-4">
          {applied.map((f, i) => {
            // Server data: fall back to teal if an unknown color name arrives.
            const colors = FEATURE_COLORS[f.color] ?? FEATURE_COLORS.teal;
            return (
              <span
                key={`${f.feature_idx}-${i}`}
                className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] border ${colors.border} ${colors.light} ${colors.text}`}
                title={f.label || undefined}
              >
                #{f.feature_idx}
                {f.label && <span className="text-slate-400 max-w-[10rem] truncate">{f.label}</span>}
                <span className="font-mono">
                  @ {f.strength > 0 ? '+' : ''}
                  {f.strength}
                </span>
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
}
