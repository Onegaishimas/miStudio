/**
 * Run an intervention on a pinned token, with its matched control.
 *
 * WHY IT LIVES HERE. The intervention endpoint takes a d_model direction, which
 * no browser can produce — that requirement is precisely why this capability
 * shipped with no UI at all. The server now resolves a direction from a SINGLE
 * TOKEN's unembedding row, and a token is the one thing this panel always has
 * on screen. So the affordance is "intervene along this pinned token".
 *
 * THE CONTROL IS NOT OPTIONAL (BR-018). There is no checkbox to skip it: `k`
 * and `control_seed` are always sent, and the result reports the intervened
 * outcome, the control outcome and their difference — so a reader can see the
 * control actually ran rather than taking it on trust. `excess_over_control` is
 * the finding; the intervened outcome alone is not one.
 *
 * RUNG 1, NOT 2. This measures displacement in lens space. That is evidence
 * about the coordinate and is not causal proof the model used it.
 */

import { useRef, useState } from 'react';
import { Loader2, Zap } from 'lucide-react';
import { jlensApi } from '../../api/jlens';
import { getTaskStatus } from '../../api/models';

const POLL_MS = 4000;

/** The four primitives (BR-017). Recorded with every result. */
export const PRIMITIVES = [
  { id: 'additive', label: 'Additive', hint: 'Steer along the direction' },
  {
    id: 'projective_ablation',
    label: 'Projective ablation',
    hint: "Remove the activation's component along it",
  },
] as const;

interface InterventionCardProps {
  modelId: string;
  /** The prompt the readout on screen describes. */
  prompt: string;
  /** Tokens the user pinned — the directions available to act along. */
  pinned: string[];
  /** Layers the current readout covers, so a request cannot name an absent one. */
  layers: number[];
  artifactId: string | null;
}

/**
 * One arm's hit rate with its Wilson 95% interval.
 *
 * A RATE AND ITS INTERVAL, never a bare number. With twenty trials a ten-point
 * gap is noise, and the interval is what says so.
 */
interface Rates {
  hits: number;
  n: number;
  rate: number;
  ci95_low: number;
  ci95_high: number;
}

/**
 * What the task returns — `CausalReport.summary()`, verbatim.
 *
 * THIS SHAPE IS THE RUNG-2 ONE. The card was written against the rung-1 result
 * (`intervened_outcome` / `control_outcome` / `excess_over_control`, a lens-space
 * displacement) and never updated when the measurement became a real forward-pass
 * intervention. Every key it read had been gone since the rewrite, so the success
 * path called `.toFixed` on `undefined` and took the panel down — on success, and
 * only on success, which is why nothing noticed.
 */
interface Outcome {
  target_token: string;
  primitive: string;
  layers: number[];
  strength: number | null;
  n_trials: number;
  baseline_top1: Rates;
  intervened_top1: Rates;
  control_top1: Rates;
  baseline_top5: Rates;
  intervened_top5: Rates;
  control_top5: Rates;
  excess_top1_over_control: number;
  excess_top5_over_control: number;
  /** TRUE means the intervened and control intervals are DISJOINT. */
  separated_from_control: boolean;
}

/** `12/24 = 0.500 [0.31, 0.69]` — the count, the rate and the interval. */
function fmtRates(r: Rates | undefined): string {
  if (!r) return 'n/a';
  return (
    `${r.hits}/${r.n} = ${r.rate.toFixed(3)} ` +
    `[${r.ci95_low.toFixed(2)}, ${r.ci95_high.toFixed(2)}]`
  );
}

export function InterventionCard({
  prompt,
  modelId,
  pinned,
  layers,
  artifactId,
}: InterventionCardProps) {
  const [open, setOpen] = useState(false);
  const [token, setToken] = useState('');
  const [primitive, setPrimitive] = useState<string>('additive');
  const [strength, setStrength] = useState(1);
  const [k, setK] = useState(4);
  const [seed, setSeed] = useState(20260802);
  const [state, setState] = useState<'idle' | 'running'>('idle');
  const [result, setResult] = useState<Outcome | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const chosen = token || pinned[0] || '';
  const canRun = state === 'idle' && !!modelId && !!chosen && layers.length > 0;

  const poll = (taskId: string) => {
    timer.current = setTimeout(async () => {
      try {
        const status = await getTaskStatus(taskId);
        if (status.state === 'SUCCESS') {
          setResult(status.result as Outcome);
          setState('idle');
          return;
        }
        if (status.state === 'FAILURE') {
          setError(status.error ?? 'The intervention failed with no reason given.');
          setState('idle');
          return;
        }
        poll(taskId);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Lost track of the run.');
        setState('idle');
      }
    }, POLL_MS);
  };

  const run = async () => {
    if (!canRun) return;
    setError(null);
    setResult(null);
    setState('running');
    try {
      const accepted = await jlensApi.intervene({
        model_id: modelId,
        // THE PROMPT ON SCREEN. This was `''` — an empty string — so every
        // intervention launched from this card scored a forward pass over
        // nothing, while the readout beside it described a real prompt. The
        // result named a layer and a direction and measured neither in the
        // context the reader was looking at.
        prompt,
        primitive,
        layers,
        direction_token: chosen,
        strength,
        // Always sent. An intervention without a size-matched control is not a
        // weaker finding — it is not a finding.
        k,
        control_seed: seed,
        artifact_id: artifactId,
      });
      poll(accepted.task_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not queue the run.');
      setState('idle');
    }
  };

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-3 dark:border-slate-700 dark:bg-slate-800">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs font-medium text-slate-600 dark:text-slate-400">
          Intervention
        </span>
        <span className="text-[10px] text-slate-500 dark:text-slate-500">
          rung 2 · real intervention, scored against a matched control
        </span>
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          disabled={!pinned.length}
          title={
            pinned.length
              ? undefined
              : 'Pin a token first — it supplies the direction to act along.'
          }
          className="ml-auto flex items-center gap-1 rounded border border-slate-300 px-2 py-1 text-xs text-slate-700 hover:bg-slate-100 disabled:opacity-50 dark:border-slate-600 dark:text-slate-300 dark:hover:bg-slate-700"
        >
          <Zap className="h-3 w-3" />
          {open ? 'Close' : 'Intervene…'}
        </button>
      </div>

      {open && (
        <div className="mt-3 space-y-3">
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-600 dark:text-slate-400">
                Direction — a pinned token
              </span>
              <select
                value={chosen}
                onChange={(e) => setToken(e.target.value)}
                disabled={pinned.length === 0}
                className="rounded border border-slate-300 bg-white px-2 py-1.5 text-xs text-slate-900 disabled:opacity-60 dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
              >
                {/* AN EMPTY LIST MUST EXPLAIN ITSELF. With nothing pinned this
                    rendered as a blank select beside three fields that DO
                    accept typing, which reads as a broken control rather than
                    as a missing prerequisite — and the caption "a pinned token"
                    only makes sense once you already know what pinning is. */}
                {pinned.length === 0 ? (
                  <option value="">
                    No pinned tokens — click one in the readout below to pin it
                  </option>
                ) : (
                  pinned.map((p) => (
                    <option key={p} value={p}>
                      {p}
                    </option>
                  ))
                )}
              </select>
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-600 dark:text-slate-400">
                Primitive
              </span>
              <select
                value={primitive}
                onChange={(e) => setPrimitive(e.target.value)}
                className="rounded border border-slate-300 bg-white px-2 py-1.5 text-xs text-slate-900 dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
              >
                {PRIMITIVES.map((p) => (
                  <option key={p.id} value={p.id} title={p.hint}>
                    {p.label}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <div className="grid grid-cols-3 gap-3">
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-600 dark:text-slate-400">
                Strength
              </span>
              <input
                type="number"
                step="0.1"
                value={strength}
                onChange={(e) => setStrength(Number(e.target.value))}
                className="rounded border border-slate-300 bg-white px-2 py-1.5 text-xs dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-600 dark:text-slate-400">
                Control size k
              </span>
              <input
                type="number"
                min={1}
                value={k}
                onChange={(e) => setK(Math.max(1, Number(e.target.value)))}
                className="rounded border border-slate-300 bg-white px-2 py-1.5 text-xs dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-600 dark:text-slate-400">
                Control seed
              </span>
              <input
                type="number"
                value={seed}
                onChange={(e) => setSeed(Number(e.target.value))}
                className="rounded border border-slate-300 bg-white px-2 py-1.5 text-xs dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
              />
            </label>
          </div>
          <p className="text-[10px] text-slate-500 dark:text-slate-500">
            The control runs every time and cannot be turned off. &ldquo;A random
            direction&rdquo; is not a control; &ldquo;k random directions from
            seed s&rdquo; is, and one nobody can reconstruct is not one either.
          </p>

          <button
            type="button"
            onClick={run}
            disabled={!canRun}
            className="rounded bg-emerald-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-emerald-700 disabled:cursor-not-allowed disabled:bg-slate-300 dark:disabled:bg-slate-700"
          >
            {state === 'running' ? (
              <span className="flex items-center gap-1">
                <Loader2 className="h-3 w-3 animate-spin" /> Running with control…
              </span>
            ) : (
              'Run with control'
            )}
          </button>

          {error && (
            <p className="text-[11px] text-red-600 dark:text-red-400" role="alert">
              {error}
            </p>
          )}

          {result && (
            <div
              className="rounded border border-slate-200 p-2 dark:border-slate-700"
              data-testid="intervention-result"
            >
              {/* THE VERDICT FIRST, in the terms the measurement supports:
                  disjoint intervals or not. Overlap is "not demonstrated
                  here", never "demonstrated absent" — the asymmetry is the
                  whole reason the control exists. */}
              <p
                className={`text-[11px] font-medium ${
                  result.separated_from_control
                    ? 'text-emerald-700 dark:text-emerald-400'
                    : 'text-amber-700 dark:text-amber-400'
                }`}
              >
                {result.separated_from_control
                  ? 'The intervals are DISJOINT — an effect over the matched control was demonstrated.'
                  : 'The intervals OVERLAP — no effect was demonstrated here, which is not the same as none existing.'}
              </p>

              {/* ALL THREE ARMS. The baseline is not decoration: an
                  intervention that "achieves" top-1 on prompts where the model
                  already answered that way has moved nothing, and without the
                  baseline any prompt set can manufacture a result. */}
              <dl className="mt-1.5 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 font-mono text-[10px]">
                <dt className="text-slate-500 dark:text-slate-400">baseline</dt>
                <dd className="text-slate-700 dark:text-slate-200">
                  {fmtRates(result.baseline_top1)}
                </dd>
                <dt className="text-slate-500 dark:text-slate-400">intervened</dt>
                <dd className="text-slate-700 dark:text-slate-200">
                  {fmtRates(result.intervened_top1)}
                </dd>
                <dt className="text-slate-500 dark:text-slate-400">control</dt>
                <dd className="text-slate-700 dark:text-slate-200">
                  {fmtRates(result.control_top1)}
                </dd>
              </dl>

              <p className="mt-1 font-mono text-[10px] text-slate-500 dark:text-slate-500">
                excess top-1 over control{' '}
                {result.excess_top1_over_control.toFixed(4)} · {result.n_trials}{' '}
                trial{result.n_trials === 1 ? '' : 's'}
              </p>
              {result.n_trials === 1 && (
                // ONE TRIAL IS ONE OBSERVATION. Its Wilson interval spans
                // almost the whole range, and a reader seeing "1/1 = 1.000"
                // without this reads a certainty the arithmetic does not carry.
                <p className="mt-1 text-[10px] text-amber-700 dark:text-amber-400">
                  One trial. The interval spans nearly the whole range — add
                  prompts before reading anything into this.
                </p>
              )}
            </div>
          )}
        </div>
      )}
    </section>
  );
}
