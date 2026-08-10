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
 * and `control_seed` are always sent, and the result reports all three arms —
 * baseline, intervened and control — so a reader can see the control actually
 * ran rather than taking it on trust. The SEPARATION is the finding; the
 * intervened rate alone is not one.
 *
 * RUNG 2, AND THIS DOCSTRING ONCE SAID OTHERWISE. It described a lens-space
 * displacement, which is what this measured before the causal rewrite —
 * `intervened_outcome`, `control_outcome` and `excess_over_control` are all
 * from that shape and none of them exist any more. What runs now perturbs
 * inside the model's own forward pass, lets it continue, and scores the target
 * token's RANK in the model's real output.
 *
 * MANY PROMPTS, NOT ONE. Below four trials NO outcome separates the intervened
 * and control intervals — a perfect intervened arm against a perfect null
 * control still overlaps — so a one-prompt run can only ever report "no effect
 * demonstrated", which reads as a fact about the direction and is a fact about
 * the sample size. This card is the only surface that can supply more, which
 * is why it has a trial-prompt list.
 */

import { useRef, useState } from 'react';
import { Loader2, Zap } from 'lucide-react';
import { jlensApi } from '../../api/jlens';
import { getTaskStatus } from '../../api/models';

const POLL_MS = 4000;

/**
 * Fewest trials at which disjoint Wilson intervals are arithmetically possible.
 *
 * MIRRORS `MIN_TRIALS_FOR_SEPARATION` in `services/jlens_causal.py`, where it is
 * derived rather than chosen. Duplicated here only to warn BEFORE a GPU job is
 * queued; the authority is the server, which reports
 * `min_trials_for_separation` with every result and is what the verdict above
 * renders.
 */
const MIN_TRIALS = 4;

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
  /**
   * FALSE when no outcome at this trial count COULD have separated them.
   *
   * A different question from `separated_from_control`, and the answers read
   * oppositely: one is "no effect was demonstrated", the other is "nothing
   * could have been demonstrated". The panel grew this branch and this card
   * did not, so the card kept printing the sentence the change exists to
   * remove — and it is the only surface from which a projective_ablation can
   * be run at all.
   */
  separation_attainable?: boolean;
  min_trials_for_separation?: number;
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
  /**
   * Extra trial prompts, one per line.
   *
   * THE ONLY WAY TO REACH A SEPARABLE RESULT. Below four trials no outcome
   * separates the intervened and control intervals, and every surface sent a
   * single prompt — so "no effect was demonstrated" was the only verdict the
   * product could produce, and the panel's remedy pointed here at a card that
   * had no such control.
   */
  const [extraPrompts, setExtraPrompts] = useState('');
  const [k, setK] = useState(4);
  const [seed, setSeed] = useState(20260802);
  const [state, setState] = useState<'idle' | 'running'>('idle');
  const [result, setResult] = useState<Outcome | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const chosen = token || pinned[0] || '';

  /**
   * The prompts this run will score, `prompt` first.
   *
   * DE-DUPLICATED AND TRIMMED HERE, not on the server: two identical trials are
   * one observation counted twice, which narrows the Wilson interval on
   * evidence that does not exist.
   */
  const trialPrompts = [
    prompt,
    ...extraPrompts.split('\n').map((p) => p.trim()),
  ].filter((p, i, all) => p.length > 0 && all.indexOf(p) === i);
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
        // ONE TRIAL EACH. The paper reports a FRACTION of trials — 50 two-hop
        // prompts, 192 swap trials — never one number from one prompt.
        prompts: trialPrompts.length > 1 ? trialPrompts : undefined,
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

          <label className="flex flex-col gap-1">
            <span className="text-xs text-slate-600 dark:text-slate-400">
              More trial prompts — one per line
            </span>
            <textarea
              value={extraPrompts}
              onChange={(e) => setExtraPrompts(e.target.value)}
              rows={4}
              placeholder={'The capital of Italy is\nThe capital of Japan is'}
              className="rounded border border-slate-300 bg-white px-2 py-1.5 font-mono text-xs text-slate-900 dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
              data-testid="intervention-prompts"
            />
          </label>
          {/* THE COUNT AND WHAT IT BUYS, before the run rather than after it.
              Below four trials NO outcome separates the intervened and control
              intervals — a perfect intervened arm against a perfect null
              control still overlaps — so a run at this size can only report
              "no effect demonstrated", which reads as a fact about the
              direction and is a fact about the sample. Saying so afterwards
              costs a GPU job to learn. */}
          <p
            className={`text-[10px] ${
              trialPrompts.length < MIN_TRIALS
                ? 'text-amber-700 dark:text-amber-400'
                : 'text-slate-500 dark:text-slate-500'
            }`}
            data-testid="intervention-trial-count"
          >
            {trialPrompts.length} trial{trialPrompts.length === 1 ? '' : 's'}
            {trialPrompts.length < MIN_TRIALS
              ? ` — separation is not attainable below ${MIN_TRIALS}. This will run, and its verdict will describe the sample rather than the direction.`
              : ' — enough for the intervals to separate if there is an effect.'}
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
                {result.separation_attainable === false
                  ? `Only ${result.n_trials} trial${
                      result.n_trials === 1 ? '' : 's'
                    } — separation is not attainable below ${
                      result.min_trials_for_separation ?? 4
                    }. This says nothing about the direction yet; add prompts below.`
                  : result.separated_from_control
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

            </div>
          )}
        </div>
      )}
    </section>
  );
}
