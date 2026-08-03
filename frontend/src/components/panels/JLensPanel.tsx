/**
 * J-Lens — position x layer readout viewer (Feature 023).
 *
 * Ported from the interaction specification at `0xcc/brds/JSpacePanel.jsx`
 * (BR-010). Four things changed in the port and each was a correctness issue,
 * not a style one:
 *
 *   1. The mock's `LAYERS` constant (21 layers at 0,5,...,100) is GONE. The
 *      axis is `meta.layers_by_type[type]`; real models here have 16 or 26.
 *   2. The mock's `BAND = { workspaceStart: 40, motorStart: 90 }` is GONE and
 *      has no replacement. Those are the source paper's Sonnet-4.5 boundaries;
 *      BR-002 forbids porting them and requires that porting be impossible by
 *      construction, so bands render only from a BandReport.
 *   3. The mock's `TOP_N = 8` is GONE. The colour ramp and the chart domain use
 *      `meta.top_n`, because the server decides how deep the readout goes.
 *   4. `FIXTURES` / `buildFixture` / `scoreAt` / `NOISE` are GONE. Everything
 *      here comes from a live readout; a synthetic one is indistinguishable
 *      from a real one once rendered.
 */

import { useEffect, useMemo, useState } from 'react';
import { Check, ChevronRight, Eraser, Layers, Link2, Pin, PinOff } from 'lucide-react';
import { ArtifactsStrip } from '../jlens/ArtifactsStrip';
import { ByLayerRail } from '../jlens/ByLayerRail';
import { EvidenceRungCard } from '../jlens/EvidenceRungCard';
import { FitLensCard } from '../jlens/FitLensCard';
import { InterventionCard } from '../jlens/InterventionCard';
import { WatchlistCard } from '../jlens/WatchlistCard';
import { LensModeTabs } from '../jlens/LensModeTabs';
import { ProvenanceStrip } from '../jlens/ProvenanceStrip';
import { ReadoutGrid } from '../jlens/ReadoutGrid';
import { TrajectoryChart } from '../jlens/TrajectoryChart';
import { PIN_COLORS, displayToken } from '../jlens/utils';
import {
  MAX_PROMPT_CHARS,
  artifactSlugFor,
  axisFor,
  readTypeFor,
  decodePermalink,
  encodePermalink,
  sliceFor,
  tokenAtPosition,
  useJLensStore,
} from '../../stores/jlensStore';
import { useModelsStore } from '../../stores/modelsStore';

export function JLensPanel() {
  const {
    modelId,
    prompt,
    meta,
    tokens,
    bandReport,
    provenance,
    lensMode,
    selPos,
    selLayerIdx,
    pinned,
    hover,
    isLoading,
    error,
    artifacts,
    modelRepoId,
    readoutPrompt,
    restored,
    setModelId,
    setPrompt,
    setLensMode,
    setSelPos,
    setSelLayerIdx,
    setHover,
    togglePin,
    fetchReadout,
    clearConfig,
  } = useJLensStore();

  // Selector-scoped: the models store ticks on every download-progress frame,
  // and subscribing to the whole store re-rendered the entire readout grid on
  // each tick even though only the dropdown depends on it.
  const models = useModelsStore((s) => s.models);
  const [promptDraft, setPromptDraft] = useState(prompt);
  const [fitPrefill, setFitPrefill] = useState<number[] | null>(null);
  const [copied, setCopied] = useState(false);

  // MOUNT ONCE, and deliberately not keyed on the action identities. Zustand
  // actions are stable in production, so `[fetchModels, fetchArtifacts]` looks
  // equivalent — but it makes a one-shot fetch depend on a reference staying
  // stable, and the moment one does not, the effect re-runs, sets state,
  // re-renders and re-runs. Reading the actions off the store here removes the
  // dependency instead of assuming it holds.
  useEffect(() => {
    useModelsStore.getState().fetchModels();
    void useJLensStore.getState().fetchArtifacts();
  }, []);

  // A permalink WINS over the persisted setup, and only on arrival. Someone
  // following a link is asking for that link's readout, not for whatever they
  // were last looking at — and re-applying it on every render would make the
  // form unusable, since every edit would be reverted.
  useEffect(() => {
    const link = decodePermalink(window.location.hash);
    if (!link) return;
    const store = useJLensStore.getState();
    if (link.prompt) {
      store.setPrompt(link.prompt);
      setPromptDraft(link.prompt);
    }
    store.setLensMode(link.mode);
    store.clearPins();
    for (const pin of link.pins) store.togglePin(pin);
    if (link.repo) {
      // The model is resolved by REPO ID: `m_xxxxxxxx` is local to one
      // installation, so a link built from it means nothing anywhere else.
      const match = useModelsStore
        .getState()
        .models.find((m) => m.repo_id === link.repo);
      if (match) store.setModelId(match.id, link.repo);
    }
  }, [models]);

  const readyModels = useMemo(
    () => models.filter((m) => m.status === 'ready'),
    [models]
  );

  // Real dimensions or nothing. The envelope check derives its bound from
  // these; a guessed value passes on one model while missing a real
  // materialisation on another, which is why the API requires them.
  const artifactDims = useMemo(() => {
    const chosen = readyModels.find((m) => m.id === modelId);
    const cfg = chosen?.architecture_config;
    if (!cfg?.hidden_size || !cfg?.num_hidden_layers || !cfg?.vocab_size) {
      return null;
    }
    return {
      d_model: cfg.hidden_size,
      n_layers: cfg.num_hidden_layers,
      n_vocab: cfg.vocab_size,
    };
  }, [readyModels, modelId]);

  const readType = readTypeFor(lensMode);
  const axis = useMemo(() => axisFor(meta, readType), [meta, readType]);
  const topN = meta?.top_n ?? 0;

  // By POSITION, not by array index: the wire format allows a readout over a
  // subset of positions, where the two differ.
  const selToken = tokenAtPosition(tokens, selPos);
  const selSlice = sliceFor(selToken, readType);
  const overLimit = promptDraft.length > MAX_PROMPT_CHARS;

  // The readout detail follows the pointer when there is one and the SELECTION
  // otherwise, so the top-k list — and therefore pinning — stays reachable
  // without a mouse.
  const detail = useMemo(() => {
    const source = hover ?? { pos: selPos, layerIdx: selLayerIdx };
    const token = tokenAtPosition(tokens, source.pos);
    const row = sliceFor(token, readType)?.top_tokens[source.layerIdx];
    if (!token || !row) return null;
    return { ...source, tokens: row };
  }, [hover, selPos, selLayerIdx, tokens, readType]);

  const submit = () => {
    setPrompt(promptDraft);
    // setPrompt lands before fetchReadout reads it: zustand's set is synchronous.
    void fetchReadout();
  };

  return (
    // A TWO-REGION PANEL. The request block stays put while the readout
    // scrolls under it: the grid is tall, and losing the model selector and
    // the prompt off the top of the page meant scrolling back up to change
    // either. Height is the viewport minus the app header, which is `h-14`
    // and `sticky` — taking 100dvh here would push the request block off the
    // bottom by exactly that much.
    <div className="flex h-[calc(100dvh-3.5rem)] flex-col px-6 pt-8">
      <header className="mb-4 flex shrink-0 flex-wrap items-center gap-3">
        <div className="flex items-center gap-2">
          <Layers className="h-5 w-5 text-emerald-500 dark:text-emerald-400" />
          <h1 className="text-lg font-semibold tracking-tight text-slate-900 dark:text-slate-100">
            J-Lens Readout
          </h1>
        </div>
        {meta && (
          <span className="rounded border border-slate-300 bg-slate-100 px-2 py-0.5 font-mono text-xs text-slate-700 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-300">
            {meta.model}
          </span>
        )}
        {/* A RESTORED GRID THAT NO LONGER MATCHES THE PROMPT BOX.
            Keeping the readout across a refresh is worth a model load and a
            minute of GPU; the cost is that editing the prompt afterwards
            leaves a grid that looks current and is not. Say which prompt it
            describes rather than clearing it or letting it pass as fresh. */}
        {meta && readoutPrompt && readoutPrompt !== promptDraft && (
          <span
            className="rounded border border-amber-400 px-2 py-0.5 text-[11px] text-amber-700 dark:border-amber-600 dark:text-amber-400"
            title={readoutPrompt}
          >
            showing the readout for “{readoutPrompt.slice(0, 40)}
            {readoutPrompt.length > 40 ? '…' : ''}” — read out again to update
          </span>
        )}
        {meta && restored && readoutPrompt === promptDraft && (
          <span className="text-[11px] text-slate-500 dark:text-slate-500">
            restored from your last session
          </span>
        )}
        <div className="ml-auto">
          <LensModeTabs
            meta={meta}
            mode={lensMode}
            onChange={setLensMode}
            // Derived from the SAME registry the readout resolves against, so
            // the tab's reason cannot claim an artifact the readout will not
            // find (or deny one it will).
            hasArtifact={artifacts.some(
              (a) => a.slug === artifactSlugFor(modelRepoId)
            )}
          />
        </div>
      </header>

      {/* ------------------------------------------------------------ request */}
      <section className="mb-4 shrink-0 rounded-lg border border-slate-200 bg-white p-3 dark:border-slate-700 dark:bg-slate-800">
        <div className="flex flex-wrap items-end gap-2">
          <label className="flex flex-col gap-1">
            <span className="text-xs font-medium text-slate-600 dark:text-slate-400">
              Model
            </span>
            <select
              value={modelId}
              onChange={(e) => {
                // The repo id travels with the selection: the artifact slug is
                // derived from it, and that derivation is how a lens fitted for
                // a base model is kept off its instruction-tuned variant.
                const chosen = readyModels.find((m) => m.id === e.target.value);
                setModelId(e.target.value, chosen?.repo_id ?? '');
              }}
              className="rounded border border-slate-300 bg-white px-2 py-1.5 text-sm text-slate-900 dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
            >
              <option value="">Select a model…</option>
              {readyModels.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name}
                </option>
              ))}
            </select>
          </label>
          <label className="flex min-w-[18rem] flex-1 flex-col gap-1">
            <span className="text-xs font-medium text-slate-600 dark:text-slate-400">
              Prompt
            </span>
            <input
              value={promptDraft}
              onChange={(e) => setPromptDraft(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && submit()}
              placeholder="The capital of France is"
              maxLength={MAX_PROMPT_CHARS}
              className="rounded border border-slate-300 bg-white px-2 py-1.5 text-sm text-slate-900 dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
            />
          </label>
          <button
            type="button"
            onClick={submit}
            disabled={isLoading || !modelId || !promptDraft.trim() || overLimit}
            className="rounded bg-emerald-600 px-3 py-1.5 text-sm font-medium text-white transition-colors hover:bg-emerald-700 disabled:cursor-not-allowed disabled:bg-slate-300 dark:disabled:bg-slate-700"
          >
            {isLoading ? 'Reading…' : 'Read out'}
          </button>
          <button
            type="button"
            onClick={() => {
              clearConfig();
              // The draft is component state, so clearing the store alone
              // leaves the old text sitting in the box — visibly contradicting
              // the "cleared" state and re-submittable with one click.
              setPromptDraft('');
            }}
            disabled={isLoading || (!modelId && !promptDraft && !prompt)}
            title="Forget the saved model, prompt, lens mode and pins"
            className="flex items-center gap-1 rounded border border-slate-300 px-2.5 py-1.5 text-sm text-slate-600 transition-colors hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-40 dark:border-slate-600 dark:text-slate-400 dark:hover:bg-slate-700"
          >
            <Eraser className="h-3.5 w-3.5" />
            Clear
          </button>
          <button
            type="button"
            onClick={async () => {
              const link =
                window.location.origin +
                window.location.pathname +
                encodePermalink({
                  repo: modelRepoId,
                  prompt: promptDraft,
                  mode: lensMode,
                  pins: pinned,
                });
              try {
                await navigator.clipboard.writeText(link);
                setCopied(true);
                window.setTimeout(() => setCopied(false), 1800);
              } catch {
                // Clipboard access is denied outside a secure context. Say so
                // rather than showing a tick for something that did not happen.
                setCopied(false);
              }
            }}
            disabled={!modelRepoId && !promptDraft}
            title="Copy a link to this model, prompt, lens and pins. Setup only — the readout is recomputed by whoever opens it."
            className="flex items-center gap-1 rounded border border-slate-300 px-2.5 py-1.5 text-sm text-slate-600 transition-colors hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-40 dark:border-slate-600 dark:text-slate-400 dark:hover:bg-slate-700"
          >
            {copied ? (
              <Check className="h-3.5 w-3.5 text-emerald-500" />
            ) : (
              <Link2 className="h-3.5 w-3.5" />
            )}
            {copied ? 'Copied' : 'Link'}
          </button>
        </div>
        {promptDraft.length > MAX_PROMPT_CHARS * 0.9 && (
          <p className="mt-2 text-xs text-slate-500 dark:text-slate-500">
            {promptDraft.length} / {MAX_PROMPT_CHARS} characters. Readout cost
            grows with position count, so longer prompts are refused rather than
            truncated.
          </p>
        )}
        {error && (
          <p className="mt-2 text-xs text-red-600 dark:text-red-400" role="alert">
            {error}
          </p>
        )}
      </section>

      {/* Everything below scrolls as ONE region, so the artifact strip, the
          cards and the grid move together rather than the grid scrolling
          inside a box inside a scrolling page. `min-h-0` is load-bearing: a
          flex child defaults to min-height:auto and would refuse to shrink,
          which makes the container grow instead of scrolling. */}
      <div className="min-h-0 flex-1 overflow-y-auto pb-8">
      <div className="mb-4">
        <ArtifactsStrip
          artifacts={artifacts}
          expectedSlug={artifactSlugFor(modelRepoId)}
          // NULL when the model's real dimensions are unknown, which disables
          // the Validate button. The envelope bound is derived from these, so
          // sending placeholders would produce a bound derived from nothing —
          // a check that reports a verdict it never computed.
          dims={artifactDims}
          // THE UNION, not the complement. Fitting only the missing layers
          // would produce an artifact holding only those — the server refuses
          // it now, and offering it here would be offering a coverage loss.
          onFitMissing={(missing) =>
            setFitPrefill(
              Array.from(
                new Set([
                  ...(artifacts.find(
                    (a) => a.slug === artifactSlugFor(modelRepoId)
                  )?.layers ?? []),
                  ...missing,
                ])
              ).sort((a, b) => a - b)
            )
          }
        />
      </div>

      <div className="mb-4">
        <FitLensCard
          modelId={modelId}
          prefillLayers={fitPrefill}
          // Re-list the registry rather than optimistically inserting a row:
          // the filesystem IS the registry (PADR IDL-46), and a row this client
          // invented would be a second registry that can disagree with the one
          // the readout path actually reads.
          onFitted={() => void useJLensStore.getState().fetchArtifacts()}
        />
      </div>

      <div className="mb-4 grid grid-cols-1 gap-4 lg:grid-cols-2">
        <InterventionCard
          modelId={modelId}
          // Pinned tokens ARE the available directions: the server resolves a
          // token's unembedding row, which is what makes this reachable at all.
          pinned={pinned}
          layers={axis}
          artifactId={
            artifacts.find((a) => a.slug === artifactSlugFor(modelRepoId))?.slug ??
            null
          }
        />
        <WatchlistCard
          artifactId={
            artifacts.find((a) => a.slug === artifactSlugFor(modelRepoId))?.slug ??
            null
          }
        />
      </div>

      {!meta ? (
        <section className="rounded-lg border border-slate-200 bg-white p-6 text-center dark:border-slate-700 dark:bg-slate-800">
          <p className="text-sm text-slate-600 dark:text-slate-400">
            Enter a prompt to read out what the model is poised to say at every
            layer and position.
          </p>
          <p className="mt-1 text-xs text-slate-500 dark:text-slate-500">
            The logit lens needs no fitted artifact and works on any model.
          </p>
        </section>
      ) : (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1fr_320px]">
          <div className="min-w-0 space-y-4">
            {/* prompt strip */}
            <section className="rounded-lg border border-slate-200 bg-white p-3 dark:border-slate-700 dark:bg-slate-800">
              <div className="mb-2 flex items-center gap-2 text-xs font-medium text-slate-600 dark:text-slate-400">
                Prompt
                <span className="text-slate-400 dark:text-slate-600">·</span>
                <span className="font-normal text-slate-500 dark:text-slate-500">
                  click a token to inspect its position
                </span>
              </div>
              <div className="flex flex-wrap gap-1">
                {tokens.map((t) => (
                  <button
                    key={t.position}
                    type="button"
                    onClick={() => setSelPos(t.position)}
                    className={`rounded border px-1.5 py-1 font-mono text-xs transition ${
                      selPos === t.position
                        ? 'border-emerald-500 bg-emerald-50 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-100'
                        : 'border-slate-200 bg-slate-50 text-slate-700 hover:border-slate-400 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300 dark:hover:border-slate-500'
                    }`}
                  >
                    {displayToken(t.token)}
                  </button>
                ))}
                {tokens.some((t) => t.is_generated) && (
                  <span className="ml-2 flex items-center gap-1 font-mono text-xs text-slate-500">
                    <ChevronRight className="h-3 w-3" />
                    generated
                  </span>
                )}
              </div>
            </section>

            {/* grid */}
            <section className="rounded-lg border border-slate-200 bg-white p-3 dark:border-slate-700 dark:bg-slate-800">
              <ReadoutGrid
                axis={axis}
                tokens={tokens}
                topN={topN}
                mode={lensMode}
                sliceOf={(t) => sliceFor(t, readType)}
                logitSliceOf={(t) => sliceFor(t, 'LOGIT_LENS')}
                // The logit lens's OWN axis. Diff aligns by absolute layer,
                // and the two axes are independent now that a partial artifact
                // reports only the layers it was fitted for.
                logitAxis={axisFor(meta, 'LOGIT_LENS')}
                // Where J is the identity, Diff is empty by construction. The
                // grid says so rather than letting an empty top row read as a
                // measurement.
                degenerateLayers={
                  artifacts.find((a) => a.slug === artifactSlugFor(modelRepoId))
                    ?.degenerate_layers ?? []
                }
                pinned={pinned}
                selPos={selPos}
                selLayerIdx={selLayerIdx}
                bandReport={bandReport}
                onSelect={(pos, layerIdx) => {
                  setSelPos(pos);
                  setSelLayerIdx(layerIdx);
                }}
                onHover={setHover}
              />

              {/* Readout detail. Falls back to the SELECTED cell when nothing
                  is hovered, because hover is pointer-only: with a placeholder
                  here instead, pinning — the panel's core interaction — had no
                  keyboard path at all. The prompt strip and the by-layer rail
                  are buttons, so selecting a cell is reachable; this makes the
                  top-k of that cell reachable too. */}
              <div
                data-testid="jlens-hover-detail"
                className="mt-2 min-h-[54px] rounded border border-slate-200 bg-slate-50 p-2 dark:border-slate-700 dark:bg-slate-900"
              >
                {detail ? (
                  <>
                    <div className="mb-1 font-mono text-[10px] text-slate-500">
                      L{axis[detail.layerIdx]} · pos {detail.pos}
                      {!hover && <span className="ml-1">(selected)</span>}
                    </div>
                    <div className="flex flex-wrap gap-1">
                      {detail.tokens.map((t, k) => (
                        <button
                          key={k}
                          type="button"
                          onClick={() => togglePin(t)}
                          className={`rounded px-1.5 py-0.5 font-mono text-[10px] ${
                            pinned.includes(t)
                              ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-800 dark:text-emerald-100'
                              : 'bg-slate-200 text-slate-700 hover:bg-slate-300 dark:bg-slate-800 dark:text-slate-300 dark:hover:bg-slate-700'
                          }`}
                        >
                          {displayToken(t)}
                        </button>
                      ))}
                    </div>
                    <p className="mt-1 text-[10px] text-slate-500 dark:text-slate-600">
                      Full top-{topN} readout. Click a token to pin it.
                    </p>
                  </>
                ) : (
                  <p className="text-[11px] text-slate-500 dark:text-slate-600">
                    Hover or select a cell for its full top-{topN} readout.
                  </p>
                )}
              </div>
            </section>

            {pinned.length > 0 && (
              <section className="rounded-lg border border-slate-200 bg-white p-3 dark:border-slate-700 dark:bg-slate-800">
                <TrajectoryChart
                  // Overlay the OTHER lens whenever the readout carries both,
                  // so "the Jacobian leads the logit lens" is something you can
                  // see rather than something the docs assert.
                  compareSlice={
                    readType === 'JACOBIAN_LENS'
                      ? sliceFor(selToken, 'LOGIT_LENS')
                      : undefined
                  }
                  compareAxis={
                    readType === 'JACOBIAN_LENS'
                      ? axisFor(meta, 'LOGIT_LENS')
                      : undefined
                  }
                  axis={axis}
                  slice={selSlice}
                  pinned={pinned}
                  topN={topN}
                  selPos={selPos}
                  bandReport={bandReport}
                />
              </section>
            )}
          </div>

          {/* ------------------------------------------------------ right rail */}
          <div className="space-y-4">
            <section className="rounded-lg border border-slate-200 bg-white p-3 dark:border-slate-700 dark:bg-slate-800">
              <div className="mb-2 flex items-center gap-1.5 text-xs font-medium text-slate-600 dark:text-slate-400">
                <Pin className="h-3.5 w-3.5" /> Pinned tokens
              </div>
              {pinned.length === 0 ? (
                <p className="text-[11px] text-slate-500 dark:text-slate-600">
                  Nothing pinned. Pin a token to turn the grid into a rank
                  heatmap.
                </p>
              ) : (
                <div className="flex flex-wrap gap-1.5">
                  {pinned.map((p, i) => (
                    <button
                      key={p}
                      type="button"
                      onClick={() => togglePin(p)}
                      className="group flex items-center gap-1 rounded border border-slate-300 bg-slate-50 px-2 py-1 font-mono text-[11px] text-slate-800 hover:border-slate-400 dark:border-slate-600 dark:bg-slate-900 dark:text-slate-200 dark:hover:border-slate-500"
                    >
                      <span
                        className="inline-block h-2 w-2 rounded-sm"
                        style={{ background: PIN_COLORS[i % PIN_COLORS.length] }}
                      />
                      {p}
                      <PinOff className="h-3 w-3 text-slate-400 group-hover:text-slate-600 dark:text-slate-600 dark:group-hover:text-slate-300" />
                    </button>
                  ))}
                </div>
              )}
            </section>

            <section className="rounded-lg border border-slate-200 bg-white p-3 dark:border-slate-700 dark:bg-slate-800">
              <ByLayerRail
                axis={axis}
                slice={selSlice}
                pinned={pinned}
                selLayerIdx={selLayerIdx}
                selPos={selPos}
                positionToken={selToken?.token ?? ''}
                onSelectLayer={setSelLayerIdx}
              />
            </section>

            <section className="rounded-lg border border-slate-200 bg-white p-3 dark:border-slate-700 dark:bg-slate-800">
              <EvidenceRungCard />
            </section>
          </div>
        </div>
      )}

      <ProvenanceStrip provenance={provenance} bandsAvailable={bandReport != null} />
      </div>
    </div>
  );
}
