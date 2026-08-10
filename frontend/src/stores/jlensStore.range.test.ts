/**
 * The layer range is a REQUEST parameter, not a display filter.
 *
 * `check_readout_budget` bounds positions x layers BEFORE capture, so narrowing
 * has to reach the server to be worth anything — reading every layer and then
 * hiding most of them pays the whole cost and calls it a saving.
 *
 * MUTATION CONTROLS (each must turn this file red):
 *   * drop `layers` from the readout body     -> "sends the range" fails
 *   * send a range when none is set           -> "asks for every layer" fails
 *   * send the two bounds instead of the span -> "expands to every layer in it" fails
 */
import { describe, expect, it, vi, beforeEach } from 'vitest';

vi.mock('../api/jlens', () => ({
  jlensApi: {
    readout: vi.fn(),
    readoutResult: vi.fn(),
    listArtifacts: vi.fn().mockResolvedValue([]),
    intervene: vi.fn(),
  },
}));

import { jlensApi } from '../api/jlens';
import { useJLensStore } from './jlensStore';

function readoutReturns() {
  vi.mocked(jlensApi.readout).mockResolvedValue({
    task_id: 't1',
    model_id: 'm_1',
    status: 'queued',
  } as never);
  vi.mocked(jlensApi.readoutResult).mockResolvedValue({
    task_id: 't1',
    status: 'SUCCESS',
    readout: {
      meta: {
        kind: 'meta',
        model: 'org/m',
        types: ['LOGIT_LENS'],
        layers_by_type: { LOGIT_LENS: [4, 5, 6] },
        top_n: 4,
        prompt_len: 1,
      },
      tokens: [],
    },
  } as never);
}

beforeEach(() => {
  vi.clearAllMocks();
  useJLensStore.getState().reset();
  readoutReturns();
});

describe('the layer range reaches the server', () => {
  it('SENDS the range, expanded to every layer in it', async () => {
    /**
     * The endpoint takes an explicit list and has no notion of a range —
     * inventing one on the wire would be a miStudio-shaped field in a format
     * that is not ours to design (BR-029).
     *
     * MUTATION CONTROL: drop `layers` from the body, or send `[lo, hi]` rather
     * than the span, and this fails.
     */
    useJLensStore.setState({
      modelId: 'm_1',
      prompt: 'hello',
      layerRange: [4, 7],
    });
    await useJLensStore.getState().fetchReadout();

    expect(jlensApi.readout).toHaveBeenCalledTimes(1);
    const sent = vi.mocked(jlensApi.readout).mock.calls[0][0];
    expect(sent.layers).toEqual([4, 5, 6, 7]);
  });

  it('asks for EVERY layer when nothing is narrowed', async () => {
    /**
     * Absent, not an invented full range: `layers: null` is the endpoint's own
     * "all of them", and sending a computed span would pin the request to
     * whatever the client believed the model had.
     *
     * MUTATION CONTROL: always send a range and this fails.
     */
    useJLensStore.setState({ modelId: 'm_1', prompt: 'hello', layerRange: null });
    await useJLensStore.getState().fetchReadout();
    const sent = vi.mocked(jlensApi.readout).mock.calls[0][0];
    expect(sent.layers).toBeUndefined();
  });

  it('sends nothing for a range whose ends have crossed', async () => {
    /** Better to read everything than to ask for an empty selection. */
    useJLensStore.setState({ modelId: 'm_1', prompt: 'hello', layerRange: [9, 2] });
    await useJLensStore.getState().fetchReadout();
    const sent = vi.mocked(jlensApi.readout).mock.calls[0][0];
    expect(sent.layers).toBeUndefined();
  });

  it('survives a reload: the range persists with the rest of the setup', () => {
    useJLensStore.setState({ layerRange: [3, 9] });
    const persisted = JSON.parse(
      localStorage.getItem('miStudio-jlens') ?? '{"state":{}}',
    );
    expect(persisted.state.layerRange).toEqual([3, 9]);
  });
});
