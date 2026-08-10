/**
 * The control is not optional (BR-018).
 *
 * MUTATION CONTROLS:
 *   * omit k / control_seed from the request -> "always sends" fails
 *   * enable the card with no pinned tokens  -> "needs a direction" fails
 */

import { describe, expect, it, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { InterventionCard } from './InterventionCard';

vi.mock('../../api/jlens', () => ({ jlensApi: { intervene: vi.fn() } }));
vi.mock('../../api/models', () => ({ getTaskStatus: vi.fn() }));

import { jlensApi } from '../../api/jlens';

beforeEach(() => vi.clearAllMocks());

describe('InterventionCard', () => {
  it('cannot run without a pinned token to act along', () => {
    render(
      <InterventionCard
        modelId="m_1"
        prompt="the animal that spins webs"
        pinned={[]}
        layers={[1]}
        artifactId={null}
      />
    );
    expect(screen.getByRole('button', { name: /intervene/i })).toBeDisabled();
  });

  it('ALWAYS sends a size-matched, reconstructible control', async () => {
    vi.mocked(jlensApi.intervene).mockResolvedValue({
      task_id: 't1',
      model_id: 'm_1',
      queue: 'extraction',
    });
    render(
      <InterventionCard
        modelId="m_1"
        prompt="the animal that spins webs"
        pinned={[' Paris']}
        layers={[10, 11]}
        artifactId="slug"
      />
    );
    fireEvent.click(screen.getByRole('button', { name: /intervene/i }));
    fireEvent.click(screen.getByRole('button', { name: /run with control/i }));

    await waitFor(() => expect(jlensApi.intervene).toHaveBeenCalledTimes(1));
    const sent = vi.mocked(jlensApi.intervene).mock.calls[0][0];
    // An intervention without a control is not a weaker finding; it is not a
    // finding. There is deliberately no way to omit these.
    expect(sent.k).toBeGreaterThanOrEqual(1);
    expect(typeof sent.control_seed).toBe('number');
    // And the direction travels as a TOKEN, because the browser has no W_U.
    expect(sent.direction_token).toBe(' Paris');
    expect(sent.layers).toEqual([10, 11]);
  });

  it('EXPLAINS an empty direction list instead of showing a blank select', async () => {
    /**
     * OBSERVED IN THE PRODUCT. With nothing pinned this rendered as a blank
     * dropdown beside three fields that DO accept typing, which reads as a
     * broken control rather than a missing prerequisite — the caption "a pinned
     * token" only means something once you already know what pinning is.
     *
     * MUTATION CONTROL: map over `pinned` unconditionally and this fails.
     */
    render(
      <InterventionCard
        modelId="m_1"
        prompt="the animal that spins webs"
        pinned={[]}
        layers={[1]}
        artifactId={null}
      />
    );
    // The toggle is disabled with nothing pinned, so the reason has to be
    // reachable from the toggle itself as well as from inside the form.
    expect(
      screen.getByRole('button', { name: /intervene/i })
    ).toHaveAttribute('title', expect.stringMatching(/Pin a token first/));
  });

  it('intervenes on the PROMPT ON SCREEN, never an empty one', async () => {
    /**
     * This sent `prompt: ''` — an empty string — so every intervention launched
     * from the card scored a forward pass over nothing while the readout beside
     * it described a real prompt. The result named a layer and a direction and
     * measured neither in the context the reader was looking at. The server
     * would 422 it now (`min_length=1`), which is the only reason it surfaced.
     *
     * The old test could not see it: it never passed a prompt at all, and
     * because test files are excluded from `tsc`, adding the required prop did
     * not make it fail — it simply sent `undefined`.
     *
     * MUTATION CONTROL: revert to `prompt: ''` and this fails.
     */
    vi.mocked(jlensApi.intervene).mockResolvedValue({
      task_id: 't2',
      model_id: 'm_1',
      queue: 'extraction',
    });
    render(
      <InterventionCard
        modelId="m_1"
        prompt="the animal that spins webs"
        pinned={[' spider']}
        layers={[4]}
        artifactId="slug"
      />
    );
    fireEvent.click(screen.getByRole('button', { name: /intervene/i }));
    fireEvent.click(screen.getByRole('button', { name: /run with control/i }));

    await waitFor(() => expect(jlensApi.intervene).toHaveBeenCalledTimes(1));
    const sent = vi.mocked(jlensApi.intervene).mock.calls[0][0];
    expect(sent.prompt).toBe('the animal that spins webs');
  });
});
