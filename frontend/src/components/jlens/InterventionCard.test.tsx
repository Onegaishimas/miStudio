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
      <InterventionCard modelId="m_1" pinned={[]} layers={[1]} artifactId={null} />
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
});
