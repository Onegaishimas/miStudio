/**
 * The two lenses ranked side by side, with Swap and Steer on the token.
 *
 * MUTATION CONTROLS (each must turn this file red):
 *   * drop the range/top-n from the header      -> "states its denominator" fails
 *   * hide non-words without saying how many    -> "says how many it hid" fails
 *   * enable Swap with no reason supplied       -> "Swap is disabled WITH a reason" fails
 *   * hand Steer the whole axis                 -> "Steer carries the token's OWN layers" fails
 */
import { describe, expect, it, vi } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { RankedReadouts } from './RankedReadouts';

function tok(position: number, rows: Record<string, string[][]>) {
  return {
    kind: 'token' as const,
    position,
    token: `t${position}`,
    id: position,
    is_generated: false,
    results: Object.entries(rows).map(([type, top_tokens]) => ({
      type,
      top_tokens,
      top_probs: top_tokens.map((r) => r.map(() => 0.1)),
    })),
  };
}

/** Axes are passed explicitly so no test can agree with a hardcoded default. */
const AXES = { JACOBIAN_LENS: [8, 9, 10], LOGIT_LENS: [8, 9, 10] };

function base(overrides = {}) {
  return {
    tokens: [
      tok(0, {
        JACOBIAN_LENS: [['dog'], ['dog', 'cat'], ['cat']],
        LOGIT_LENS: [['a'], ['the'], ['an']],
      }),
    ],
    axes: AXES,
    types: ['JACOBIAN_LENS', 'LOGIT_LENS'],
    topN: 8,
    range: null as [number, number] | null,
    hideNonWords: true,
    onToggleNonWords: vi.fn(),
    ...overrides,
  };
}

describe('RankedReadouts', () => {
  it('renders BOTH lenses as separate columns', () => {
    render(<RankedReadouts {...base()} />);
    expect(screen.getByTestId('ranked-JACOBIAN_LENS')).toBeInTheDocument();
    expect(screen.getByTestId('ranked-LOGIT_LENS')).toBeInTheDocument();
  });

  it('states EACH COLUMN’s own denominator, because the axes differ', () => {
    /**
     * Layer axes are per lens type. A partial Jacobian artifact covers fewer
     * layers than the logit lens, which needs no artifact at all — on LFM2 the
     * Jacobian is 15 of 16. One shared span therefore misstates the second
     * column's counts.
     *
     * THE FIXTURE GIVES THE TWO LENSES DIFFERENT AXES ON PURPOSE. An earlier
     * version gave both the same axis, so a single shared denominator agreed
     * with both by construction and the bug was invisible.
     *
     * MUTATION CONTROL: compute the span from `columns[0]` and print it once —
     * this fails on the Logit column.
     */
    const axes = { JACOBIAN_LENS: [8, 9], LOGIT_LENS: [8, 9, 10] };
    const tokens = [
      tok(0, {
        JACOBIAN_LENS: [['dog'], ['cat']],
        LOGIT_LENS: [['a'], ['the'], ['an']],
      }),
    ];
    render(<RankedReadouts {...base({ axes, tokens })} />);

    expect(screen.getByTestId('ranked-span-JACOBIAN_LENS')).toHaveTextContent(
      '2 layers, L8–L9',
    );
    expect(screen.getByTestId('ranked-span-LOGIT_LENS')).toHaveTextContent(
      '3 layers, L8–L10',
    );
  });

  it('does not describe a NON-CONTIGUOUS axis as a range', () => {
    /**
     * `L8–L20` for [8, 12, 20] implies thirteen contributing layers when three
     * contributed, and the count beside it is then read against a denominator
     * four times too large. A partial fit produces exactly that shape.
     *
     * MUTATION CONTROL: render `L{first}–L{last}` unconditionally and this
     * fails.
     */
    const axes = { JACOBIAN_LENS: [8, 12, 20] };
    const tokens = [tok(0, { JACOBIAN_LENS: [['dog'], ['cat'], ['pet']] })];
    render(
      <RankedReadouts {...base({ axes, tokens, types: ['JACOBIAN_LENS'] })} />,
    );
    const span = screen.getByTestId('ranked-span-JACOBIAN_LENS');
    expect(span).toHaveTextContent('3 layers');
    expect(span).not.toHaveTextContent('L8–L20');
  });

  it('says when it has CAPPED the list rather than implying it showed everything', () => {
    /**
     * Each row draws one bar per in-range layer, so an uncapped list is
     * O(tokens x layers) DOM nodes. A cap is right; a cap the reader cannot see
     * is a ranked list quietly claiming to be complete.
     *
     * MUTATION CONTROL: drop the "showing top N of M" line and this fails.
     */
    const many = Array.from({ length: 80 }, (_, i) => `w${i}`);
    const tokens = [tok(0, { JACOBIAN_LENS: [many] })];
    render(
      <RankedReadouts
        {...base({ axes: { JACOBIAN_LENS: [8] }, tokens, types: ['JACOBIAN_LENS'] })}
      />,
    );
    expect(screen.getByText(/Showing the top 60 of 80 tokens/)).toBeInTheDocument();
  });

  it('counts only the SELECTED range', () => {
    render(<RankedReadouts {...base({ range: [10, 10] })} />);
    const col = screen.getByTestId('ranked-JACOBIAN_LENS');
    // At L10 only 'cat' appears; 'dog' lives at L8 and L9.
    expect(within(col).getByText('cat')).toBeInTheDocument();
    expect(within(col).queryByText('dog')).not.toBeInTheDocument();
  });

  it('says how many non-words it hid rather than hiding them silently', () => {
    // MUTATION CONTROL: stop reporting the count and this fails.
    const tokens = [tok(0, { JACOBIAN_LENS: [['dog', '^(@)', '$.'], [], []] })];
    render(
      <RankedReadouts
        {...base({ tokens, types: ['JACOBIAN_LENS'], hideNonWords: true })}
      />,
    );
    expect(screen.getByText(/2 non-words hidden/)).toBeInTheDocument();
  });

  it('Steer carries the token’s OWN layers, not the whole axis', async () => {
    /**
     * Handing the intervention every layer is what the standalone card did, and
     * it produces a result describing an intervention at every layer at once.
     *
     * MUTATION CONTROL: pass the full axis and this fails.
     */
    const onSteer = vi.fn();
    render(<RankedReadouts {...base({ onSteer, types: ['JACOBIAN_LENS'] })} />);
    const col = screen.getByTestId('ranked-JACOBIAN_LENS');
    await userEvent.click(within(col).getAllByTitle(/Steer along dog/)[0]);
    expect(onSteer).toHaveBeenCalledWith('dog', [8, 9]);
  });

  it('Swap is disabled WITH A STATED REASON, PER TOKEN', async () => {
    /**
     * A swap needs two coordinates. Disabling it silently leaves the reader to
     * guess; inventing the second token would be a guess of our own.
     *
     * PER TOKEN, not per column: availability depends on there being a
     * DIFFERENT token to exchange with, so with one token pinned every other
     * row can swap and the pinned row itself cannot.
     *
     * MUTATION CONTROL: enable it regardless and this fails.
     */
    const onSwap = vi.fn();
    render(
      <RankedReadouts
        {...base({
          onSwap,
          types: ['JACOBIAN_LENS'],
          swapDisabledFor: () => 'Pin a second token to swap with.',
        })}
      />,
    );
    const col = screen.getByTestId('ranked-JACOBIAN_LENS');
    const swap = within(col).getAllByTitle('Pin a second token to swap with.')[0];
    expect(swap).toBeDisabled();
    await userEvent.click(swap);
    expect(onSwap).not.toHaveBeenCalled();
  });

  it('Swap fires with the token and its layers when it is available', async () => {
    const onSwap = vi.fn();
    render(<RankedReadouts {...base({ onSwap, types: ['JACOBIAN_LENS'] })} />);
    const col = screen.getByTestId('ranked-JACOBIAN_LENS');
    await userEvent.click(within(col).getAllByTitle(/Swap dog/)[0]);
    expect(onSwap).toHaveBeenCalledWith('dog', [8, 9]);
  });

  it('blocks ONLY the token that has no partner', async () => {
    /**
     * With one token pinned, that row alone cannot swap — every other row can
     * exchange with it. A column-wide guard gets exactly this case wrong, and
     * it is the case that queued a request with no partner.
     *
     * MUTATION CONTROL: return one reason for the whole column and this fails.
     */
    const tokens = [tok(0, { JACOBIAN_LENS: [['dog'], ['cat'], ['cat']] })];
    render(
      <RankedReadouts
        {...base({
          tokens,
          types: ['JACOBIAN_LENS'],
          swapDisabledFor: (t: string) =>
            t === 'dog' ? 'Pin a token other than dog — a swap needs two.' : undefined,
        })}
      />,
    );
    const col = screen.getByTestId('ranked-JACOBIAN_LENS');
    expect(within(col).getByTitle(/other than dog/)).toBeDisabled();
    expect(within(col).getByTitle(/^Swap cat/)).toBeEnabled();
  });

  it('renders nothing rather than an empty frame when there are no types', () => {
    const { container } = render(<RankedReadouts {...base({ types: [] })} />);
    expect(container.firstChild).toBeNull();
  });
});
