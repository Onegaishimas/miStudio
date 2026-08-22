/**
 * The two icon-only actions on a completed extraction had no visible label.
 *
 * A bare chevron and a bare circular arrow sat next to a green "Label Features"
 * button, and the only way to learn what either did was to hover and wait for a
 * tooltip. One opens the feature browser; the other DISCARDS all NLP analysis
 * and recomputes it. Those carry very different consequences and looked
 * identical in weight.
 *
 * Both now carry text, and each is tinted to the thing it acts on — emerald for
 * features, cyan for NLP, matching the count and the badge already on the card.
 *
 * MUTATION CONTROLS:
 *   * remove the <span> from either button      -> its label test fails
 *   * drop the feature count from the expand label -> count test fails
 *   * swap the rotating chevron back to two icons  -> rotation test fails
 *   * drop aria-expanded                        -> a11y test fails
 */

import { describe, it, expect, vi, afterEach } from 'vitest';
import { screen, fireEvent } from '@testing-library/react';
import { renderWithProviders as render } from '../../test/renderWithProviders';
import { ExtractionJobCard } from './ExtractionJobCard';

vi.mock('../../api/models', () => ({
  triggerNlpAnalysis: vi.fn(),
  cancelNlpAnalysis: vi.fn(),
  resetNlpAnalysis: vi.fn(),
}));

// Expanding fetches the feature list. Left unmocked it fires a real XHR that
// rejects after the test has finished, which surfaces as an unhandled error and
// makes a passing suite look broken.
vi.mock('axios', () => ({
  default: {
    get: vi.fn().mockResolvedValue({ data: { features: [], total: 0 } }),
    post: vi.fn().mockResolvedValue({ data: {} }),
    delete: vi.fn().mockResolvedValue({ data: {} }),
  },
}));

const completed = {
  id: 'extr_1',
  status: 'completed',
  progress: 1,
  features_extracted: 32766,
  total_features: 32766,
  sae_name: 'SAE from granite-4.1-8b (L36-residual)',
  created_at: new Date().toISOString(),
  config: {},
  nlp_status: 'completed',
  statistics: { total_features: 32766, interpretable_count: 22478 },
};

function renderCard(overrides: Record<string, unknown> = {}) {
  return render(
    <ExtractionJobCard
      extraction={{ ...completed, ...overrides } as any}
      onDelete={vi.fn()}
      onCancel={vi.fn()}
    />,
  );
}

describe('ExtractionJobCard action labels', () => {
  afterEach(() => vi.clearAllMocks());

  it('says what the expand button opens, and how much is in it', () => {
    renderCard();

    // The count is the clearest possible statement of what expanding reveals.
    expect(screen.getByText('Browse 32,766 features')).toBeInTheDocument();
  });

  it('falls back to a plain label when the count is unknown', () => {
    renderCard({ statistics: undefined });

    expect(screen.getByText('Browse features')).toBeInTheDocument();
  });

  it('switches to a hide label once open', () => {
    renderCard();

    fireEvent.click(screen.getByText('Browse 32,766 features'));

    expect(screen.getByText('Hide features')).toBeInTheDocument();
  });

  it('reports its open state to assistive tech', () => {
    renderCard();
    const button = screen.getByText('Browse 32,766 features').closest('button')!;

    expect(button).toHaveAttribute('aria-expanded', 'false');
    fireEvent.click(button);
    expect(
      screen.getByText('Hide features').closest('button'),
    ).toHaveAttribute('aria-expanded', 'true');
  });

  it('names the NLP action instead of leaving a bare glyph', () => {
    renderCard();

    expect(screen.getByText('Re-run NLP')).toBeInTheDocument();
  });

  it('warns in the NLP tooltip that existing analysis is discarded', () => {
    renderCard();
    const button = screen.getByText('Re-run NLP').closest('button')!;

    // The consequence belongs somewhere the user can find BEFORE clicking, not
    // only in the confirm() that follows.
    expect(button.getAttribute('title')).toMatch(/discard/i);
  });

  it('uses ONE chevron that rotates, not two that swap', () => {
    const { container } = renderCard();
    const button = screen.getByText('Browse 32,766 features').closest('button')!;

    const icon = button.querySelector('svg')!;
    expect(icon.getAttribute('class')).toContain('transition-transform');
    expect(icon.getAttribute('class')).not.toContain('rotate-180');

    fireEvent.click(button);
    const openIcon = screen
      .getByText('Hide features')
      .closest('button')!
      .querySelector('svg')!;
    expect(openIcon.getAttribute('class')).toContain('rotate-180');
    expect(container).toBeTruthy();
  });
});
