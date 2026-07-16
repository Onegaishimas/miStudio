/**
 * Tests for ClusterBudgetBar (Feature 013).
 *
 * The bar is the budget-model's visibility surface: it must show consumption
 * against B, surface coherence flags, warn when over budget, and — when the
 * model is gated (low cohesion) — show ONLY the notice, never a budget.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ClusterBudgetBar } from './ClusterBudgetBar';
import { useSteeringStore } from '../../stores/steeringStore';

const budget = {
  B: 2.4,
  B_dir: 2.4,
  G: 0.87,
  flags: [] as string[],
  approximate: false,
  weightsByInstance: {} as Record<string, number>,
};

const feature = (idx: number, strength: number) =>
  ({
    feature_idx: idx,
    layer: 6,
    strength,
    color: 'teal',
    instance_id: `i-${idx}`,
    strengthSource: 'cluster',
  }) as never;

describe('ClusterBudgetBar', () => {
  beforeEach(() => {
    useSteeringStore.setState({
      clusterBudget: null,
      clusterNotice: null,
      selectedFeatures: [],
    });
  });

  it('renders nothing without a budget or notice', () => {
    const { container } = render(<ClusterBudgetBar />);
    expect(container).toBeEmptyDOMElement();
  });

  it('shows consumption / B and the gain G', () => {
    useSteeringStore.setState({
      clusterBudget: budget,
      selectedFeatures: [feature(1, 1.2), feature(2, -0.6)],
    });
    render(<ClusterBudgetBar />);
    // |1.2| + |−0.6| = 1.8 of 2.4
    expect(screen.getByText(/1\.8 \/ 2\.4/)).toBeInTheDocument();
    expect(screen.getByText(/G 0\.87/)).toBeInTheDocument();
  });

  it('turns amber when over budget', () => {
    useSteeringStore.setState({
      clusterBudget: budget,
      selectedFeatures: [feature(1, 2.0), feature(2, 1.0)],
    });
    render(<ClusterBudgetBar />);
    expect(screen.getByText(/3\.0 \/ 2\.4/)).toHaveClass('text-amber-400');
  });

  it('surfaces coherence flags with their copy', () => {
    useSteeringStore.setState({
      clusterBudget: { ...budget, flags: ['cancellation', 'approximate'] },
      selectedFeatures: [feature(1, 1.0)],
    });
    render(<ClusterBudgetBar />);
    expect(screen.getByText(/partially cancel/)).toBeInTheDocument();
    expect(screen.getByText(/constant-budget approximation/)).toBeInTheDocument();
  });

  it('ignores unknown flags from a future server', () => {
    useSteeringStore.setState({
      clusterBudget: { ...budget, flags: ['grain_limited', 'brand_new_flag'] },
      selectedFeatures: [feature(1, 1.0)],
    });
    render(<ClusterBudgetBar />);
    expect(screen.queryByText(/brand_new_flag/)).not.toBeInTheDocument();
  });

  it('gated cluster: renders the notice only, no budget numbers', () => {
    useSteeringStore.setState({
      clusterBudget: null,
      clusterNotice: 'Low cluster cohesion — kept per-feature baselines (budget model gated)',
      selectedFeatures: [feature(1, 10)],
    });
    render(<ClusterBudgetBar />);
    expect(screen.getByText(/Low cluster cohesion/)).toBeInTheDocument();
    expect(screen.queryByText(/Cluster budget/)).not.toBeInTheDocument();
  });
});
