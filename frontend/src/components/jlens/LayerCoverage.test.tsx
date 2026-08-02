/**
 * Coverage must be visible, and UNKNOWN must not read as ZERO.
 *
 * A 9-of-16-layer LFM2 lens was indistinguishable from a full one everywhere
 * in the product until a readout refused at layer 0. The information was
 * available the whole time.
 *
 * MUTATION CONTROLS:
 *   * render an empty strip for unknown coverage -> "unknown" fails
 *   * missingLayers returns the covered set      -> "complement" fails
 */

import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { LayerCoverage, missingLayers } from './LayerCoverage';

describe('LayerCoverage', () => {
  it('shows how many of the model’s layers are fitted', () => {
    render(<LayerCoverage covered={[1, 2, 3, 10, 11]} total={16} />);
    expect(screen.getByText('5/16 layers')).toBeInTheDocument();
    expect(
      screen.getByLabelText('covers 5 of 16 layers')
    ).toBeInTheDocument();
  });

  it('says coverage is NOT RECORDED rather than drawing zero', () => {
    // An artifact whose config could not be read still holds whatever it
    // holds. An empty strip would assert coverage nobody checked.
    render(<LayerCoverage covered={[]} total={16} />);
    expect(screen.getByText(/coverage not recorded/i)).toBeInTheDocument();
    expect(screen.queryByText('0/16 layers')).toBeNull();
  });

  it('says so when the model’s dimensions are unknown', () => {
    render(<LayerCoverage covered={[1]} total={null} />);
    expect(screen.getByText(/needs the model's dimensions/i)).toBeInTheDocument();
  });
});

describe('missingLayers', () => {
  it('returns the complement, in order', () => {
    expect(missingLayers([1, 2, 3, 10, 11, 12, 13, 14, 15], 16)).toEqual([
      0, 4, 5, 6, 7, 8, 9,
    ]);
  });

  it('is empty for a full artifact', () => {
    expect(missingLayers([0, 1, 2], 3)).toEqual([]);
  });
});
