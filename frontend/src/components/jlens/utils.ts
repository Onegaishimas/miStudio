/**
 * Shared rendering helpers for the J-Lens panel.
 *
 * Nothing here encodes a model property. In particular there is NO band
 * constant: the reference implementation's `BAND = { workspaceStart: 40,
 * motorStart: 90 }` are the source paper's Sonnet-4.5 figures, and BR-002
 * requires the product make porting them impossible by construction. Band
 * geometry arrives as a BandReport or not at all.
 */

/** Make whitespace visible without changing the token's identity. */
export function displayToken(token: string): string {
  return token
    .replace(/^ /, '·')
    .replace(/\n/g, '⏎')
    .replace(/\t/g, '→');
}

/**
 * Emerald ramp keyed to rank, scaled by the top-n the SERVER actually sent.
 *
 * `topN` is a required parameter rather than a module constant on purpose: the
 * reference implementation hardcodes 8, which mis-scales the whole heatmap the
 * moment a readout comes back with a different top-n — legibly, and wrongly.
 */
export function rankColor(rank: number | null, topN: number): string {
  if (rank == null) return 'transparent';
  const alpha = Math.max(0.06, 1 - Math.log(rank) / Math.log(Math.max(topN, 2) + 6));
  return `rgba(52, 211, 153, ${alpha.toFixed(3)})`;
}

/**
 * Top-1 probability below which a readout is DIFFUSE — the top token carries so
 * little of the distribution that reading it as content is unsafe.
 *
 * This is a statement about the readout's own distribution, measured per cell.
 * It is explicitly NOT a layer boundary and asserts nothing about which layers
 * are sensory, workspace or motor; those come from a band report or not at all.
 * Diffuse readouts cluster in early layers, which is exactly the expectation
 * FPRD §3.7 requires the panel to surface rather than hide.
 */
export const DIFFUSE_TOP1_PROB = 0.1;

export function isDiffuse(topProb: number | undefined): boolean {
  return topProb === undefined || topProb < DIFFUSE_TOP1_PROB;
}

/** Stable colours for pinned-token series, shared by the grid legend and chart. */
export const PIN_COLORS = [
  '#34d399',
  '#60a5fa',
  '#f59e0b',
  '#f472b6',
  '#a78bfa',
  '#2dd4bf',
];
