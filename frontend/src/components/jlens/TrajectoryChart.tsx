/**
 * Rank-vs-layer trajectories for the pinned tokens at one position.
 *
 * Three details are requirements rather than styling:
 *  - the Y axis is REVERSED, because rank 1 is the strongest reading;
 *  - the domain is [1, meta.top_n] — not a constant, since the server decides
 *    how deep the readout goes;
 *  - `connectNulls={false}`, because a layer where the token left the top-k is
 *    a gap in the evidence, and bridging it draws a trajectory that was never
 *    measured.
 *
 * Band shading appears only when a BandReport exists (BR-002).
 */

import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { PIN_COLORS } from './utils';
import { rankOf } from '../../stores/jlensStore';
import type { BandReport, LensTypeSlice } from '../../types/jlens';

interface TrajectoryChartProps {
  axis: number[];
  slice: LensTypeSlice | undefined;
  pinned: string[];
  topN: number;
  selPos: number;
  bandReport: BandReport | null;
}

export function TrajectoryChart({
  axis,
  slice,
  pinned,
  topN,
  selPos,
  bandReport,
}: TrajectoryChartProps) {
  /**
   * Series are keyed by an ALIAS, never by the token text.
   *
   * Each chart row is a flat object that also carries the x-axis value under
   * `layer`. Token text goes into the SAME namespace, so pinning the token
   * "layer" — an ordinary English word, and a token in every vocabulary here —
   * overwrites the x-axis value with a rank. The chart then plots every point
   * at x = its own rank, which is wrong in a way that still looks like a chart.
   * Aliasing keeps the two namespaces apart for any token at all; the token
   * text still reaches the tooltip via `name`.
   */
  const series = pinned.map((token, i) => ({ token, key: `s${i}` }));

  const data = axis.map((layer, i) => {
    const row: Record<string, number | null> = { layer };
    for (const { token, key } of series) {
      row[key] = rankOf(slice, i, token);
    }
    return row;
  });

  return (
    <div>
      <div className="mb-2 flex items-center justify-between">
        <span className="text-xs font-medium text-slate-600 dark:text-slate-400">
          Rank across layers · position {selPos}
        </span>
        <span className="text-[10px] text-slate-500 dark:text-slate-500">
          lower is stronger · gaps are layers where the token left the top-{topN}
        </span>
      </div>
      <div className="h-52">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: -18 }}>
            <CartesianGrid stroke="#334155" strokeDasharray="2 4" />
            {bandReport && (
              <ReferenceArea
                x1={bandReport.workspace_start}
                x2={bandReport.motor_start}
                fill="#34d399"
                fillOpacity={0.07}
              />
            )}
            <XAxis
              dataKey="layer"
              // NUMERIC, not the default category axis. A band ReferenceArea is
              // positioned by layer NUMBER, and on a category axis those
              // numbers are matched against category labels — which lands the
              // shading in the wrong place, or nowhere, on any sparse axis.
              type="number"
              domain={['dataMin', 'dataMax']}
              allowDecimals={false}
              stroke="#64748b"
              tick={{ fontSize: 10 }}
              label={{
                value: 'layer',
                position: 'insideBottom',
                offset: -2,
                fill: '#64748b',
                fontSize: 10,
              }}
            />
            <YAxis
              reversed
              domain={[1, topN]}
              stroke="#64748b"
              tick={{ fontSize: 10 }}
              allowDecimals={false}
            />
            <Tooltip
              contentStyle={{
                background: '#0f172a',
                border: '1px solid #334155',
                borderRadius: 6,
                fontSize: 11,
              }}
              labelStyle={{ color: '#94a3b8' }}
            />
            {series.map(({ token, key }, i) => (
              <Line
                key={key}
                type="monotone"
                dataKey={key}
                name={token}
                stroke={PIN_COLORS[i % PIN_COLORS.length]}
                strokeWidth={2}
                dot={false}
                connectNulls={false}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
