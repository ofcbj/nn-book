import { useMemo, useState } from 'react';
import { Box, Paper, Typography } from '@mui/material';
import { useTranslation } from 'react-i18next';
import type { LossPoint } from '../lib/types';

interface LossChartProps {
  history: LossPoint[];
  /** Panel title (defaults to "Loss over Epochs") */
  title?: string;
  /** Label for the x axis unit (defaults to "Epoch") */
  xLabel?: string;
  /** Plot a trailing moving average of this many points instead of the raw series */
  smoothingWindow?: number;
}

// Chart geometry in SVG user units; the SVG scales to the panel width.
const WIDTH = 248;
const HEIGHT = 120;
const PAD = { top: 10, right: 12, bottom: 18, left: 34 };
const PLOT_W = WIDTH - PAD.left - PAD.right;
const PLOT_H = HEIGHT - PAD.top - PAD.bottom;
/** Points beyond this are min/max-bucketed so spikes survive downsampling */
const MAX_DRAWN_POINTS = 240;

const LINE_COLOR = '#60a5fa';
const GRID_COLOR = 'rgba(148, 163, 184, 0.18)';
const TEXT_COLOR = '#94a3b8';

function formatLoss(v: number): string {
  return v >= 10 ? v.toFixed(1) : v >= 1 ? v.toFixed(2) : v.toFixed(3);
}

/** Trailing moving average; each point averages the previous `window` losses */
function smooth(points: LossPoint[], window: number): LossPoint[] {
  if (window <= 1) return points;
  const out: LossPoint[] = [];
  let sum = 0;
  for (let i = 0; i < points.length; i++) {
    sum += points[i].loss;
    if (i >= window) sum -= points[i - window].loss;
    out.push({ epoch: points[i].epoch, loss: sum / Math.min(i + 1, window) });
  }
  return out;
}

/** Reduce a long series to at most MAX_DRAWN_POINTS while keeping each bucket's min and max */
function downsample(points: LossPoint[]): LossPoint[] {
  if (points.length <= MAX_DRAWN_POINTS) return points;
  const bucketSize = Math.ceil(points.length / (MAX_DRAWN_POINTS / 2));
  const out: LossPoint[] = [];
  for (let i = 0; i < points.length; i += bucketSize) {
    const bucket = points.slice(i, i + bucketSize);
    let min = bucket[0], max = bucket[0];
    for (const p of bucket) {
      if (p.loss < min.loss) min = p;
      if (p.loss > max.loss) max = p;
    }
    out.push(...(min.epoch <= max.epoch ? [min, max] : [max, min]).filter((p, k, arr) => k === 0 || p !== arr[0]));
  }
  return out;
}

export default function LossChart({ history, title, xLabel, smoothingWindow = 1 }: LossChartProps) {
  const { t } = useTranslation();
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const chartTitle = title ?? t('lossChart.title');
  const unit = xLabel ?? t('stats.epoch');

  const smoothed = useMemo(() => smooth(history, smoothingWindow), [history, smoothingWindow]);
  const points = useMemo(() => downsample(smoothed), [smoothed]);

  const scale = useMemo(() => {
    const maxLoss = points.reduce((m, p) => Math.max(m, p.loss), 0);
    const yMax = maxLoss > 0 ? maxLoss * 1.05 : 1;
    const firstEpoch = points[0]?.epoch ?? 0;
    const lastEpoch = points[points.length - 1]?.epoch ?? firstEpoch;
    const epochSpan = Math.max(lastEpoch - firstEpoch, 1);
    return {
      yMax,
      x: (epoch: number) => PAD.left + ((epoch - firstEpoch) / epochSpan) * PLOT_W,
      y: (loss: number) => PAD.top + PLOT_H - (loss / yMax) * PLOT_H,
      firstEpoch,
      lastEpoch,
    };
  }, [points]);

  const path = useMemo(() => {
    if (points.length === 0) return '';
    return points
      .map((p, i) => `${i === 0 ? 'M' : 'L'}${scale.x(p.epoch).toFixed(1)},${scale.y(p.loss).toFixed(1)}`)
      .join(' ');
  }, [points, scale]);

  const last = smoothed[smoothed.length - 1];
  const minLoss = useMemo(() => smoothed.reduce((m, p) => Math.min(m, p.loss), Infinity), [smoothed]);
  const hovered = hoverIndex !== null ? points[hoverIndex] : null;

  const handleMove = (event: React.MouseEvent<SVGSVGElement>) => {
    if (points.length === 0) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const svgX = ((event.clientX - rect.left) / rect.width) * WIDTH;
    let nearest = 0;
    let best = Infinity;
    points.forEach((p, i) => {
      const d = Math.abs(scale.x(p.epoch) - svgX);
      if (d < best) { best = d; nearest = i; }
    });
    setHoverIndex(nearest);
  };

  return (
    <Paper sx={{ p: 2, bgcolor: '#0f172a', borderRadius: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', mb: 1 }}>
        <Typography variant="h3" sx={{ fontSize: '0.95rem' }}>
          📉 {chartTitle}
        </Typography>
        {history.length > 0 && (
          <Typography variant="caption" sx={{ color: 'text.secondary', fontFamily: 'monospace' }}>
            {t('lossChart.min')} {formatLoss(minLoss)}
          </Typography>
        )}
      </Box>

      {history.length === 0 ? (
        <Typography variant="body2" color="text.secondary" sx={{ py: 3, textAlign: 'center' }}>
          {t('lossChart.empty')}
        </Typography>
      ) : (
        <svg
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          width="100%"
          role="img"
          aria-label={`${chartTitle}: ${history.length} ${unit}`}
          style={{ display: 'block', cursor: 'crosshair' }}
          onMouseMove={handleMove}
          onMouseLeave={() => setHoverIndex(null)}
        >
          {/* Recessive grid: top, middle, baseline */}
          {[0, 0.5, 1].map(f => {
            const y = PAD.top + PLOT_H * f;
            return <line key={f} x1={PAD.left} x2={WIDTH - PAD.right} y1={y} y2={y} stroke={GRID_COLOR} strokeWidth={1} />;
          })}

          {/* Y axis labels */}
          <text x={PAD.left - 4} y={PAD.top + 4} fill={TEXT_COLOR} fontSize={9} fontFamily="monospace" textAnchor="end">
            {formatLoss(scale.yMax)}
          </text>
          <text x={PAD.left - 4} y={PAD.top + PLOT_H} fill={TEXT_COLOR} fontSize={9} fontFamily="monospace" textAnchor="end">
            0
          </text>

          {/* X axis labels: first and last epoch */}
          <text x={PAD.left} y={HEIGHT - 4} fill={TEXT_COLOR} fontSize={9} fontFamily="monospace">
            {unit} {scale.firstEpoch}
          </text>
          <text x={WIDTH - PAD.right} y={HEIGHT - 4} fill={TEXT_COLOR} fontSize={9} fontFamily="monospace" textAnchor="end">
            {scale.lastEpoch}
          </text>

          {/* Series */}
          <path d={path} fill="none" stroke={LINE_COLOR} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />

          {/* Direct label on the latest value (only when not hovering) */}
          {last && !hovered && (
            <>
              <circle cx={scale.x(last.epoch)} cy={scale.y(last.loss)} r={3} fill={LINE_COLOR} stroke="#0f172a" strokeWidth={2} />
              <text
                x={Math.min(scale.x(last.epoch) + 5, WIDTH - PAD.right)}
                y={Math.max(scale.y(last.loss) - 6, PAD.top + 8)}
                fill="#e2e8f0" fontSize={10} fontWeight={700} fontFamily="monospace" textAnchor="end"
              >
                {formatLoss(last.loss)}
              </text>
            </>
          )}

          {/* Hover crosshair + tooltip */}
          {hovered && (() => {
            const hx = scale.x(hovered.epoch);
            const hy = scale.y(hovered.loss);
            const label = `${unit} ${hovered.epoch}: ${formatLoss(hovered.loss)}`;
            const labelW = label.length * 6 + 8;
            const labelX = Math.min(Math.max(hx - labelW / 2, PAD.left), WIDTH - PAD.right - labelW);
            return (
              <>
                <line x1={hx} x2={hx} y1={PAD.top} y2={PAD.top + PLOT_H} stroke="rgba(226, 232, 240, 0.35)" strokeWidth={1} />
                <circle cx={hx} cy={hy} r={4} fill={LINE_COLOR} stroke="#0f172a" strokeWidth={2} />
                <rect x={labelX} y={PAD.top - 8} width={labelW} height={14} rx={3} fill="rgba(15, 23, 42, 0.95)" stroke="rgba(148, 163, 184, 0.4)" />
                <text x={labelX + labelW / 2} y={PAD.top + 3} fill="#e2e8f0" fontSize={9} fontFamily="monospace" textAnchor="middle">
                  {label}
                </text>
              </>
            );
          })()}
        </svg>
      )}
    </Paper>
  );
}
