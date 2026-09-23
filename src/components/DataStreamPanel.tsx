import { useEffect, useRef } from 'react';
import { Box, Paper, Typography, IconButton, Button, ToggleButton, ToggleButtonGroup, LinearProgress, Tooltip, Stack } from '@mui/material';
import { useTranslation } from 'react-i18next';
import type { UseDatasetTrainingReturn, StreamSpeed } from '../hooks/useDatasetTraining';
import { STREAM_SPEEDS } from '../hooks/useDatasetTraining';
import type { Candidate } from '../lib/core';
import type { Inspection } from '../hooks/useNeuralNetwork';

interface DataStreamPanelProps {
  stream: UseDatasetTrainingReturn;
  /** Candidate whose training step is currently animated on the network, if any */
  inspection: Inspection | null;
  /** Row click: animate one training step on that candidate */
  onInspect: (candidate: Candidate) => void;
  onToggleInspection: () => void;
  onStopInspection: () => void;
  /** Height of the scrolling list area */
  listHeight?: number | string;
}

const CLASS_COLORS = ['#f87171', '#fb923c', '#34d399'] as const;

function percent(numerator: number, denominator: number): string {
  return denominator > 0 ? `${Math.round((numerator / denominator) * 100)}%` : '—';
}

export default function DataStreamPanel({
  stream,
  inspection,
  onInspect,
  onToggleInspection,
  onStopInspection,
  listHeight = 'calc(100vh - 560px)',
}: DataStreamPanelProps) {
  const { t } = useTranslation();
  const classNames = [t('classes.fail'), t('classes.pending'), t('classes.pass')];
  const listRef = useRef<HTMLDivElement>(null);
  const { dataset, results, lastIndex, cursor, pass, running, speed, evaluation, processedTotal, onlineCorrect } = stream;

  // Keep the sample that just went through in view
  useEffect(() => {
    if (lastIndex === null || !listRef.current) return;
    const row = listRef.current.children[lastIndex] as HTMLElement | undefined;
    row?.scrollIntoView({ block: 'nearest' });
  }, [lastIndex]);

  const overallAccuracy = evaluation ? evaluation.accuracy : 0;
  const inspecting = inspection !== null;

  return (
    <Paper sx={{ p: 2, bgcolor: '#0f172a', display: 'flex', flexDirection: 'column', gap: 1.5 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
        <Typography variant="h3" sx={{ fontSize: '0.95rem' }}>
          🗂️ {t('dataset.title')}
        </Typography>
        <Typography variant="caption" sx={{ color: 'text.secondary', fontFamily: 'monospace' }}>
          {t('dataset.pass')} {pass} · {Math.min(cursor, dataset.length)}/{dataset.length}
        </Typography>
      </Box>

      {/* Accuracy */}
      <Box>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
          <Typography variant="body2" color="text.secondary">{t('dataset.overallAccuracy', { count: dataset.length })}</Typography>
          <Typography variant="body2" sx={{ fontFamily: 'monospace', fontWeight: 700, color: overallAccuracy >= 0.95 ? 'secondary.light' : 'primary.light' }}>
            {evaluation ? `${evaluation.correctCount}/${evaluation.total} (${percent(evaluation.correctCount, evaluation.total)})` : '—'}
          </Typography>
        </Box>
        <LinearProgress
          variant="determinate"
          value={overallAccuracy * 100}
          sx={{ height: 8, borderRadius: 1, bgcolor: 'rgba(59, 130, 246, 0.15)', '& .MuiLinearProgress-bar': { bgcolor: overallAccuracy >= 0.95 ? 'secondary.main' : 'primary.main', transition: 'transform 0.2s' } }}
        />
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
          <Tooltip title={t('dataset.onlineAccuracyHint')}>
            <Typography variant="body2" color="text.secondary" sx={{ cursor: 'help', textDecoration: 'underline dotted' }}>
              {t('dataset.onlineAccuracy')}
            </Typography>
          </Tooltip>
          <Typography variant="body2" sx={{ fontFamily: 'monospace', fontWeight: 700 }}>
            {processedTotal > 0 ? `${onlineCorrect}/${processedTotal} (${percent(onlineCorrect, processedTotal)})` : '—'}
          </Typography>
        </Box>
      </Box>

      {/* Controls */}
      <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
        <Tooltip title={running ? t('dataset.pause') : t('dataset.play')}>
          <span>
            <IconButton onClick={stream.toggle} disabled={inspecting} aria-label={running ? t('dataset.pause') : t('dataset.play')} sx={{ bgcolor: running ? 'warning.main' : 'primary.main', color: 'white', '&:hover': { bgcolor: running ? 'warning.dark' : 'primary.dark' }, '&.Mui-disabled': { bgcolor: 'rgba(148, 163, 184, 0.2)' } }}>
              {running ? '⏸' : '▶'}
            </IconButton>
          </span>
        </Tooltip>
        <Tooltip title={t('dataset.step')}>
          <span>
            <IconButton onClick={stream.step} disabled={running || inspecting} aria-label={t('dataset.step')} sx={{ bgcolor: 'rgba(59, 130, 246, 0.2)', color: 'white' }}>
              ⏭
            </IconButton>
          </span>
        </Tooltip>
        <Tooltip title={t('dataset.restart')}>
          <IconButton onClick={stream.restart} disabled={inspecting} aria-label={t('dataset.restart')} sx={{ bgcolor: 'rgba(148, 163, 184, 0.2)', color: 'white' }}>
            ↺
          </IconButton>
        </Tooltip>
        <ToggleButtonGroup
          size="small"
          exclusive
          value={speed}
          onChange={(_, v: StreamSpeed | null) => { if (v) stream.setSpeed(v); }}
          aria-label={t('dataset.speed')}
          sx={{ ml: 'auto', '& .MuiToggleButton-root': { px: 1, py: 0.25, fontSize: '0.75rem', fontFamily: 'monospace', color: 'text.secondary', borderColor: '#334155', '&.Mui-selected': { color: 'white', bgcolor: 'rgba(59, 130, 246, 0.4)' } } }}
        >
          {STREAM_SPEEDS.map(s => <ToggleButton key={s} value={s}>{s}×</ToggleButton>)}
        </ToggleButtonGroup>
      </Stack>

      {inspection ? (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, p: 1, borderRadius: 1, bgcolor: 'rgba(168, 85, 247, 0.15)', border: '1px solid rgba(168, 85, 247, 0.5)' }}>
          <Typography variant="body2" sx={{ flex: 1, fontWeight: 600 }}>
            {t('dataset.inspecting', { id: inspection.candidate.id })}
          </Typography>
          <Button size="small" variant="contained" color={inspection.isPaused ? 'success' : 'warning'} onClick={onToggleInspection} sx={{ py: 0.25, px: 1.5, minWidth: 0 }}>
            {inspection.isPaused ? t('dataset.inspectResume') : t('dataset.inspectPause')}
          </Button>
          <Button size="small" variant="outlined" color="inherit" onClick={onStopInspection} sx={{ py: 0.25, px: 1.5, minWidth: 0 }}>
            {t('dataset.inspectStop')}
          </Button>
        </Box>
      ) : processedTotal === 0 && !running ? (
        <Typography variant="caption" color="text.secondary">{t('dataset.hint')}</Typography>
      ) : null}

      {/* Sample list (plain elements: 200 rows re-render on every step) */}
      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: '2.2em 1fr 4.5em 5em',
          px: 1,
          fontSize: '0.7rem',
          color: 'text.secondary',
          fontFamily: 'monospace',
        }}
      >
        <span>#</span>
        <span>{t('controls.grade')} · {t('controls.attitude')} · {t('controls.response')}</span>
        <span>{t('dataset.label')}</span>
        <span>{t('dataset.prediction')}</span>
      </Box>
      <Box
        ref={listRef}
        sx={{
          height: listHeight,
          minHeight: 240,
          overflowY: 'auto',
          border: '1px solid #334155',
          borderRadius: 1,
          fontFamily: 'monospace',
          fontSize: '0.75rem',
          '& > div': {
            display: 'grid',
            gridTemplateColumns: '2.2em 1fr 4.5em 5em',
            alignItems: 'center',
            px: 1,
            py: 0.25,
            borderBottom: '1px solid rgba(51, 65, 85, 0.5)',
            cursor: inspecting ? 'default' : 'pointer',
          },
          '& > div:hover': { bgcolor: inspecting ? undefined : 'rgba(148, 163, 184, 0.12)' },
          '& > div.pending': { opacity: 0.45 },
          '& > div.current': { bgcolor: 'rgba(59, 130, 246, 0.25)', outline: '1px solid rgba(96, 165, 250, 0.7)' },
          '& > div.inspected': { opacity: 1, bgcolor: 'rgba(168, 85, 247, 0.3)', outline: '1px solid rgba(192, 132, 252, 0.9)' },
        }}
      >
        {dataset.map((c, i) => {
          const r = results[i];
          const className = inspection?.candidate.id === c.id ? 'inspected' : i === lastIndex ? 'current' : r === null ? 'pending' : '';
          return (
            <div
              key={c.id}
              className={className}
              onClick={() => { if (!inspecting) onInspect(c); }}
              title={inspecting ? undefined : t('dataset.rowHint')}
            >
              <span style={{ color: '#64748b' }}>{c.id}</span>
              <span>{c.grade.toFixed(2)} · {c.attitude.toFixed(2)} · {c.response.toFixed(2)}</span>
              <span style={{ color: CLASS_COLORS[c.label], fontWeight: 600 }}>{classNames[c.label]}</span>
              <span>
                {r === null ? '' : (
                  <>
                    <span aria-label={r.correct ? t('dataset.correct') : t('dataset.wrong')}>{r.correct ? '✅' : '❌'}</span>
                    <span style={{ color: CLASS_COLORS[r.predicted], marginLeft: 4 }}>{classNames[r.predicted]}</span>
                  </>
                )}
              </span>
            </div>
          );
        })}
      </Box>
    </Paper>
  );
}
