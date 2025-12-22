import { Box, Paper, Typography, Stack } from '@mui/material';

interface StatsDisplayProps {
  epoch: number;
  loss: number;
  output: number[] | null;
}

const classNames = ['불합격', '보류', '합격'];

export default function StatsDisplay({ epoch, loss, output }: StatsDisplayProps) {
  const getOutputText = () => {
    if (!output) return '0.000';
    return output.map((prob, i) => 
      `${classNames[i]}: ${(prob * 100).toFixed(1)}%`
    ).join(' | ');
  };

  const getOutputColor = () => {
    if (!output) return 'primary.light';
    const maxIndex = output.indexOf(Math.max(...output));
    return maxIndex === 0 ? 'error.main' : maxIndex === 1 ? 'warning.main' : 'secondary.main';
  };

  return (
    <Paper 
      sx={{ 
        p: 2, 
        bgcolor: '#0f172a', 
        borderRadius: 2 
      }}
    >
      <Typography variant="h3" sx={{ mb: 2, pb: 1, borderBottom: '2px solid #334155' }}>
        📈 통계 정보
      </Typography>
      
      <Stack spacing={1.5}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', py: 1, borderBottom: '1px solid #1e293b' }}>
          <Typography variant="body2" color="text.secondary">
            Epoch:
          </Typography>
          <Typography 
            sx={{ 
              fontFamily: 'monospace', 
              fontWeight: 700,
              color: 'primary.light'
            }}
          >
            {epoch}
          </Typography>
        </Box>
        
        <Box sx={{ display: 'flex', justifyContent: 'space-between', py: 1, borderBottom: '1px solid #1e293b' }}>
          <Typography variant="body2" color="text.secondary">
            Loss:
          </Typography>
          <Typography 
            sx={{ 
              fontFamily: 'monospace', 
              fontWeight: 700,
              color: 'primary.light'
            }}
          >
            {loss.toFixed(6)}
          </Typography>
        </Box>
        
        <Box sx={{ display: 'flex', justifyContent: 'space-between', py: 1 }}>
          <Typography variant="body2" color="text.secondary">
            Output:
          </Typography>
          <Typography 
            sx={{ 
              fontFamily: 'monospace', 
              fontWeight: 700,
              color: getOutputColor(),
              fontSize: '0.85rem'
            }}
          >
            {getOutputText()}
          </Typography>
        </Box>
      </Stack>
    </Paper>
  );
}
