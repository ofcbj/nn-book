import { Typography, Paper } from '@mui/material';

export default function Header() {
  return (
    <Paper
      sx={{
        textAlign: 'center',
        py: 4,
        px: 3,
        mb: 2.5,
        background: 'linear-gradient(135deg, #1e293b 0%, #334155 100%)',
      }}
    >
      <Typography variant="h1" sx={{ color: 'white', mb: 1.5 }}>
        🧠 Neural Network: 면접 시스템 시각화
      </Typography>
      <Typography variant="body1" color="text.secondary">
        입력값과 가중치의 내적, 그리고 활성화 함수를 통한 의사결정 과정
      </Typography>
    </Paper>
  );
}

export function Footer() {
  return (
    <Paper sx={{ textAlign: 'center', py: 2.5, px: 3, mt: 2.5 }}>
      <Typography color="text.secondary" sx={{ mb: 1 }}>
        Architecture: <strong>3 inputs</strong> → <strong>5 neurons(1차)</strong> → 
        <strong> 3 neurons(2차)</strong> → <strong>3 outputs</strong>
      </Typography>
      <Typography variant="body2" color="text.disabled" sx={{ maxWidth: 800, mx: 'auto' }}>
        이 시각화는 면접자의 3가지 속성(성적, 태도, 응답수준)이 1차 면접관(5명)을 거쳐 
        2차 면접관(3명)에게 전달되고 최종 결정으로 이어지는 과정을 보여줍니다.
      </Typography>
    </Paper>
  );
}
