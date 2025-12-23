import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  Typography,
} from '@mui/material';

interface HelpModalProps {
  open: boolean;
  onClose: () => void;
}

export default function HelpModal({ open, onClose }: HelpModalProps) {
  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="md"
      fullWidth
      PaperProps={{
        sx: {
          bgcolor: 'background.paper',
          backgroundImage: 'linear-gradient(135deg, #1e293b 0%, #334155 100%)',
          border: '2px solid #60a5fa',
        },
      }}
    >
      <DialogTitle sx={{ textAlign: 'center', borderBottom: '2px solid #475569', pb: 2 }}>
        <Typography variant="h5" fontWeight={700}>
          📚 신경망으로 이해하는 면접 과정
        </Typography>
      </DialogTitle>
      
      <DialogContent sx={{ pt: 3 }}>
        {/* Introduction */}
        <Box sx={{ mb: 3, p: 2, bgcolor: 'rgba(59, 130, 246, 0.1)', borderRadius: 2, border: '1px solid rgba(59, 130, 246, 0.3)' }}>
          <Typography variant="h6" fontWeight={600} sx={{ mb: 1.5 }}>
            🎯 이 시스템은 무엇인가요?
          </Typography>
          <Typography fontSize="0.95rem" sx={{ lineHeight: 1.7 }}>
            면접 과정을 신경망(Neural Network)으로 시각화한 교육용 도구입니다. 
            실제 면접처럼 여러 단계를 거쳐 최종 합격 여부를 결정하는 과정을 
            신경망의 학습 과정으로 표현했습니다.
          </Typography>
        </Box>

        {/* Network Structure */}
        <Box sx={{ mb: 3, p: 2, bgcolor: 'rgba(34, 197, 94, 0.1)', borderRadius: 2, border: '1px solid rgba(34, 197, 94, 0.3)' }}>
          <Typography variant="h6" fontWeight={600} sx={{ mb: 1.5 }}>
            🏗️ 네트워크 구조
          </Typography>
          
          <Box sx={{ mb: 2 }}>
            <Typography fontWeight={600} fontSize="0.95rem" sx={{ mb: 0.5 }}>📥 입력 레이어 (3개)</Typography>
            <Typography fontSize="0.9rem" color="text.secondary" sx={{ ml: 2 }}>
              • 성적 (0.0 ~ 1.0)<br/>
              • 태도 (0.0 ~ 1.0)<br/>
              • 응답수준 (0.0 ~ 1.0)
            </Typography>
          </Box>

          <Box sx={{ mb: 2 }}>
            <Typography fontWeight={600} fontSize="0.95rem" sx={{ mb: 0.5 }}>👥 1차 면접관 (5명)</Typography>
            <Typography fontSize="0.9rem" color="text.secondary" sx={{ ml: 2 }}>
              면접자의 3가지 특성을 각자의 방식으로 평가합니다.
            </Typography>
          </Box>

          <Box sx={{ mb: 2 }}>
            <Typography fontWeight={600} fontSize="0.95rem" sx={{ mb: 0.5 }}>👔 2차 면접관 (3명)</Typography>
            <Typography fontSize="0.9rem" color="text.secondary" sx={{ ml: 2 }}>
              1차 면접관 5명의 평가를 종합하여 재평가합니다.
            </Typography>
          </Box>

          <Box>
            <Typography fontWeight={600} fontSize="0.95rem" sx={{ mb: 0.5 }}>⚖️ 최종 결정 (3가지)</Typography>
            <Typography fontSize="0.9rem" color="text.secondary" sx={{ ml: 2 }}>
              • 불합격<br/>
              • 보류<br/>
              • 합격
            </Typography>
          </Box>
        </Box>

        {/* How it works */}
        <Box sx={{ mb: 3, p: 2, bgcolor: 'rgba(249, 115, 22, 0.1)', borderRadius: 2, border: '1px solid rgba(249, 115, 22, 0.3)' }}>
          <Typography variant="h6" fontWeight={600} sx={{ mb: 1.5 }}>
            ⚙️ 어떻게 동작하나요?
          </Typography>
          
          <Box sx={{ mb: 2 }}>
            <Typography fontWeight={600} fontSize="0.95rem" sx={{ mb: 0.5 }}>1️⃣ Forward Pass (순전파)</Typography>
            <Typography fontSize="0.9rem" color="text.secondary" sx={{ ml: 2 }}>
              입력값이 각 레이어를 거쳐 최종 예측값을 만들어냅니다.<br/>
              각 뉴런은 가중치(W)와 바이어스(b)를 사용해 계산합니다.
            </Typography>
          </Box>

          <Box>
            <Typography fontWeight={600} fontSize="0.95rem" sx={{ mb: 0.5 }}>2️⃣ Backpropagation (역전파)</Typography>
            <Typography fontSize="0.9rem" color="text.secondary" sx={{ ml: 2 }}>
              예측이 틀렸을 때, 오류를 역으로 전파하여 가중치를 조정합니다.<br/>
              이 과정을 반복하면 신경망이 점점 더 정확해집니다.
            </Typography>
          </Box>
        </Box>

        {/* Controls */}
        <Box sx={{ p: 2, bgcolor: 'rgba(239, 68, 68, 0.1)', borderRadius: 2, border: '1px solid rgba(239, 68, 68, 0.3)' }}>
          <Typography variant="h6" fontWeight={600} sx={{ mb: 1.5 }}>
            🎮 주요 기능
          </Typography>
          
          <Typography fontSize="0.9rem" color="text.secondary" component="div">
            • <strong>입력 조절</strong>: 슬라이더로 면접자 특성 변경<br/>
            • <strong>1회 학습</strong>: 애니메이션으로 학습 과정 확인<br/>
            • <strong>자동 학습</strong>: 빠르게 여러 번 학습<br/>
            • <strong>수동 모드</strong>: 캔버스 클릭 또는 "다음 단계" 버튼으로 직접 진행<br/>
            • <strong>애니메이션 속도</strong>: 0~3배 조절 (0은 수동 모드와 동일)<br/>
            • <strong>초기화</strong>: 신경망을 처음 상태로 되돌림
          </Typography>
        </Box>
      </DialogContent>

      <DialogActions sx={{ p: 2, borderTop: '1px solid #475569' }}>
        <Button 
          onClick={onClose} 
          variant="contained"
          sx={{
            background: 'linear-gradient(90deg, #3b82f6 0%, #60a5fa 100%)',
            fontWeight: 600,
            px: 4
          }}
        >
          닫기
        </Button>
      </DialogActions>
    </Dialog>
  );
}
