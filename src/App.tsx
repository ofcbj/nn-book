import { useEffect } from 'react';
import { Box, Container, Stack } from '@mui/material';
import Header, { Footer } from './components/Header';
import ControlPanel from './components/ControlPanel';
import StatsDisplay from './components/StatsDisplay';
import NetworkCanvas from './components/NetworkCanvas';
import CalculationPanel from './components/CalculationPanel';
import LossModal from './components/LossModal';
import { useNeuralNetwork } from './hooks/useNeuralNetwork';

export default function App() {
  const {
    nn,
    setVisualizer,
    grade,
    attitude,
    response,
    targetValue,
    learningRate,
    animationSpeed,
    isManualMode,
    setGrade,
    setAttitude,
    setResponse,
    setTargetValue,
    setLearningRate,
    setAnimationSpeed,
    setIsManualMode,
    nextStep,
    epoch,
    loss,
    output,
    steps,
    isTraining,
    isAnimating,
    showLossModal,
    lossModalData,
    trainOneStepWithAnimation,
    toggleTraining,
    reset,
    closeLossModal,
    updateVisualization,
  } = useNeuralNetwork();

  // Initial visualization
  useEffect(() => {
    updateVisualization();
  }, [updateVisualization]);

  // Update visualization when inputs change
  useEffect(() => {
    updateVisualization();
  }, [grade, attitude, response, updateVisualization]);

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: 'background.default' }}>
      <Container maxWidth="xl" sx={{ py: 2.5 }}>
        <Header />

        <Box 
          sx={{ 
            display: 'flex', 
            gap: 2.5, 
            mb: 2.5,
            flexDirection: { xs: 'column', lg: 'row' }
          }}
        >
          {/* Left Panel: Controls */}
          <Box sx={{ width: { xs: '100%', lg: 220 }, flexShrink: 0 }}>
            <Stack spacing={2.5}>
              <ControlPanel
                grade={grade}
                attitude={attitude}
                response={response}
                targetValue={targetValue}
                learningRate={learningRate}
                animationSpeed={animationSpeed}
                isManualMode={isManualMode}
                onGradeChange={setGrade}
                onAttitudeChange={setAttitude}
                onResponseChange={setResponse}
                onTargetChange={setTargetValue}
                onLearningRateChange={setLearningRate}
                onAnimationSpeedChange={setAnimationSpeed}
                onManualModeChange={setIsManualMode}
                onNextStep={nextStep}
                onStep={trainOneStepWithAnimation}
                onTrainToggle={toggleTraining}
                onReset={reset}
                isTraining={isTraining}
                isAnimating={isAnimating}
              />
              <StatsDisplay epoch={epoch} loss={loss} output={output} />
            </Stack>
          </Box>

          {/* Center: Network Visualization */}
          <Box sx={{ flex: 1, minWidth: 0 }}>
            <NetworkCanvas nn={nn} onVisualizerReady={setVisualizer} />
          </Box>

          {/* Right Panel: Calculation Display */}
          <Box sx={{ width: { xs: '100%', lg: 280 }, flexShrink: 0 }}>
            <CalculationPanel steps={steps} />
          </Box>
        </Box>

        <Footer />

        {/* Loss Modal */}
        {lossModalData && (
          <LossModal
            open={showLossModal}
            targetClass={lossModalData.targetClass}
            predictions={lossModalData.predictions}
            loss={lossModalData.loss}
            onClose={closeLossModal}
          />
        )}
      </Container>
    </Box>
  );
}
