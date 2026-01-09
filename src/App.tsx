import { useEffect, useState } from 'react';
import { Box, Container, Stack } from '@mui/material';
import Header, { Footer } from './components/Header';
import ControlPanel from './components/ControlPanel';
import StatsDisplay from './components/StatsDisplay';
import NetworkCanvas from './components/NetworkCanvas';
import CalculationPanel from './components/CalculationPanel';
import LossModal from './components/LossModal';
import BackpropModal from './components/BackpropModal';
import HelpModal from './components/HelpModal';
import ActivationHeatmap from './components/ActivationHeatmap';
import WeightComparisonModal from './components/WeightComparisonModal';
import { NetworkProvider, useNetworkContext } from './hooks/NetworkContext';

function AppContent() {
  // Get state from context instead of directly from useNeuralNetwork
  const {
    network,
    inputs,
    controls,
    stats,
    training,
    modals,
    visualizer,
    actions,
  } = useNetworkContext();

  // Initial visualizer
  useEffect(() => {
    actions.computeAndRefreshDisplay();
  }, [actions.computeAndRefreshDisplay]);

  // Update visualizer when inputs change
  useEffect(() => {
    actions.computeAndRefreshDisplay();
  }, [inputs.grade, inputs.attitude, inputs.response, actions.computeAndRefreshDisplay]);

  // Help modal state
  const [showHelpModal, setShowHelpModal] = useState(false);

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: 'background.default' }}>
      <Container maxWidth="xl" sx={{ py: 2.5 }}>
        <Header onHelpClick={() => setShowHelpModal(true)} />

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
            <ControlPanel
              grade={inputs.grade}
              attitude={inputs.attitude}
              response={inputs.response}
              targetValue={inputs.targetValue}
              animationSpeed={inputs.animationSpeed}
              onGradeChange={controls.setGrade}
              onAttitudeChange={controls.setAttitude}
              onResponseChange={controls.setResponse}
              onTargetChange={controls.setTargetValue}
              onAnimationSpeedChange={controls.setAnimationSpeed}
              onStep={actions.trainOneStep}
              onReset={actions.reset}
              isAnimating={training.isAnimating}
              isJumped={training.isJumped}
            />
          </Box>
          {/* Center: Network Visualizer */}
          <Box sx={{ flex: 1, minWidth: 0 }}>
            <Stack spacing={2.5}>
              <NetworkCanvas nn={network.nn} onVisualizerReady={network.setVisualizer} onCanvasClick={actions.handleCanvasClick} />

              {/* Activation Heatmap - Always visible */}
              <ActivationHeatmap activations={visualizer.activations} />
            </Stack>
          </Box>
          {/* Right Panel: Stats, Weight Comparison, Calculation Display */}
          <Box sx={{ width: { xs: '100%', lg: 280 }, flexShrink: 0 }}>
            <Stack spacing={2.5}>
              <StatsDisplay 
                epoch={stats.epoch} 
                loss={stats.loss} 
                output={stats.output}
                learningRate={inputs.learningRate}
                isTraining={training.isTraining}
                onLearningRateChange={controls.setLearningRate}
                onTrainOnce={actions.trainOneEpoch}
                onTrainToggle={actions.toggleTraining}
              />
              <CalculationPanel steps={stats.steps} hasComparisonData={modals.comparison.data !== null} onViewComparison={modals.comparison.open} />
            </Stack>
          </Box>
        </Box>

        <Footer />
        {/* Loss Modal */}
        {modals.loss.data && (
          <LossModal
            open={modals.loss.show}
            targetClass={modals.loss.data.targetClass}
            predictions={modals.loss.data.predictions}
            loss={modals.loss.data.loss}
            onClose={modals.loss.close}
          />
        )}

        {/* Backprop Summary Modal */}
        <BackpropModal
          open={modals.backprop.show}
          data={modals.backprop.data}
          onClose={modals.backprop.close}
        />

        {/* Help Modal */}
        <HelpModal
          open={showHelpModal}
          onClose={() => setShowHelpModal(false)}
        />

        {/* Weight Comparison Modal */}
        <WeightComparisonModal
          open={modals.comparison.show}
          data={modals.comparison.data}
          onClose={modals.comparison.close}
        />
      </Container>
    </Box>
  );
}

// Main App component - wraps with NetworkProvider
export default function App() {
  return (
    <NetworkProvider>
      <AppContent />
    </NetworkProvider>
  );
}
