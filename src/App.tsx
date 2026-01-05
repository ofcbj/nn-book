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
import { useNeuralNetwork } from './hooks/useNeuralNetwork';

export default function App() {
  const {
    network,
    inputs,
    controls,
    stats,
    training,
    modals,
    visualization,
    actions,
  } = useNeuralNetwork();

  // Initial visualization
  useEffect(() => {
    actions.updateVisualization();
  }, [actions.updateVisualization]);

  // Update visualization when inputs change
  useEffect(() => {
    actions.updateVisualization();
  }, [inputs.grade, inputs.attitude, inputs.response, actions.updateVisualization]);

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
            <Stack spacing={2.5}>
              <ControlPanel
                grade={inputs.grade}
                attitude={inputs.attitude}
                response={inputs.response}
                targetValue={inputs.targetValue}
                learningRate={inputs.learningRate}
                animationSpeed={inputs.animationSpeed}
                isManualMode={inputs.isManualMode}
                onGradeChange={controls.setGrade}
                onAttitudeChange={controls.setAttitude}
                onResponseChange={controls.setResponse}
                onTargetChange={controls.setTargetValue}
                onLearningRateChange={controls.setLearningRate}
                onAnimationSpeedChange={controls.setAnimationSpeed}
                onManualModeChange={controls.setIsManualMode}
                onNextStep={controls.nextStep}
                onStep={actions.trainOneStep}
                onTrainToggle={actions.toggleTraining}
                onReset={actions.reset}
                isTraining={training.isTraining}
                isAnimating={training.isAnimating}
                showCanvasHeatmap={visualization.showCanvasHeatmap}
                showGridHeatmap={visualization.showGridHeatmap}
                onToggleCanvasHeatmap={visualization.toggleCanvasHeatmap}
                onToggleGridHeatmap={visualization.toggleGridHeatmap}
                hasComparisonData={modals.comparison.data !== null}
                onViewComparison={modals.comparison.open}
              />
              <StatsDisplay epoch={stats.epoch} loss={stats.loss} output={stats.output} />
            </Stack>
          </Box>
          {/* Center: Network Visualization */}
          <Box sx={{ flex: 1, minWidth: 0 }}>
            <Stack spacing={2.5}>
              <NetworkCanvas nn={network.nn} onVisualizerReady={network.setVisualizer} onCanvasClick={actions.handleCanvasClick} />
              
              {/* Activation Heatmap */}
              {visualization.showGridHeatmap && <ActivationHeatmap activations={visualization.activations} />}
            </Stack>
          </Box>
          {/* Right Panel: Calculation Display */}
          <Box sx={{ width: { xs: '100%', lg: 280 }, flexShrink: 0 }}>
            <CalculationPanel steps={stats.steps} />
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
