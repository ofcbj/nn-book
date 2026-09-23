import { useEffect, useState } from 'react';
import { Box, Container, Stack } from '@mui/material';
import { useTranslation } from 'react-i18next';
import Header, { Footer } from './components/Header';
import ControlPanel from './components/ControlPanel';
import StatsDisplay from './components/StatsDisplay';
import NetworkCanvas from './components/NetworkCanvas';
import CalculationPanel from './components/CalculationPanel';
import LossChart from './components/LossChart';
import DataStreamPanel from './components/DataStreamPanel';
import LossModal from './components/LossModal';
import BackpropModal from './components/BackpropModal';
import HelpModal from './components/HelpModal';
import ActivationHeatmap from './components/ActivationHeatmap';
import WeightComparisonModal from './components/WeightComparisonModal';
import { useNeuralNetwork } from './hooks/useNeuralNetwork';

export default function App() {
  const { t } = useTranslation();
  const {
    inputs,
    controls,
    stats,
    training,
    stream,
    inspection,
    modals,
    visualizer,
    actions,
    network,
  } = useNeuralNetwork();

  const dataMode = training.mode === 'dataset';

  // Recompute the forward pass on mount and whenever an input changes.
  // Only the inputs are dependencies on purpose: `actions` changes identity on
  // every animation tick and must not trigger a recomputation.
  const { computeAndRefreshDisplay } = actions;
  useEffect(() => {
    computeAndRefreshDisplay();
  }, [inputs.grade, inputs.attitude, inputs.response]);

  const [showHelpModal, setShowHelpModal] = useState(false);

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: 'background.default' }}>
      <Container maxWidth={false} sx={{ py: 1.5, px: { xs: 1.5, md: 2 } }}>
        <Header
          mode={training.mode}
          onModeChange={training.setMode}
          modeDisabled={training.isAnimating || training.isTraining}
          onHelpClick={() => setShowHelpModal(true)}
        />

        <Box
          sx={{
            display: 'flex',
            gap: 1.5,
            mb: 1.5,
            flexDirection: { xs: 'column', lg: 'row' },
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
              isPaused={training.isPaused}
              dataMode={dataMode}
            />
          </Box>

          {/* Center: Network Visualizer */}
          <Box sx={{ flex: 1, minWidth: 0 }}>
            <Stack spacing={1.5}>
              <NetworkCanvas
                onVisualizerReady={network.setVisualizer}
                onRedraw={network.redraw}
                onCanvasClick={actions.handleCanvasClick}
              />
              <ActivationHeatmap activations={visualizer.activations} />
            </Stack>
          </Box>

          {/* Right Panel */}
          <Box sx={{ width: { xs: '100%', lg: dataMode ? 380 : 280 }, flexShrink: 0 }}>
            <Stack spacing={1.5}>
              {dataMode ? (
                <>
                  <DataStreamPanel
                    stream={stream}
                    inspection={inspection}
                    onInspect={actions.inspectCandidate}
                    onToggleInspection={actions.toggleInspection}
                    onStopInspection={actions.stopInspection}
                  />
                  <LossChart
                    history={stats.lossHistory}
                    title={t('dataset.lossTitle')}
                    xLabel={t('dataset.sample')}
                    smoothingWindow={20}
                  />
                </>
              ) : (
                <>
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
                  <LossChart history={stats.lossHistory} />
                  <CalculationPanel
                    steps={stats.steps}
                    hasComparisonData={modals.comparison.data !== null}
                    onViewComparison={modals.comparison.open}
                  />
                </>
              )}
            </Stack>
          </Box>
        </Box>

        <Footer />

        {modals.loss.data && (
          <LossModal
            open={modals.loss.show}
            targetClass={modals.loss.data.targetClass}
            predictions={modals.loss.data.predictions}
            loss={modals.loss.data.loss}
            onClose={modals.loss.close}
          />
        )}

        <BackpropModal
          open={modals.backprop.show}
          data={modals.backprop.data}
          onClose={modals.backprop.close}
        />

        <HelpModal
          open={showHelpModal}
          onClose={() => setShowHelpModal(false)}
        />

        <WeightComparisonModal
          open={modals.comparison.show}
          data={modals.comparison.data}
          onClose={modals.comparison.close}
        />
      </Container>
    </Box>
  );
}
