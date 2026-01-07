import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  Typography,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
}  from '@mui/material';
import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import type { WeightComparisonData, LayerWeightComparison } from '../lib/types';

interface WeightComparisonModalProps {
  open: boolean;
  data: WeightComparisonData | null;
  onClose: () => void;
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`weight-tabpanel-${index}`}
      aria-labelledby={`weight-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ pt: 2 }}>{children}</Box>}
    </div>
  );
}

function WeightChangeCell({ delta }: { delta: number }) {
  const isPositive = delta > 0;
  const absValue = Math.abs(delta);
  
  // Color intensity based on magnitude (capped at 0.8 for readability)
  const intensity = Math.min(absValue * 50, 0.8);
  const bgColor = isPositive
    ? `rgba(34, 197, 94, ${intensity})` // Green for increase
    : `rgba(239, 68, 68, ${intensity})`; // Red for decrease

  return (
    <TableCell
      sx={{
        bgcolor: bgColor,
        textAlign: 'center',
        fontWeight: 600,
        color: intensity > 0.4 ? 'white' : 'text.primary',
      }}
    >
      <Typography variant="body2" component="span">
        {isPositive ? '+' : ''}{delta.toFixed(4)}
      </Typography>
    </TableCell>
  );
}

function LayerComparisonTable({ layerData }: { layerData: LayerWeightComparison }) {
  const { t } = useTranslation();
  const numNeurons = layerData.oldWeights.length;
  const numInputs = layerData.oldWeights[0]?.length || 0;

  return (
    <Box>
      {/* Summary Statistics */}
      <Box sx={{ mb: 2, p: 2, bgcolor: 'rgba(59, 130, 246, 0.1)', borderRadius: 1 }}>
        <Typography variant="body2" sx={{ mb: 0.5 }}>
          <strong>{t('comparison.neurons')}:</strong> {numNeurons}
        </Typography>
        <Typography variant="body2" sx={{ mb: 0.5 }}>
          <strong>{t('comparison.inputsPerNeuron')}:</strong> {numInputs}
        </Typography>
        <Typography variant="body2">
          <strong>{t('comparison.totalParameters')}:</strong> {numNeurons * numInputs + numNeurons}
        </Typography>
      </Box>

      <TableContainer component={Paper} sx={{ maxHeight: 500, bgcolor: 'background.paper' }}>
        <Table stickyHeader size="small">
          <TableHead>
            <TableRow>
              <TableCell sx={{ fontWeight: 700, minWidth: 100 }}>{t('comparison.neuron')}</TableCell>
              <TableCell sx={{ fontWeight: 700, minWidth: 80 }}>{t('comparison.parameter')}</TableCell>
              <TableCell sx={{ fontWeight: 700, minWidth: 100, textAlign: 'center' }}>{t('comparison.before')}</TableCell>
              <TableCell sx={{ fontWeight: 700, minWidth: 100, textAlign: 'center' }}>{t('comparison.delta')}</TableCell>
              <TableCell sx={{ fontWeight: 700, minWidth: 100, textAlign: 'center' }}>{t('comparison.after')}</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {layerData.oldWeights.map((neuronWeights, neuronIdx) => (
              <React.Fragment key={`neuron-${neuronIdx}`}>
                {/* Weights */}
                {neuronWeights.map((oldWeight, weightIdx) => (
                  <TableRow 
                    key={`n${neuronIdx}-w${weightIdx}`}
                    sx={{ '&:hover': { bgcolor: 'action.hover' } }}
                  >
                    {weightIdx === 0 && (
                      <TableCell 
                        rowSpan={neuronWeights.length + 1}
                        sx={{ 
                          fontWeight: 600,
                          borderRight: '2px solid',
                          borderColor: 'divider',
                          bgcolor: 'rgba(100, 100, 100, 0.05)',
                        }}
                      >
                        #{neuronIdx + 1}
                      </TableCell>
                    )}
                    <TableCell sx={{ color: 'text.secondary' }}>
                      W<sub>{weightIdx + 1}</sub>
                    </TableCell>
                    <TableCell sx={{ textAlign: 'center', fontFamily: 'monospace' }}>
                      {oldWeight.toFixed(4)}
                    </TableCell>
                    <WeightChangeCell delta={layerData.weightDeltas[neuronIdx][weightIdx]} />
                    <TableCell sx={{ textAlign: 'center', fontFamily: 'monospace', fontWeight: 600 }}>
                      {layerData.newWeights[neuronIdx][weightIdx].toFixed(4)}
                    </TableCell>
                  </TableRow>
                ))}
                
                {/* Bias */}
                <TableRow sx={{ bgcolor: 'rgba(59, 130, 246, 0.05)', '&:hover': { bgcolor: 'action.hover' } }}>
                  <TableCell sx={{ color: 'text.secondary', fontStyle: 'italic' }}>
                    bias
                  </TableCell>
                  <TableCell sx={{ textAlign: 'center', fontFamily: 'monospace' }}>
                    {layerData.oldBiases[neuronIdx].toFixed(4)}
                  </TableCell>
                  <WeightChangeCell delta={layerData.biasDeltas[neuronIdx]} />
                  <TableCell sx={{ textAlign: 'center', fontFamily: 'monospace', fontWeight: 600 }}>
                    {layerData.newBiases[neuronIdx].toFixed(4)}
                  </TableCell>
                </TableRow>
              </React.Fragment>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
}

export default function WeightComparisonModal({ open, data, onClose }: WeightComparisonModalProps) {
  const { t } = useTranslation();
  const [selectedTab, setSelectedTab] = useState(0);

  if (!data) return null;

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setSelectedTab(newValue);
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="lg"
      fullWidth
      PaperProps={{
        sx: {
          bgcolor: 'background.paper',
          backgroundImage: 'linear-gradient(135deg, #1e293b 0%, #334155 100%)',
          border: '2px solid #60a5fa',
          minHeight: '70vh',
        },
      }}
    >
      <DialogTitle sx={{ textAlign: 'center', borderBottom: '2px solid #475569', pb: 2 }}>
        <Typography variant="h5" component="div" fontWeight={700}>
          {t('comparison.title')}
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
          {t('comparison.learningRate')}: {data.learningRate.toFixed(3)} | 
          {' '}{t('comparison.totalChange')}: {data.totalChange.toFixed(6)} |
          {' '}{t('comparison.maxChange')}: {data.maxWeightChange.toFixed(6)}
        </Typography>
      </DialogTitle>

      <DialogContent sx={{ pt: 0 }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs 
            value={selectedTab} 
            onChange={handleTabChange}
            variant="fullWidth"
            sx={{
              '& .MuiTab-root': {
                fontWeight: 600,
                fontSize: '1rem',
              },
            }}
          >
            <Tab label={`📥 ${t('layers.layer1Prefix')}${t('layers.interviewer')}`} />
            <Tab label={`👔 ${t('layers.layer2Prefix')}${t('layers.interviewer')}`} />
            <Tab label={`⚖️ ${t('layers.output')}`} />
          </Tabs>
        </Box>

        <TabPanel value={selectedTab} index={0}>
          <LayerComparisonTable layerData={data.layer1} />
        </TabPanel>
        <TabPanel value={selectedTab} index={1}>
          <LayerComparisonTable layerData={data.layer2} />
        </TabPanel>
        <TabPanel value={selectedTab} index={2}>
          <LayerComparisonTable layerData={data.output} />
        </TabPanel>
      </DialogContent>

      <DialogActions sx={{ p: 2, borderTop: '1px solid #475569' }}>
        <Button
          onClick={onClose}
          variant="contained"
          sx={{
            background: 'linear-gradient(90deg, #3b82f6 0%, #60a5fa 100%)',
            fontWeight: 600,
            px: 4,
          }}
        >
          {t('comparison.close')}
        </Button>
      </DialogActions>
    </Dialog>
  );
}
