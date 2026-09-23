/**
 * Data Training Mode
 *
 * Streams a generated dataset through the network one candidate at a time:
 * predict → train → redraw. Tracks, for the current pass, what the network
 * predicted for each candidate *before* learning from it (the "online"
 * accuracy) and re-evaluates the whole dataset after every step so the
 * overall accuracy can be watched rising.
 *
 * The mutable stream state lives in a ref (the interval callback needs the
 * latest values synchronously); `publish()` copies it into React state.
 */

import { useCallback, useEffect, useRef, useState, RefObject } from 'react';
import { NeuralNetwork, generateDataset, candidateInputs, argmax, evaluateDataset, toOneHot } from '../lib/core';
import type { Candidate, DatasetEvaluation } from '../lib/core';
import type { StatsSetters } from './useNetworkState';

export type StreamSpeed = 1 | 2 | 5 | 20;
export const STREAM_SPEEDS: readonly StreamSpeed[] = [1, 2, 5, 20];
/** Interval between samples at 1× */
const BASE_INTERVAL_MS = 600;

export interface SampleResult {
  /** Class predicted before training on this sample */
  predicted: number;
  correct: boolean;
  probabilities: number[];
  loss: number;
}

export interface DatasetTrainingState {
  dataset: Candidate[];
  /** Index of the next sample to train */
  cursor: number;
  /** Index of the most recently trained sample in the current pass, or null */
  lastIndex: number | null;
  /** 1-based pass number over the dataset */
  pass: number;
  /** Per-sample results for the current pass */
  results: (SampleResult | null)[];
  /** Samples trained so far, across passes */
  processedTotal: number;
  /** Correct before-training predictions so far, across passes */
  onlineCorrect: number;
  running: boolean;
  speed: StreamSpeed;
  /** Whole-dataset evaluation of the current network */
  evaluation: DatasetEvaluation | null;
}

export interface UseDatasetTrainingReturn extends DatasetTrainingState {
  play: () => void;
  pause: () => void;
  toggle: () => void;
  /** Train on the next sample */
  step: () => void;
  setSpeed: (speed: StreamSpeed) => void;
  /** Back to the first sample of pass 1; the network is left as it is */
  restart: () => void;
  /** Re-evaluate the dataset with the current network (after it was trained elsewhere) */
  refresh: () => void;
}

interface UseDatasetTrainingDeps {
  /** Feed inputs through the network, update stats/steps and redraw the canvas */
  displayInputs: (inputs: number[]) => void;
  statsSetters: StatsSetters;
  /** False while another mode is active: the stream is paused and stays paused */
  enabled: boolean;
}

function createInitialState(dataset: Candidate[]): DatasetTrainingState {
  return {
    dataset,
    cursor: 0,
    lastIndex: null,
    pass: 1,
    results: Array<SampleResult | null>(dataset.length).fill(null),
    processedTotal: 0,
    onlineCorrect: 0,
    running: false,
    speed: 5,
    evaluation: null,
  };
}

export function useDatasetTraining(
  nnRef: RefObject<NeuralNetwork>,
  { displayInputs, statsSetters, enabled }: UseDatasetTrainingDeps
): UseDatasetTrainingReturn {
  const coreRef = useRef<DatasetTrainingState>(createInitialState(generateDataset()));
  const [state, setState] = useState<DatasetTrainingState>(coreRef.current);

  const publish = useCallback(() => {
    setState({ ...coreRef.current, results: [...coreRef.current.results] });
  }, []);

  /** Evaluate the current network on the dataset (also used after the network is replaced) */
  const refreshEvaluation = useCallback(() => {
    coreRef.current.evaluation = evaluateDataset(nnRef.current, coreRef.current.dataset);
  }, [nnRef]);

  const step = useCallback(() => {
    const core = coreRef.current;
    const nn = nnRef.current;
    if (core.dataset.length === 0) return;

    if (core.cursor >= core.dataset.length) {
      core.cursor = 0;
      core.pass += 1;
      core.results = Array<SampleResult | null>(core.dataset.length).fill(null);
    }

    const candidate = core.dataset[core.cursor];
    const inputs = candidateInputs(candidate);

    // What does the network think *before* seeing the answer?
    const probabilities = nn.predict(inputs);
    const predicted = argmax(probabilities);
    const correct = predicted === candidate.label;

    nn.train(inputs, toOneHot(candidate.label));
    const loss = nn.lastLoss;

    core.results[core.cursor] = { predicted, correct, probabilities, loss };
    core.processedTotal += 1;
    if (correct) core.onlineCorrect += 1;
    core.lastIndex = core.cursor;
    core.cursor += 1;
    refreshEvaluation();

    statsSetters.setLoss(loss);
    statsSetters.recordLoss(loss);
    displayInputs(inputs); // canvas shows this candidate flowing through the updated network
    publish();
  }, [nnRef, refreshEvaluation, statsSetters, displayInputs, publish]);

  const stepRef = useRef(step);
  useEffect(() => {
    stepRef.current = step;
  }, [step]);

  const setRunning = useCallback((running: boolean) => {
    if (coreRef.current.running === running) return;
    coreRef.current.running = running;
    publish();
  }, [publish]);

  const play = useCallback(() => {
    if (coreRef.current.evaluation === null) refreshEvaluation();
    setRunning(true);
  }, [setRunning, refreshEvaluation]);
  const pause = useCallback(() => setRunning(false), [setRunning]);
  const toggle = useCallback(() => setRunning(!coreRef.current.running), [setRunning]);

  const setSpeed = useCallback((speed: StreamSpeed) => {
    coreRef.current.speed = speed;
    publish();
  }, [publish]);

  const restart = useCallback(() => {
    const fresh = createInitialState(coreRef.current.dataset);
    fresh.speed = coreRef.current.speed;
    coreRef.current = fresh;
    refreshEvaluation();
    publish();
  }, [refreshEvaluation, publish]);

  // Drive the stream while running
  useEffect(() => {
    if (!state.running) return;
    const id = window.setInterval(() => stepRef.current(), BASE_INTERVAL_MS / state.speed);
    return () => window.clearInterval(id);
  }, [state.running, state.speed]);

  // Entering data mode: show the current network's accuracy right away. Leaving it: stop.
  useEffect(() => {
    if (enabled) {
      refreshEvaluation();
      publish();
    } else {
      setRunning(false);
    }
  }, [enabled, refreshEvaluation, publish, setRunning]);

  const refresh = useCallback(() => {
    refreshEvaluation();
    publish();
  }, [refreshEvaluation, publish]);

  return { ...state, play, pause, toggle, step, setSpeed, restart, refresh };
}
