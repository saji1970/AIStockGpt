import React, { useCallback, useEffect, useState } from 'react';
import { Cpu, Play, GitCommit, Upload, Loader2, RefreshCw, Power, Square, CheckCircle2, XCircle, Clock, Activity } from 'lucide-react';
import toast from 'react-hot-toast';
import {
  getTrainingStatus,
  startTraining,
  commitModels,
  pushModels,
  trainCommitPush,
  getPipelineStatus,
  startPipeline,
  stopPipeline,
} from '../services/adminApi';

export default function AdminTrainingPanel({ title = 'Model training pipeline' }) {
  const [loading, setLoading] = useState(true);
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [pipelineRunning, setPipelineRunning] = useState(false);
  const [pipelineManaged, setPipelineManaged] = useState(false);
  const [startingPipeline, setStartingPipeline] = useState(false);
  const [stoppingPipeline, setStoppingPipeline] = useState(false);
  const [trainMode, setTrainMode] = useState('quick');
  const [commitMsg, setCommitMsg] = useState('');

  const pollStatus = useCallback(async () => {
    // Check pipeline process status
    let plRunning = false;
    let plManaged = false;
    try {
      const data = await getPipelineStatus();
      plRunning = data.running;
      plManaged = data.managed;
    } catch {
      // endpoint failed - don't assume not running yet
    }

    // Check training status (also proves pipeline is reachable)
    let tStatus = null;
    try {
      tStatus = await getTrainingStatus();
    } catch (e) {
      tStatus = { status: 'unavailable', error: e.message };
    }

    // If training status came back successfully, the pipeline IS running
    // even if the pipeline-status endpoint failed
    const trainingReachable = tStatus && tStatus.status !== 'unavailable';
    setPipelineRunning(plRunning || trainingReachable);
    setPipelineManaged(plManaged);
    setTrainingStatus(tStatus);
  }, []);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      await pollStatus();
    } catch (e) {
      toast.error(e.response?.data?.detail || e.message || 'Failed to load status');
    } finally {
      setLoading(false);
    }
  }, [pollStatus]);

  useEffect(() => {
    refresh();
    const id = setInterval(pollStatus, 5000);
    return () => clearInterval(id);
  }, [refresh, pollStatus]);

  const handleStartPipeline = async () => {
    setStartingPipeline(true);
    try {
      const result = await startPipeline();
      toast.success(result.message || 'Pipeline started');
      await refresh();
    } catch (err) {
      const detail = err.response?.data?.detail;
      toast.error(typeof detail === 'string' ? detail : 'Failed to start pipeline');
    } finally {
      setStartingPipeline(false);
    }
  };

  const handleStopPipeline = async () => {
    setStoppingPipeline(true);
    try {
      const result = await stopPipeline();
      toast.success(result.message || 'Pipeline stopped');
      await refresh();
    } catch (err) {
      const detail = err.response?.data?.detail;
      toast.error(typeof detail === 'string' ? detail : 'Failed to stop pipeline');
    } finally {
      setStoppingPipeline(false);
    }
  };

  const runTraining = async (action) => {
    try {
      let result;
      if (action === 'train') result = await startTraining({ mode: trainMode });
      else if (action === 'commit') result = await commitModels(commitMsg || undefined);
      else if (action === 'push') result = await pushModels();
      else result = await trainCommitPush({ mode: trainMode, message: commitMsg || undefined });
      toast.success(result.message || 'Started');
      pollStatus();
    } catch (err) {
      const detail = err.response?.data?.detail;
      toast.error(typeof detail === 'string' ? detail : JSON.stringify(detail) || 'Failed');
    }
  };

  const formatElapsed = (seconds) => {
    if (!seconds) return '0s';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    if (mins === 0) return `${secs}s`;
    return `${mins}m ${secs}s`;
  };

  return (
    <div className="space-y-6 max-w-2xl">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 flex items-center gap-2">
          <Cpu className="w-5 h-5 text-indigo-600" />
          {title}
        </h3>
        <div className="flex items-center gap-2">
          {pipelineRunning && pipelineManaged && (
            <button
              type="button"
              onClick={handleStopPipeline}
              disabled={stoppingPipeline}
              className="flex items-center gap-2 px-3 py-2 text-sm bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-700 text-red-700 dark:text-red-300 rounded-lg hover:bg-red-100 dark:hover:bg-red-900/50 disabled:opacity-50"
            >
              {stoppingPipeline ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Square className="w-4 h-4" />
              )}
              Stop Pipeline
            </button>
          )}
          <button
            type="button"
            onClick={refresh}
            className="flex items-center gap-2 px-3 py-2 text-sm border rounded-lg dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-800"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      {loading ? (
        <div className="flex justify-center py-8">
          <Loader2 className="w-8 h-8 animate-spin text-indigo-500" />
        </div>
      ) : !pipelineRunning ? (
        /* ── Pipeline not running ── */
        <div className="bg-gray-50 dark:bg-gray-800/50 border border-gray-200 dark:border-gray-700 rounded-xl p-8 text-center space-y-4">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-gray-200 dark:bg-gray-700">
            <Power className="w-8 h-8 text-gray-500 dark:text-gray-400" />
          </div>
          <p className="text-lg font-semibold text-gray-700 dark:text-gray-200">
            Training pipeline is not running
          </p>
          <p className="text-sm text-gray-500 dark:text-gray-400 max-w-md mx-auto">
            The training service needs to be started before you can train models,
            commit, or push to GitHub.
          </p>
          <button
            type="button"
            onClick={handleStartPipeline}
            disabled={startingPipeline}
            className="inline-flex items-center gap-2 px-6 py-3 bg-green-600 text-white rounded-lg text-sm font-medium hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {startingPipeline ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Starting pipeline...
              </>
            ) : (
              <>
                <Power className="w-5 h-5" />
                Start Pipeline
              </>
            )}
          </button>
          <p className="text-xs text-gray-400 dark:text-gray-500">
            Or run manually: <code className="bg-gray-200 dark:bg-gray-700 px-1 rounded">python train_pipe.py</code>
          </p>
        </div>
      ) : (
        /* ── Pipeline running - show training controls ── */
        <>
          <div className="flex items-center gap-2 text-sm text-green-700 dark:text-green-400 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg px-4 py-2">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            Pipeline is running
            {pipelineManaged && <span className="text-xs text-green-600 dark:text-green-500 ml-1">(managed)</span>}
          </div>

          {trainingStatus && trainingStatus.status !== 'unavailable' && (
            <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5 space-y-4">
              {/* Status badge + elapsed time */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {trainingStatus.status === 'idle' && (
                    <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300">
                      <div className="w-1.5 h-1.5 rounded-full bg-gray-400" />
                      Idle
                    </span>
                  )}
                  {trainingStatus.status === 'training' && (
                    <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-indigo-100 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300">
                      <Loader2 className="w-3 h-3 animate-spin" />
                      Training
                    </span>
                  )}
                  {trainingStatus.status === 'done' && (
                    <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300">
                      <CheckCircle2 className="w-3 h-3" />
                      Completed
                    </span>
                  )}
                  {trainingStatus.status === 'failed' && (
                    <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300">
                      <XCircle className="w-3 h-3" />
                      Failed
                    </span>
                  )}
                  {trainingStatus.mode && (
                    <span className="text-xs text-gray-500 dark:text-gray-400">
                      ({trainingStatus.mode} mode)
                    </span>
                  )}
                </div>
                {trainingStatus.elapsed > 0 && (
                  <div className="flex items-center gap-1 text-xs text-gray-500 dark:text-gray-400">
                    <Clock className="w-3 h-3" />
                    {formatElapsed(trainingStatus.elapsed)}
                  </div>
                )}
              </div>

              {/* Progress bar */}
              {trainingStatus.total > 0 && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-gray-600 dark:text-gray-400">
                      {trainingStatus.progress}/{trainingStatus.total} symbols
                    </span>
                    <span className="font-medium text-gray-700 dark:text-gray-300">
                      {Math.round((trainingStatus.progress / trainingStatus.total) * 100)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5 overflow-hidden">
                    <div
                      className={`h-2.5 rounded-full transition-all duration-500 ease-out ${
                        trainingStatus.status === 'failed'
                          ? 'bg-red-500'
                          : trainingStatus.status === 'done'
                          ? 'bg-green-500'
                          : 'bg-indigo-500'
                      }`}
                      style={{ width: `${(trainingStatus.progress / trainingStatus.total) * 100}%` }}
                    />
                  </div>
                </div>
              )}

              {/* Current symbol being trained */}
              {trainingStatus.status === 'training' && trainingStatus.current_symbol && (
                <div className="flex items-center gap-2 text-sm text-indigo-700 dark:text-indigo-300 bg-indigo-50 dark:bg-indigo-900/20 rounded-lg px-3 py-2">
                  <Activity className="w-4 h-4 animate-pulse" />
                  <span>
                    Training <span className="font-semibold">{trainingStatus.current_symbol}</span>...
                  </span>
                </div>
              )}

              {/* Results summary */}
              {trainingStatus.results && trainingStatus.results.total > 0 && (
                <div className="flex items-center gap-4 text-sm">
                  <div className="flex items-center gap-1.5 text-green-600 dark:text-green-400">
                    <CheckCircle2 className="w-4 h-4" />
                    <span className="font-medium">{trainingStatus.results.succeeded}</span>
                    <span className="text-gray-500 dark:text-gray-400 text-xs">succeeded</span>
                  </div>
                  {trainingStatus.results.failed > 0 && (
                    <div className="flex items-center gap-1.5 text-red-600 dark:text-red-400">
                      <XCircle className="w-4 h-4" />
                      <span className="font-medium">{trainingStatus.results.failed}</span>
                      <span className="text-gray-500 dark:text-gray-400 text-xs">failed</span>
                    </div>
                  )}
                </div>
              )}

              {/* Failed symbols list */}
              {trainingStatus.results && trainingStatus.results.symbols_failed && trainingStatus.results.symbols_failed.length > 0 && (
                <div className="text-xs text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/10 rounded-lg px-3 py-2">
                  <span className="font-medium">Failed:</span>{' '}
                  {trainingStatus.results.symbols_failed.join(', ')}
                </div>
              )}

              {/* Error message */}
              {trainingStatus.status === 'failed' && trainingStatus.error && (
                <div className="text-xs text-red-700 dark:text-red-300 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg px-3 py-2">
                  <span className="font-medium">Error:</span> {trainingStatus.error}
                </div>
              )}

              {/* Idle state message */}
              {trainingStatus.status === 'idle' && (!trainingStatus.results || !trainingStatus.results.total) && (
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  No training in progress. Use the controls below to start training.
                </p>
              )}
            </div>
          )}

          <div className="flex flex-wrap gap-3 items-end">
            <label className="text-sm dark:text-gray-300">
              Mode
              <select
                value={trainMode}
                onChange={(e) => setTrainMode(e.target.value)}
                className="block mt-1 px-3 py-2 border rounded-lg dark:bg-gray-800 dark:border-gray-600"
              >
                <option value="quick">Quick (14 symbols)</option>
                <option value="full">Full (all symbols)</option>
              </select>
            </label>
            <input
              type="text"
              placeholder="Commit message (optional)"
              value={commitMsg}
              onChange={(e) => setCommitMsg(e.target.value)}
              className="flex-1 min-w-[180px] px-3 py-2 border rounded-lg dark:bg-gray-800 dark:border-gray-600 dark:text-gray-100"
            />
          </div>

          <div className="flex flex-wrap gap-3">
            <button
              type="button"
              onClick={() => runTraining('train')}
              className="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg text-sm hover:bg-indigo-700"
            >
              <Play className="w-4 h-4" />
              Train
            </button>
            <button
              type="button"
              onClick={() => runTraining('commit')}
              className="flex items-center gap-2 px-4 py-2 bg-gray-700 text-white rounded-lg text-sm hover:bg-gray-800"
            >
              <GitCommit className="w-4 h-4" />
              Commit models
            </button>
            <button
              type="button"
              onClick={() => runTraining('push')}
              className="flex items-center gap-2 px-4 py-2 bg-gray-700 text-white rounded-lg text-sm hover:bg-gray-800"
            >
              <Upload className="w-4 h-4" />
              Push to GitHub
            </button>
            <button
              type="button"
              onClick={() => runTraining('all')}
              className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg text-sm hover:bg-green-700"
            >
              <Cpu className="w-4 h-4" />
              Train + commit + push
            </button>
          </div>
        </>
      )}
    </div>
  );
}
