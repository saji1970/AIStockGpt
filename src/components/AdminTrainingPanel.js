import React, { useCallback, useEffect, useState } from 'react';
import { Cpu, Play, GitCommit, Upload, Loader2, RefreshCw, Power, Square } from 'lucide-react';
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
            <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5">
              <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Training Status</p>
              <pre className="text-xs bg-gray-50 dark:bg-gray-900 p-4 rounded-lg overflow-auto max-h-48 text-gray-800 dark:text-gray-200">
                {JSON.stringify(trainingStatus, null, 2)}
              </pre>
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
