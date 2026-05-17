import React, { useCallback, useEffect, useState } from 'react';
import { Cpu, Play, GitCommit, Upload, Loader2, RefreshCw } from 'lucide-react';
import toast from 'react-hot-toast';
import {
  getTrainingStatus,
  startTraining,
  commitModels,
  pushModels,
  trainCommitPush,
} from '../services/adminApi';

export default function AdminTrainingPanel({ title = 'Model training pipeline' }) {
  const [loading, setLoading] = useState(true);
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [trainMode, setTrainMode] = useState('quick');
  const [commitMsg, setCommitMsg] = useState('');

  const loadTraining = useCallback(async () => {
    try {
      const data = await getTrainingStatus();
      setTrainingStatus(data);
    } catch (e) {
      setTrainingStatus({ status: 'unavailable', error: e.message });
    }
  }, []);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      await loadTraining();
    } catch (e) {
      toast.error(e.response?.data?.detail || e.message || 'Failed to load status');
    } finally {
      setLoading(false);
    }
  }, [loadTraining]);

  useEffect(() => {
    refresh();
    const id = setInterval(loadTraining, 5000);
    return () => clearInterval(id);
  }, [refresh, loadTraining]);

  const runTraining = async (action) => {
    try {
      let result;
      if (action === 'train') result = await startTraining({ mode: trainMode });
      else if (action === 'commit') result = await commitModels(commitMsg || undefined);
      else if (action === 'push') result = await pushModels();
      else result = await trainCommitPush({ mode: trainMode, message: commitMsg || undefined });
      toast.success(result.message || 'Started');
      loadTraining();
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
        <button
          type="button"
          onClick={refresh}
          className="flex items-center gap-2 px-3 py-2 text-sm border rounded-lg dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-800"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {loading ? (
        <div className="flex justify-center py-8">
          <Loader2 className="w-8 h-8 animate-spin text-indigo-500" />
        </div>
      ) : (
        <>
          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5">
            <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Status</p>
            <pre className="text-xs bg-gray-50 dark:bg-gray-900 p-4 rounded-lg overflow-auto max-h-48 text-gray-800 dark:text-gray-200">
              {JSON.stringify(trainingStatus, null, 2)}
            </pre>
          </div>

          <div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-xl p-4 text-sm text-amber-900 dark:text-amber-200">
            Start the pipeline on this PC:{' '}
            <code className="bg-amber-100 dark:bg-amber-900/50 px-1 rounded">python train_pipe.py</code>
            <br />
            Backend <code className="px-1">.env</code>:{' '}
            <code className="px-1">TRAIN_PIPE_URL=http://127.0.0.1:8090</code>
          </div>

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
