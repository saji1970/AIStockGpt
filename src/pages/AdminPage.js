import React, { useCallback, useEffect, useState } from 'react';
import {
  Shield,
  Users,
  Cpu,
  RefreshCw,
  Plus,
  Trash2,
  Ban,
  CheckCircle,
  Key,
  Play,
  GitCommit,
  Upload,
  Loader2,
} from 'lucide-react';
import toast from 'react-hot-toast';
import {
  getAdminDashboard,
  listAdminUsers,
  createAdminUser,
  updateAdminUser,
  blockAdminUser,
  unblockAdminUser,
  deleteAdminUser,
  getTrainingStatus,
  startTraining,
  commitModels,
  pushModels,
  trainCommitPush,
} from '../services/adminApi';

const TABS = [
  { id: 'dashboard', label: 'Dashboard', icon: Shield },
  { id: 'users', label: 'Users', icon: Users },
  { id: 'training', label: 'Training', icon: Cpu },
];

function AdminPage() {
  const [tab, setTab] = useState('dashboard');
  const [dashboard, setDashboard] = useState(null);
  const [usersData, setUsersData] = useState({ users: [], total: 0 });
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(true);
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [trainMode, setTrainMode] = useState('quick');
  const [commitMsg, setCommitMsg] = useState('');
  const [showCreate, setShowCreate] = useState(false);
  const [newUser, setNewUser] = useState({
    email: '',
    password: '',
    first_name: '',
    last_name: '',
    is_admin: false,
    is_active: true,
  });

  const loadDashboard = useCallback(async () => {
    const data = await getAdminDashboard();
    setDashboard(data);
  }, []);

  const loadUsers = useCallback(async () => {
    const data = await listAdminUsers({ search: search || undefined, limit: 100 });
    setUsersData(data);
  }, [search]);

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
      if (tab === 'dashboard') await loadDashboard();
      if (tab === 'users') await loadUsers();
      if (tab === 'training') await loadTraining();
    } catch (e) {
      toast.error(e.response?.data?.detail || e.message || 'Request failed');
    } finally {
      setLoading(false);
    }
  }, [tab, loadDashboard, loadUsers, loadTraining]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    if (tab !== 'training') return undefined;
    const id = setInterval(loadTraining, 5000);
    return () => clearInterval(id);
  }, [tab, loadTraining]);

  const handleCreateUser = async (e) => {
    e.preventDefault();
    try {
      await createAdminUser(newUser);
      toast.success('User created');
      setShowCreate(false);
      setNewUser({ email: '', password: '', first_name: '', last_name: '', is_admin: false, is_active: true });
      loadUsers();
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Create failed');
    }
  };

  const handleBlock = async (userId, active) => {
    try {
      if (active) await blockAdminUser(userId);
      else await unblockAdminUser(userId);
      toast.success(active ? 'User blocked' : 'User unblocked');
      loadUsers();
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Action failed');
    }
  };

  const handleDelete = async (userId, email) => {
    if (!window.confirm(`Delete user ${email}? This cannot be undone.`)) return;
    try {
      await deleteAdminUser(userId);
      toast.success('User deleted');
      loadUsers();
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Delete failed');
    }
  };

  const handleToggleAdmin = async (user) => {
    try {
      await updateAdminUser(user.id, { is_admin: !user.is_admin });
      toast.success(user.is_admin ? 'Admin removed' : 'Promoted to admin');
      loadUsers();
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Update failed');
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
      loadTraining();
    } catch (err) {
      const detail = err.response?.data?.detail;
      toast.error(typeof detail === 'string' ? detail : JSON.stringify(detail) || 'Failed');
    }
  };

  return (
    <div className="flex-1 overflow-auto p-6 max-w-7xl mx-auto w-full">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100 flex items-center gap-2">
            <Shield className="w-7 h-7 text-indigo-600" />
            Admin Console
          </h2>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Users, API keys, and local training pipeline
          </p>
        </div>
        <button
          type="button"
          onClick={refresh}
          className="flex items-center gap-2 px-3 py-2 text-sm bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      <div className="flex gap-2 mb-6 border-b border-gray-200 dark:border-gray-700 pb-2">
        {TABS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            type="button"
            onClick={() => setTab(id)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              tab === id
                ? 'bg-indigo-600 text-white'
                : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800'
            }`}
          >
            <Icon className="w-4 h-4" />
            {label}
          </button>
        ))}
      </div>

      {loading && (
        <div className="flex justify-center py-12">
          <Loader2 className="w-8 h-8 animate-spin text-indigo-500" />
        </div>
      )}

      {!loading && tab === 'dashboard' && dashboard && (
        <div className="space-y-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { label: 'Total users', value: dashboard.total_users },
              { label: 'Predictions', value: dashboard.analytics?.total_predictions },
              { label: 'Daily active', value: dashboard.analytics?.daily_active_users },
              {
                label: 'Avg accuracy',
                value: dashboard.analytics?.average_prediction_accuracy
                  ? `${(dashboard.analytics.average_prediction_accuracy * 100).toFixed(1)}%`
                  : '—',
              },
            ].map((card) => (
              <div
                key={card.label}
                className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-4"
              >
                <p className="text-xs text-gray-500 uppercase tracking-wide">{card.label}</p>
                <p className="text-2xl font-semibold mt-1 text-gray-900 dark:text-gray-100">{card.value ?? '—'}</p>
              </div>
            ))}
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5">
            <h3 className="font-semibold flex items-center gap-2 mb-4 text-gray-900 dark:text-gray-100">
              <Key className="w-5 h-5 text-amber-500" />
              API keys & environment (masked)
            </h3>
            <div className="grid md:grid-cols-2 gap-2 text-sm">
              {Object.entries(dashboard.env_keys || {}).map(([name, info]) => (
                <div
                  key={name}
                  className="flex justify-between py-2 px-3 rounded-lg bg-gray-50 dark:bg-gray-900/50"
                >
                  <span className="font-mono text-gray-700 dark:text-gray-300">{name}</span>
                  <span className={info.configured ? 'text-green-600 dark:text-green-400' : 'text-gray-400'}>
                    {info.configured ? info.preview || 'set' : 'not set'}
                  </span>
                </div>
              ))}
            </div>
            <p className="text-xs text-gray-500 mt-4">
              Training pipe: {dashboard.train_pipe_url}
              {dashboard.train_pipe_configured ? ' (configured)' : ' — run: python train_pipe.py'}
            </p>
          </div>
        </div>
      )}

      {!loading && tab === 'users' && (
        <div className="space-y-4">
          <div className="flex flex-wrap gap-3 items-center justify-between">
            <input
              type="search"
              placeholder="Search email, name, username..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && loadUsers()}
              className="flex-1 min-w-[200px] px-3 py-2 border border-gray-200 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 dark:text-gray-100"
            />
            <button
              type="button"
              onClick={() => setShowCreate(!showCreate)}
              className="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg text-sm hover:bg-indigo-700"
            >
              <Plus className="w-4 h-4" />
              Add user
            </button>
          </div>

          {showCreate && (
            <form
              onSubmit={handleCreateUser}
              className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl p-4 grid md:grid-cols-2 gap-3"
            >
              {['email', 'password', 'first_name', 'last_name'].map((field) => (
                <input
                  key={field}
                  required
                  type={field === 'password' ? 'password' : 'text'}
                  placeholder={field.replace('_', ' ')}
                  value={newUser[field]}
                  onChange={(e) => setNewUser({ ...newUser, [field]: e.target.value })}
                  className="px-3 py-2 border rounded-lg dark:bg-gray-900 dark:border-gray-600 dark:text-gray-100"
                />
              ))}
              <label className="flex items-center gap-2 text-sm dark:text-gray-300">
                <input
                  type="checkbox"
                  checked={newUser.is_admin}
                  onChange={(e) => setNewUser({ ...newUser, is_admin: e.target.checked })}
                />
                Admin
              </label>
              <button type="submit" className="md:col-span-2 py-2 bg-green-600 text-white rounded-lg">
                Create user
              </button>
            </form>
          )}

          <p className="text-sm text-gray-500">{usersData.total} users</p>
          <div className="overflow-x-auto rounded-xl border border-gray-200 dark:border-gray-700">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 dark:bg-gray-900/80 text-left">
                <tr>
                  <th className="p-3">Email</th>
                  <th className="p-3">Name</th>
                  <th className="p-3">Status</th>
                  <th className="p-3">Role</th>
                  <th className="p-3">Last login</th>
                  <th className="p-3">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100 dark:divide-gray-700 bg-white dark:bg-gray-800">
                {(usersData.users || []).map((u) => (
                  <tr key={u.id}>
                    <td className="p-3 font-mono text-xs">{u.email}</td>
                    <td className="p-3">
                      {u.first_name} {u.last_name}
                    </td>
                    <td className="p-3">
                      <span
                        className={`px-2 py-0.5 rounded text-xs ${
                          u.is_active
                            ? 'bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300'
                            : 'bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300'
                        }`}
                      >
                        {u.is_active ? 'Active' : 'Blocked'}
                      </span>
                    </td>
                    <td className="p-3">{u.is_admin ? 'Admin' : 'User'}</td>
                    <td className="p-3 text-gray-500 text-xs">
                      {u.last_login ? new Date(u.last_login).toLocaleString() : '—'}
                    </td>
                    <td className="p-3">
                      <div className="flex flex-wrap gap-1">
                        <button
                          type="button"
                          title={u.is_active ? 'Block' : 'Unblock'}
                          onClick={() => handleBlock(u.id, u.is_active)}
                          className="p-1.5 rounded hover:bg-gray-100 dark:hover:bg-gray-700"
                        >
                          {u.is_active ? (
                            <Ban className="w-4 h-4 text-amber-600" />
                          ) : (
                            <CheckCircle className="w-4 h-4 text-green-600" />
                          )}
                        </button>
                        <button
                          type="button"
                          title="Toggle admin"
                          onClick={() => handleToggleAdmin(u)}
                          className="p-1.5 rounded hover:bg-gray-100 dark:hover:bg-gray-700 text-indigo-600 text-xs font-medium"
                        >
                          {u.is_admin ? '−Admin' : '+Admin'}
                        </button>
                        <button
                          type="button"
                          title="Delete"
                          onClick={() => handleDelete(u.id, u.email)}
                          className="p-1.5 rounded hover:bg-red-50 dark:hover:bg-red-900/30"
                        >
                          <Trash2 className="w-4 h-4 text-red-600" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {!loading && tab === 'training' && (
        <div className="space-y-6 max-w-2xl">
          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5">
            <h3 className="font-semibold mb-3 text-gray-900 dark:text-gray-100">Pipeline status</h3>
            <pre className="text-xs bg-gray-50 dark:bg-gray-900 p-4 rounded-lg overflow-auto max-h-48 text-gray-800 dark:text-gray-200">
              {JSON.stringify(trainingStatus, null, 2)}
            </pre>
          </div>

          <div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-xl p-4 text-sm text-amber-900 dark:text-amber-200">
            Run <code className="bg-amber-100 dark:bg-amber-900/50 px-1 rounded">python train_pipe.py</code> on
            this machine first. Set <code className="px-1">TRAIN_PIPE_URL=http://127.0.0.1:8090</code> and optional{' '}
            <code className="px-1">TRAIN_PIPE_SECRET</code> in backend <code className="px-1">.env</code>.
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
        </div>
      )}
    </div>
  );
}

export default AdminPage;
