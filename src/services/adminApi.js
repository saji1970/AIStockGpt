import api from './api';

export const getAdminDashboard = async () => {
  const { data } = await api.get('/api/admin/dashboard');
  return data;
};

export const listAdminUsers = async (params = {}) => {
  const { data } = await api.get('/api/admin/users', { params });
  return data;
};

export const getAdminUser = async (userId) => {
  const { data } = await api.get(`/api/admin/users/${userId}`);
  return data;
};

export const createAdminUser = async (payload) => {
  const { data } = await api.post('/api/admin/users', payload);
  return data;
};

export const updateAdminUser = async (userId, payload) => {
  const { data } = await api.patch(`/api/admin/users/${userId}`, payload);
  return data;
};

export const blockAdminUser = async (userId) => {
  const { data } = await api.post(`/api/admin/users/${userId}/block`);
  return data;
};

export const unblockAdminUser = async (userId) => {
  const { data } = await api.post(`/api/admin/users/${userId}/unblock`);
  return data;
};

export const deleteAdminUser = async (userId) => {
  const { data } = await api.delete(`/api/admin/users/${userId}`);
  return data;
};

export const getTrainingStatus = async () => {
  const { data } = await api.get('/api/admin/training/status');
  return data;
};

export const startTraining = async (payload = {}) => {
  const { data } = await api.post('/api/admin/training/train', payload);
  return data;
};

export const commitModels = async (message) => {
  const { data } = await api.post('/api/admin/training/commit', null, { params: message ? { message } : {} });
  return data;
};

export const pushModels = async (remote = 'origin', branch) => {
  const { data } = await api.post('/api/admin/training/push', null, {
    params: { remote, ...(branch ? { branch } : {}) },
  });
  return data;
};

export const trainCommitPush = async (payload = {}) => {
  const { data } = await api.post('/api/admin/training/train-commit-push', payload);
  return data;
};

export const getPipelineStatus = async () => {
  const { data } = await api.get('/api/admin/training/pipeline-status');
  return data;
};

export const startPipeline = async () => {
  const { data } = await api.post('/api/admin/training/start-pipeline');
  return data;
};

export const stopPipeline = async () => {
  const { data } = await api.post('/api/admin/training/stop-pipeline');
  return data;
};
