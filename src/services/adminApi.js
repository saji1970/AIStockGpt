import api from './api';

export const getAdminDashboard = async () => {
  const { data } = await api.get('/admin/dashboard');
  return data;
};

export const listAdminUsers = async (params = {}) => {
  const { data } = await api.get('/admin/users', { params });
  return data;
};

export const getAdminUser = async (userId) => {
  const { data } = await api.get(`/admin/users/${userId}`);
  return data;
};

export const createAdminUser = async (payload) => {
  const { data } = await api.post('/admin/users', payload);
  return data;
};

export const updateAdminUser = async (userId, payload) => {
  const { data } = await api.patch(`/admin/users/${userId}`, payload);
  return data;
};

export const blockAdminUser = async (userId) => {
  const { data } = await api.post(`/admin/users/${userId}/block`);
  return data;
};

export const unblockAdminUser = async (userId) => {
  const { data } = await api.post(`/admin/users/${userId}/unblock`);
  return data;
};

export const deleteAdminUser = async (userId) => {
  const { data } = await api.delete(`/admin/users/${userId}`);
  return data;
};

export const getTrainingStatus = async () => {
  const { data } = await api.get('/admin/training/status');
  return data;
};

export const startTraining = async (payload = {}) => {
  const { data } = await api.post('/admin/training/train', payload);
  return data;
};

export const commitModels = async (message) => {
  const { data } = await api.post('/admin/training/commit', null, { params: message ? { message } : {} });
  return data;
};

export const pushModels = async (remote = 'origin', branch) => {
  const { data } = await api.post('/admin/training/push', null, {
    params: { remote, ...(branch ? { branch } : {}) },
  });
  return data;
};

export const trainCommitPush = async (payload = {}) => {
  const { data } = await api.post('/admin/training/train-commit-push', payload);
  return data;
};
