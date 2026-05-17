import axios from 'axios';
import { API_BASE_URL } from '../config';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor - attach JWT token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('access_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor - handle 401
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('access_token');
      localStorage.removeItem('refresh_token');
      if (window.location.pathname !== '/login') {
        window.location.href = '/login';
      }
    }
    return Promise.reject(error);
  }
);

// Chat
export const sendMessage = async (message) => {
  try {
    const response = await api.post('/chat', {
      message: message,
      timestamp: new Date().toISOString()
    });
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Failed to send message');
  }
};

// Predictions & Analysis
export const getStockPrediction = async (symbol) => {
  try {
    const response = await api.get(`/predict/${symbol}`);
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Failed to get stock prediction');
  }
};

export const getTechnicalAnalysis = async (symbol) => {
  try {
    const response = await api.get(`/technical/${symbol}`);
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Failed to get technical analysis');
  }
};

export const getSensitivityAnalysis = async (symbol) => {
  try {
    const response = await api.get(`/sensitivity/${symbol}`);
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Failed to get sensitivity analysis');
  }
};

export const getModelStatus = async () => {
  try {
    const response = await api.get('/status');
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Failed to get model status');
  }
};

// Auth
export const loginUser = async (email, password) => {
  const response = await api.post('/auth/login', { email, password });
  return response.data;
};

export const registerUser = async (data) => {
  const response = await api.post('/auth/register', data);
  return response.data;
};

export const getProfile = async () => {
  const response = await api.get('/auth/me');
  return response.data;
};

// Password management
export const changePassword = async (currentPassword, newPassword) => {
  const response = await api.post('/auth/change-password', {
    current_password: currentPassword,
    new_password: newPassword,
  });
  return response.data;
};

export const forgotPassword = async (email) => {
  const response = await api.post('/auth/forgot-password', { email });
  return response.data;
};

export const resetPassword = async (token, newPassword) => {
  const response = await api.post('/auth/reset-password', {
    token,
    new_password: newPassword,
  });
  return response.data;
};

// Portfolio
export const createPortfolio = async (name, description) => {
  const response = await api.post('/portfolio/create', { name, description });
  return response.data;
};

export const listPortfolios = async () => {
  const response = await api.get('/portfolio/list');
  return response.data;
};

export const getPortfolio = async (portfolioId) => {
  const response = await api.get(`/portfolio/${portfolioId}`);
  return response.data;
};

export const getPortfolioSummary = async (portfolioId) => {
  const response = await api.get(`/portfolio/${portfolioId}/summary`);
  return response.data;
};

export const addStockToPortfolio = async (portfolioId, symbol, shares, purchasePrice, purchaseDate) => {
  const response = await api.post(`/portfolio/${portfolioId}/add-stock`, {
    symbol,
    shares,
    purchase_price: purchasePrice,
    purchase_date: purchaseDate,
  });
  return response.data;
};

export const deleteStockFromPortfolio = async (portfolioId, symbol) => {
  const response = await api.delete(`/portfolio/${portfolioId}/stock/${symbol}`);
  return response.data;
};

// ML-powered endpoints
export const getMarketSummary = async () => {
  const response = await api.get('/market-summary');
  return response.data;
};

export const getPortfolioRecommendation = async (data) => {
  const response = await api.post('/portfolio/recommend', data);
  return response.data;
};

export const getRiskAnalysis = async (data) => {
  const response = await api.post('/risk-analysis', data);
  return response.data;
};

export const getSentiment = async (symbol) => {
  const response = await api.get(`/sentiment/${symbol}`);
  return response.data;
};

export const getForecast = async (data) => {
  const response = await api.post('/forecast', data);
  return response.data;
};

export default api;
