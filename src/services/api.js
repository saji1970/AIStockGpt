import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

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

export default api;
