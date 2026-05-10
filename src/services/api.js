import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log('API Request:', config.method?.toUpperCase(), config.url);
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    console.log('API Response:', response.status, response.config.url);
    return response;
  },
  (error) => {
    console.error('API Error:', error.response?.status, error.response?.data);
    return Promise.reject(error);
  }
);

export const sendMessage = async (message) => {
  try {
    const response = await api.post('/chat', {
      message: message,
      timestamp: new Date().toISOString()
    });
    
    return response.data;
  } catch (error) {
    console.error('Error sending message:', error);
    throw new Error(error.response?.data?.detail || 'Failed to send message');
  }
};

export const getStockPrediction = async (symbol) => {
  try {
    const response = await api.get(`/predict/${symbol}`);
    return response.data;
  } catch (error) {
    console.error('Error getting stock prediction:', error);
    throw new Error(error.response?.data?.detail || 'Failed to get stock prediction');
  }
};

export const getTechnicalAnalysis = async (symbol) => {
  try {
    const response = await api.get(`/technical/${symbol}`);
    return response.data;
  } catch (error) {
    console.error('Error getting technical analysis:', error);
    throw new Error(error.response?.data?.detail || 'Failed to get technical analysis');
  }
};

export const getSensitivityAnalysis = async (symbol) => {
  try {
    const response = await api.get(`/sensitivity/${symbol}`);
    return response.data;
  } catch (error) {
    console.error('Error getting sensitivity analysis:', error);
    throw new Error(error.response?.data?.detail || 'Failed to get sensitivity analysis');
  }
};

export const getModelStatus = async () => {
  try {
    const response = await api.get('/status');
    return response.data;
  } catch (error) {
    console.error('Error getting model status:', error);
    throw new Error(error.response?.data?.detail || 'Failed to get model status');
  }
};

export default api;
