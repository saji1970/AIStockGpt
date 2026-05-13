import axios, {AxiosInstance} from 'axios';
import {API_BASE_URL} from '../config';
import {getAccessToken} from '../storage/tokenStorage';

const api: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  headers: {'Content-Type': 'application/json'},
  timeout: 60000,
});

// Attach JWT token to every request
api.interceptors.request.use(async config => {
  const token = await getAccessToken();
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// ---- Auth ----

export async function login(email: string, password: string) {
  const res = await api.post('/auth/login', {email, password});
  return res.data; // { access_token, refresh_token, token_type, expires_in }
}

export async function register(
  email: string,
  password: string,
  firstName: string,
  lastName: string,
  username?: string,
) {
  const res = await api.post('/auth/register', {
    email,
    password,
    first_name: firstName,
    last_name: lastName,
    username,
  });
  return res.data;
}

export async function refreshToken(token: string) {
  const res = await api.post('/auth/refresh', {refresh_token: token});
  return res.data;
}

export async function getProfile() {
  const res = await api.get('/auth/me');
  return res.data;
}

// ---- Password Management ----

export async function changePassword(currentPassword: string, newPassword: string) {
  const res = await api.post('/auth/change-password', {
    current_password: currentPassword,
    new_password: newPassword,
  });
  return res.data;
}

export async function forgotPassword(email: string) {
  const res = await api.post('/auth/forgot-password', {email});
  return res.data;
}

export async function resetPasswordWithToken(token: string, newPassword: string) {
  const res = await api.post('/auth/reset-password', {
    token,
    new_password: newPassword,
  });
  return res.data;
}

// ---- Chat ----

export async function sendMessage(message: string) {
  const res = await api.post('/chat', {
    message,
    timestamp: new Date().toISOString(),
  });
  return res.data; // { message, stockData, charts, confidence }
}

export async function getChatHistory() {
  const res = await api.get('/chat/history');
  return res.data; // { history: [...] }
}

// ---- Predictions & Analysis ----

export async function getPrediction(symbol: string, daysAhead: number = 5) {
  const res = await api.get(`/predict/${symbol}`, {params: {days_ahead: daysAhead}});
  return res.data;
}

export async function getTechnical(symbol: string) {
  const res = await api.get(`/technical/${symbol}`);
  return res.data;
}

// ---- Portfolios ----

export async function createPortfolio(name: string, description?: string) {
  const res = await api.post('/portfolio/create', {name, description});
  return res.data;
}

export async function listPortfolios() {
  const res = await api.get('/portfolio/list');
  return res.data; // { portfolios: [...] }
}

export async function getPortfolio(portfolioId: string) {
  const res = await api.get(`/portfolio/${portfolioId}`);
  return res.data;
}

export async function getPortfolioSummary(portfolioId: string) {
  const res = await api.get(`/portfolio/${portfolioId}/summary`);
  return res.data;
}

export async function deleteStockFromPortfolio(portfolioId: string, symbol: string) {
  const res = await api.delete(`/portfolio/${portfolioId}/stock/${symbol}`);
  return res.data;
}

export async function addStockToPortfolio(
  portfolioId: string,
  symbol: string,
  shares: number,
  purchasePrice: number,
  purchaseDate: string,
) {
  const res = await api.post(`/portfolio/${portfolioId}/add-stock`, {
    symbol,
    shares,
    purchase_price: purchasePrice,
    purchase_date: purchaseDate,
  });
  return res.data;
}

// ---- Analytics ----

export async function getUserAnalytics() {
  const res = await api.get('/analytics/user');
  return res.data;
}

// ---- Alerts ----

export async function createAlert(
  symbol: string,
  alertType: string,
  threshold: number,
  email: string,
) {
  const res = await api.post('/alerts/create', {
    symbol,
    alert_type: alertType,
    threshold,
    email,
  });
  return res.data;
}

export async function listAlerts() {
  const res = await api.get('/alerts/list');
  return res.data; // { alerts: [...] }
}

// ---- Health ----

export async function getHealth() {
  const res = await api.get('/health');
  return res.data;
}

// ---- ML-powered endpoints ----

export async function getMarketSummary() {
  const res = await api.get('/market-summary');
  return res.data;
}

export async function getPortfolioRecommendation(data: {
  amount: number;
  risk_level: string;
  horizon_months: number;
  goals?: string;
}) {
  const res = await api.post('/portfolio/recommend', data);
  return res.data;
}

export async function getRiskAnalysis(data: {
  portfolio_id?: string;
  symbols?: string[];
  weights?: number[];
}) {
  const res = await api.post('/risk-analysis', data);
  return res.data;
}

export async function getSentiment(symbol: string) {
  const res = await api.get(`/sentiment/${symbol}`);
  return res.data;
}

export async function getForecast(data: {
  symbols: string[];
  weights: number[];
  amount: number;
  months: number;
}) {
  const res = await api.post('/forecast', data);
  return res.data;
}

export default api;
