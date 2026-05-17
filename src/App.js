import React, { useEffect, useState } from 'react';
import { Routes, Route, Navigate, Link, useLocation } from 'react-router-dom';
import { TrendingUp, MessageSquare, Briefcase, LogOut, BarChart3, Moon, Sun, Settings, Shield } from 'lucide-react';
import useAuthStore from './store/authStore';
import ChatPage from './pages/ChatPage';
import LoginPage from './pages/LoginPage';
import PortfolioPage from './pages/PortfolioPage';
import DashboardPage from './pages/DashboardPage';
import ForgotPasswordPage from './pages/ForgotPasswordPage';
import ResetPasswordPage from './pages/ResetPasswordPage';
import SettingsPage from './pages/SettingsPage';
import AdminPage from './pages/AdminPage';
import './App.css';

function ProtectedRoute({ children }) {
  const { isAuthenticated, isLoading } = useAuthStore();
  if (isLoading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <p className="text-gray-500">Loading...</p>
      </div>
    );
  }
  return isAuthenticated ? children : <Navigate to="/login" />;
}

function AdminRoute({ children }) {
  const { isAuthenticated, isLoading, user } = useAuthStore();
  if (isLoading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <p className="text-gray-500">Loading...</p>
      </div>
    );
  }
  if (!isAuthenticated) return <Navigate to="/login" />;
  if (!user?.is_admin) return <Navigate to="/" />;
  return children;
}

function App() {
  const { isAuthenticated, user, logout, initialize, isLoading } = useAuthStore();
  const location = useLocation();
  const [darkMode, setDarkMode] = useState(() => localStorage.getItem('darkMode') === 'true');

  useEffect(() => {
    initialize();
  }, [initialize]);

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
    localStorage.setItem('darkMode', darkMode);
  }, [darkMode]);

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900">
        <p className="text-gray-500 dark:text-gray-400">Loading...</p>
      </div>
    );
  }

  // Auth pages render without the app shell
  if (['/login', '/forgot-password', '/reset-password'].includes(location.pathname)) {
    return (
      <Routes>
        <Route path="/login" element={isAuthenticated ? <Navigate to="/" /> : <LoginPage />} />
        <Route path="/forgot-password" element={<ForgotPasswordPage />} />
        <Route path="/reset-password" element={<ResetPasswordPage />} />
      </Routes>
    );
  }

  return (
    <div className="h-screen bg-gray-50 dark:bg-gray-900 flex flex-col overflow-hidden">
      {/* Header */}
      <header className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-r from-primary-500 to-primary-600 rounded-lg flex items-center justify-center">
              <TrendingUp className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900 dark:text-gray-100">AI Stock GPT</h1>
              <p className="text-sm text-gray-600 dark:text-gray-400">Intelligent Stock Analysis</p>
            </div>
          </div>

          <nav className="flex items-center space-x-6">
            <Link
              to="/"
              className={`flex items-center space-x-2 text-sm font-medium transition-colors ${
                location.pathname === '/'
                  ? 'text-primary-600'
                  : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100'
              }`}
            >
              <MessageSquare className="w-4 h-4" />
              <span>Chat</span>
            </Link>
            <Link
              to="/dashboard"
              className={`flex items-center space-x-2 text-sm font-medium transition-colors ${
                location.pathname === '/dashboard'
                  ? 'text-primary-600'
                  : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100'
              }`}
            >
              <BarChart3 className="w-4 h-4" />
              <span>Dashboard</span>
            </Link>
            <Link
              to="/portfolio"
              className={`flex items-center space-x-2 text-sm font-medium transition-colors ${
                location.pathname === '/portfolio'
                  ? 'text-primary-600'
                  : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100'
              }`}
            >
              <Briefcase className="w-4 h-4" />
              <span>Portfolio</span>
            </Link>

            {/* Dark mode toggle */}
            <button
              onClick={() => setDarkMode(!darkMode)}
              className="text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 transition-colors"
              title={darkMode ? 'Light mode' : 'Dark mode'}
            >
              {darkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </button>

            <Link
              to="/settings"
              className={`flex items-center space-x-2 text-sm font-medium transition-colors ${
                location.pathname === '/settings'
                  ? 'text-primary-600'
                  : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100'
              }`}
            >
              <Settings className="w-4 h-4" />
              <span>Settings</span>
            </Link>

            {isAuthenticated && user?.is_admin && (
              <Link
                to="/admin"
                className={`flex items-center space-x-2 text-sm font-medium transition-colors ${
                  location.pathname === '/admin'
                    ? 'text-primary-600'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100'
                }`}
              >
                <Shield className="w-4 h-4" />
                <span>Admin</span>
              </Link>
            )}

            {isAuthenticated && (
              <div className="flex items-center space-x-3 ml-4 pl-4 border-l border-gray-200 dark:border-gray-600">
                <span className="text-sm text-gray-600 dark:text-gray-400">{user?.first_name}</span>
                <button
                  onClick={logout}
                  className="text-gray-400 hover:text-red-500 transition-colors"
                  title="Sign out"
                >
                  <LogOut className="w-4 h-4" />
                </button>
              </div>
            )}
          </nav>
        </div>
      </header>

      {/* Routes */}
      <Routes>
        <Route
          path="/"
          element={
            <ProtectedRoute>
              <ChatPage />
            </ProtectedRoute>
          }
        />
        <Route
          path="/dashboard"
          element={
            <ProtectedRoute>
              <DashboardPage />
            </ProtectedRoute>
          }
        />
        <Route
          path="/portfolio"
          element={
            <ProtectedRoute>
              <PortfolioPage />
            </ProtectedRoute>
          }
        />
        <Route
          path="/settings"
          element={
            <ProtectedRoute>
              <SettingsPage />
            </ProtectedRoute>
          }
        />
        <Route
          path="/admin"
          element={
            <AdminRoute>
              <AdminPage />
            </AdminRoute>
          }
        />
        <Route path="/login" element={<LoginPage />} />
        <Route path="*" element={<Navigate to="/" />} />
      </Routes>
    </div>
  );
}

export default App;
