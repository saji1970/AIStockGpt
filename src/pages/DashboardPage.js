import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { TrendingUp, TrendingDown, Activity, DollarSign } from 'lucide-react';
import { getMarketSummary } from '../services/api';

function DashboardPage() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      setLoading(true);
      const result = await getMarketSummary();
      setData(result);
    } catch (err) {
      setError(err.message || 'Failed to load market data');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center">
          <Activity className="w-8 h-8 text-indigo-500 animate-spin mx-auto mb-2" />
          <p className="text-gray-500 dark:text-gray-400">Loading market data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center">
          <p className="text-red-500 mb-2">{error}</p>
          <button onClick={fetchData} className="text-indigo-600 hover:underline text-sm">
            Retry
          </button>
        </div>
      </div>
    );
  }

  const indices = data?.indices || {};
  const macro = data?.macro_indicators || {};
  const sectorSentiment = data?.sector_sentiment || {};
  const regime = data?.regime || 'neutral';

  const sectorData = Object.entries(sectorSentiment).map(([symbol, info]) => ({
    name: symbol,
    sentiment: parseFloat((info.sentiment * 100).toFixed(1)),
    label: info.label,
  }));

  const regimeColors = {
    bullish: 'text-green-600 bg-green-50 dark:bg-green-900/20',
    bearish: 'text-red-600 bg-red-50 dark:bg-red-900/20',
    neutral: 'text-yellow-600 bg-yellow-50 dark:bg-yellow-900/20',
  };

  const macroLabels = {
    DFF: 'Fed Funds Rate',
    CPIAUCSL: 'CPI',
    UNRATE: 'Unemployment',
    T10Y2Y: '10Y-2Y Spread',
  };

  return (
    <div className="flex-1 overflow-y-auto p-6">
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Market Regime */}
        <div className="flex items-center justify-between">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Market Dashboard</h2>
          <span className={`px-3 py-1 rounded-full text-sm font-medium ${regimeColors[regime]}`}>
            {regime.charAt(0).toUpperCase() + regime.slice(1)} Market
          </span>
        </div>

        {/* Index Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {Object.entries(indices).map(([symbol, info]) => {
            const isUp = info.change >= 0;
            return (
              <div key={symbol} className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-500 dark:text-gray-400">
                    {info.name || symbol}
                  </span>
                  {isUp ? (
                    <TrendingUp className="w-4 h-4 text-green-500" />
                  ) : (
                    <TrendingDown className="w-4 h-4 text-red-500" />
                  )}
                </div>
                <div className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                  ${info.price?.toFixed(2)}
                </div>
                <div className={`text-sm mt-1 ${isUp ? 'text-green-600' : 'text-red-600'}`}>
                  {isUp ? '+' : ''}{info.change?.toFixed(2)} ({isUp ? '+' : ''}{info.changePercent?.toFixed(2)}%)
                </div>
              </div>
            );
          })}
        </div>

        {/* Macro Indicators */}
        {Object.keys(macro).length > 0 && (
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold text-gray-700 dark:text-gray-300 mb-3">
              Macro Indicators
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {Object.entries(macro).map(([key, info]) => (
                <div key={key} className="text-center p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                  <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                    {macroLabels[key] || key}
                  </div>
                  <div className="text-xl font-bold text-gray-900 dark:text-gray-100">
                    {typeof info.value === 'number' ? info.value.toFixed(2) : info.value}
                  </div>
                  <div className="text-xs text-gray-400">{info.date}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Sector Sentiment */}
        {sectorData.length > 0 && (
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold text-gray-700 dark:text-gray-300 mb-3">
              Sector Sentiment
            </h3>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={sectorData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="name" />
                <YAxis tickFormatter={(v) => `${v}%`} />
                <Tooltip formatter={(value) => [`${value}%`, 'Sentiment']} />
                <Bar
                  dataKey="sentiment"
                  fill="#6366f1"
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  );
}

export default DashboardPage;
