import React from 'react';
import { TrendingUp, TrendingDown, DollarSign, Percent, Activity } from 'lucide-react';

const StockCard = ({ data }) => {
  if (!data) return null;

  const {
    symbol,
    currentPrice,
    predictedPrice,
    priceChange,
    priceChangePercent,
    prediction,
    confidence,
    metrics
  } = data;

  const isPositive = priceChange >= 0;

  return (
    <div className="stock-card">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h4 className="text-lg font-bold text-gray-900">{symbol}</h4>
          <p className="text-sm text-gray-600">Stock Analysis</p>
        </div>
        <div className={`p-2 rounded-full ${isPositive ? 'bg-green-100' : 'bg-red-100'}`}>
          {isPositive ? (
            <TrendingUp className="w-5 h-5 text-green-600" />
          ) : (
            <TrendingDown className="w-5 h-5 text-red-600" />
          )}
        </div>
      </div>

      {/* Current Price */}
      <div className="mb-4">
        <div className="metric-label">Current Price</div>
        <div className="metric-value text-gray-900">${currentPrice?.toFixed(2)}</div>
      </div>

      {/* Predicted Price */}
      <div className="mb-4">
        <div className="metric-label">Predicted Price</div>
        <div className="metric-value text-primary-600">${predictedPrice?.toFixed(2)}</div>
      </div>

      {/* Price Change */}
      <div className="mb-4">
        <div className="metric-label">Price Change</div>
        <div className={`metric-value ${isPositive ? 'positive-change' : 'negative-change'}`}>
          {isPositive ? '+' : ''}${priceChange?.toFixed(2)} ({priceChangePercent?.toFixed(2)}%)
        </div>
      </div>

      {/* Prediction */}
      <div className="mb-4">
        <div className="metric-label">Prediction</div>
        <div className={`text-lg font-semibold ${isPositive ? 'positive-change' : 'negative-change'}`}>
          {prediction}
        </div>
      </div>

      {/* Confidence */}
      {confidence && (
        <div className="mb-4">
          <div className="metric-label">Model Confidence</div>
          <div className="flex items-center space-x-2">
            <div className="flex-1 bg-gray-200 rounded-full h-2">
              <div 
                className="bg-primary-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${confidence}%` }}
              ></div>
            </div>
            <span className="text-sm font-medium text-gray-700">{confidence}%</span>
          </div>
        </div>
      )}

      {/* Key Metrics */}
      {metrics && (
        <div className="border-t border-gray-200 pt-4">
          <h5 className="text-sm font-semibold text-gray-900 mb-3">Key Metrics</h5>
          <div className="space-y-2">
            {Object.entries(metrics).map(([key, value]) => (
              <div key={key} className="flex justify-between items-center">
                <span className="text-sm text-gray-600 capitalize">
                  {key.replace(/([A-Z])/g, ' $1').trim()}
                </span>
                <span className="text-sm font-medium text-gray-900">
                  {typeof value === 'number' ? value.toFixed(2) : value}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Disclaimer */}
      <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
        <p className="text-xs text-yellow-800">
          ⚠️ This analysis is for informational purposes only. Always do your own research before making investment decisions.
        </p>
      </div>
    </div>
  );
};

export default StockCard;
