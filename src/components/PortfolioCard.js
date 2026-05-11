import React from 'react';
import { Briefcase } from 'lucide-react';

export default function PortfolioCard({ name, description, totalValue, totalGainLoss, stockCount, onClick }) {
  const isPositive = totalGainLoss >= 0;

  return (
    <div
      onClick={onClick}
      className="bg-white border border-gray-200 rounded-xl p-5 cursor-pointer hover:shadow-md hover:border-primary-200 transition-all"
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center space-x-2">
          <Briefcase className="w-5 h-5 text-primary-500" />
          <h3 className="text-lg font-bold text-gray-900">{name}</h3>
        </div>
        <span className="text-xs bg-primary-50 text-primary-700 px-2 py-1 rounded-full font-medium">
          {stockCount} {stockCount === 1 ? 'holding' : 'holdings'}
        </span>
      </div>
      {description && <p className="text-sm text-gray-500 mb-3">{description}</p>}
      <div className="flex justify-between items-end mt-3">
        <div>
          <p className="text-xs text-gray-500">Total Value</p>
          <p className="text-lg font-bold text-gray-900">
            ${(totalValue || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </p>
        </div>
        <div className="text-right">
          <p className="text-xs text-gray-500">Gain / Loss</p>
          <p className={`text-lg font-bold ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
            {isPositive ? '+' : ''}${(totalGainLoss || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </p>
        </div>
      </div>
    </div>
  );
}
