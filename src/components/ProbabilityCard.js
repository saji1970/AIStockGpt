import React from 'react';

function ProbabilityCard({ forecast }) {
  if (!forecast) return null;

  const {
    probability_positive,
    median_value,
    var_95,
    best_case,
    worst_case,
    initial_amount,
    months,
  } = forecast;

  const medianReturn = initial_amount
    ? (((median_value - initial_amount) / initial_amount) * 100).toFixed(1)
    : 0;

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
      <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
        Forecast Probabilities
      </h3>
      <div className="grid grid-cols-2 gap-3">
        <div className="text-center p-2 bg-green-50 dark:bg-green-900/20 rounded-lg">
          <div className="text-2xl font-bold text-green-600">
            {(probability_positive * 100).toFixed(0)}%
          </div>
          <div className="text-xs text-gray-500 dark:text-gray-400">Chance of Profit</div>
        </div>
        <div className="text-center p-2 bg-indigo-50 dark:bg-indigo-900/20 rounded-lg">
          <div className="text-2xl font-bold text-indigo-600">
            ${median_value?.toLocaleString(undefined, { maximumFractionDigits: 0 })}
          </div>
          <div className="text-xs text-gray-500 dark:text-gray-400">
            Median ({medianReturn > 0 ? '+' : ''}{medianReturn}%)
          </div>
        </div>
        <div className="text-center p-2 bg-emerald-50 dark:bg-emerald-900/20 rounded-lg">
          <div className="text-lg font-semibold text-emerald-600">
            ${best_case?.toLocaleString(undefined, { maximumFractionDigits: 0 })}
          </div>
          <div className="text-xs text-gray-500 dark:text-gray-400">Best Case (95th)</div>
        </div>
        <div className="text-center p-2 bg-red-50 dark:bg-red-900/20 rounded-lg">
          <div className="text-lg font-semibold text-red-600">
            ${worst_case?.toLocaleString(undefined, { maximumFractionDigits: 0 })}
          </div>
          <div className="text-xs text-gray-500 dark:text-gray-400">Worst Case (5th)</div>
        </div>
      </div>
      {var_95 > 0 && (
        <div className="mt-2 text-xs text-gray-500 dark:text-gray-400 text-center">
          Value at Risk (95%): ${var_95?.toLocaleString(undefined, { maximumFractionDigits: 0 })}
        </div>
      )}
    </div>
  );
}

export default ProbabilityCard;
