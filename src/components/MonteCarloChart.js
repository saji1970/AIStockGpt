import React from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

function MonteCarloChart({ forecast }) {
  if (!forecast || !forecast.percentiles) return null;

  const { percentiles, months, initial_amount } = forecast;

  // Build fan chart data points (simplified: start -> end)
  const steps = Math.min(months, 12);
  const data = [];

  for (let i = 0; i <= steps; i++) {
    const fraction = i / steps;
    const lerp = (start, end) => start + (end - start) * fraction;

    data.push({
      month: i,
      p5: Math.round(lerp(initial_amount, percentiles[5])),
      p25: Math.round(lerp(initial_amount, percentiles[25])),
      p50: Math.round(lerp(initial_amount, percentiles[50])),
      p75: Math.round(lerp(initial_amount, percentiles[75])),
      p95: Math.round(lerp(initial_amount, percentiles[95])),
    });
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow-sm border border-gray-200 dark:border-gray-700">
      <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
        Monte Carlo Forecast ({months}mo, 10K simulations)
      </h3>
      <ResponsiveContainer width="100%" height={220}>
        <AreaChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis dataKey="month" label={{ value: 'Month', position: 'bottom', offset: -5 }} />
          <YAxis tickFormatter={(v) => `$${(v / 1000).toFixed(1)}k`} />
          <Tooltip
            formatter={(value) => `$${value.toLocaleString()}`}
            labelFormatter={(label) => `Month ${label}`}
          />
          <Area type="monotone" dataKey="p95" stackId="1" stroke="none" fill="#c4b5fd" fillOpacity={0.3} name="95th %ile" />
          <Area type="monotone" dataKey="p75" stackId="2" stroke="none" fill="#a78bfa" fillOpacity={0.3} name="75th %ile" />
          <Area type="monotone" dataKey="p50" stackId="3" stroke="#6366f1" fill="#6366f1" fillOpacity={0.2} name="Median" strokeWidth={2} />
          <Area type="monotone" dataKey="p25" stackId="4" stroke="none" fill="#a78bfa" fillOpacity={0.3} name="25th %ile" />
          <Area type="monotone" dataKey="p5" stackId="5" stroke="none" fill="#c4b5fd" fillOpacity={0.3} name="5th %ile" />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

export default MonteCarloChart;
