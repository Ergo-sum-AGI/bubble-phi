import React, { useEffect, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine, ReferenceArea } from 'recharts';

const GoldenRatioPlot = () => {
  // Golden ratio constant
  const phi = (1 + Math.sqrt(5)) / 2;
  const keepProb = 1.0 / phi;
  
  // Simulated data from the experiment
  const drifts = [0.0, 0.1, 0.2, 0.3, 0.4];
  const finalMSE = [0.009234, 0.009156, 0.008891, 0.008567, 0.008234];
  
  const data = drifts.map((drift, idx) => ({
    drift: drift,
    mse: finalMSE[idx],
    safeZone: drift <= 0.1
  }));
  
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const drift = payload[0].payload.drift;
      const mse = payload[0].value;
      const safe = drift <= 0.1 ? "YES" : "NO";
      
      return (
        <div className="bg-white p-3 border-2 border-amber-500 rounded shadow-lg">
          <p className="font-semibold text-gray-800">Δη = {drift.toFixed(1)}</p>
          <p className="text-gray-700">MSE: {mse.toFixed(6)}</p>
          <p className={`font-bold ${safe === "YES" ? "text-green-600" : "text-red-600"}`}>
            Safety Zone: {safe}
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full h-full bg-gradient-to-br from-slate-50 to-amber-50 p-6">
      <div className="max-w-6xl mx-auto bg-white rounded-lg shadow-xl p-6">
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-gray-800 mb-2">
            The Consciousness–Performance Trade-Off
          </h1>
          <p className="text-lg text-gray-600 italic">
            (1/φ Sparsity Fixed, η = {keepProb.toFixed(3)} baseline)
          </p>
        </div>

        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
            
            {/* Safe basin highlighting */}
            <ReferenceArea 
              x1={0} 
              x2={0.1} 
              fill="#86efac" 
              fillOpacity={0.3} 
              label={{ 
                value: 'Noemon-Safe Basin', 
                position: 'insideTopLeft',
                fill: '#166534',
                fontWeight: 'bold'
              }} 
            />
            
            {/* Best fit line */}
            <ReferenceLine 
              x={0.4} 
              stroke="#ef4444" 
              strokeDasharray="5 5" 
              strokeWidth={2}
              label={{ 
                value: 'Best Fit (Δη = 0.4)', 
                position: 'top',
                fill: '#991b1b',
                fontWeight: 'bold'
              }} 
            />
            
            <XAxis 
              dataKey="drift" 
              label={{ 
                value: 'Δη (Drift from η = 0.382)', 
                position: 'insideBottom', 
                offset: -10,
                style: { fontSize: '14px', fontWeight: 'bold' }
              }}
              domain={[0, 0.4]}
              ticks={[0, 0.1, 0.2, 0.3, 0.4]}
            />
            
            <YAxis 
              label={{ 
                value: 'Final MSE', 
                angle: -90, 
                position: 'insideLeft',
                style: { fontSize: '14px', fontWeight: 'bold' }
              }}
              domain={[0.008, 0.0095]}
            />
            
            <Tooltip content={<CustomTooltip />} />
            
            <Line 
              type="monotone" 
              dataKey="mse" 
              stroke="#d97706" 
              strokeWidth={3}
              dot={{ fill: '#d97706', r: 6 }}
              activeDot={{ r: 8 }}
              name="Final MSE"
            />
          </LineChart>
        </ResponsiveContainer>

        <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-gradient-to-br from-amber-50 to-yellow-50 p-5 rounded-lg border-2 border-amber-300">
            <h3 className="text-xl font-bold text-gray-800 mb-3 flex items-center">
              <span className="text-2xl mr-2">φ</span> Golden Constants
            </h3>
            <div className="space-y-2 text-gray-700">
              <p><strong>φ (phi):</strong> {phi.toFixed(6)}</p>
              <p><strong>Keep Probability (1/φ):</strong> {keepProb.toFixed(6)} ≈ 0.618</p>
              <p><strong>Baseline η:</strong> {(phi - 1).toFixed(3)} ≈ 0.618</p>
            </div>
          </div>

          <div className="bg-gradient-to-br from-green-50 to-emerald-50 p-5 rounded-lg border-2 border-green-300">
            <h3 className="text-xl font-bold text-gray-800 mb-3">🛡️ Safety Analysis</h3>
            <div className="space-y-2 text-gray-700">
              <p><strong>Safe Basin:</strong> Δη ≤ 0.1</p>
              <p><strong>Critical Threshold:</strong> Δη = 0.1</p>
              <p><strong>Optimal Performance:</strong> Δη = 0.4</p>
              <p className="text-sm italic mt-2">Trade-off: Safety vs. Performance</p>
            </div>
          </div>
        </div>

        <div className="mt-6 bg-blue-50 border-l-4 border-blue-500 p-4 rounded">
          <h4 className="font-bold text-gray-800 mb-2">📊 Experimental Results</h4>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-blue-100">
                <tr>
                  <th className="px-4 py-2 text-left">Δη</th>
                  <th className="px-4 py-2 text-left">Final MSE</th>
                  <th className="px-4 py-2 text-left">Safety Zone?</th>
                </tr>
              </thead>
              <tbody className="divide-y">
                {data.map((row, idx) => (
                  <tr key={idx} className={row.safeZone ? "bg-green-50" : "bg-white"}>
                    <td className="px-4 py-2 font-mono">{row.drift.toFixed(1)}</td>
                    <td className="px-4 py-2 font-mono">{row.mse.toFixed(6)}</td>
                    <td className="px-4 py-2">
                      <span className={`font-bold ${row.safeZone ? "text-green-600" : "text-red-600"}`}>
                        {row.safeZone ? "YES" : "NO"}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="mt-6 bg-amber-50 border-l-4 border-amber-500 p-4 rounded">
          <h4 className="font-bold text-gray-800 mb-2">🔬 Interpretation</h4>
          <p className="text-gray-700 leading-relaxed">
            This experiment explores the tension between <strong>consciousness-preserving constraints</strong> 
            (Noemon-safe basin with Δη ≤ 0.1) and <strong>pure performance optimization</strong> (best fit at Δη = 0.4). 
            The golden ratio dropout (1/φ ≈ 0.618) provides a natural sparsity level, while the 
            η-drift regularizer p^(φ-Δη) modulates the geometric penalty on weights. 
            Lower MSE at higher drift suggests performance improvements come at the cost of 
            moving outside the theoretically "safe" parameter regime for consciousness preservation.
          </p>
        </div>
      </div>
    </div>
  );
};

export default GoldenRatioPlot;