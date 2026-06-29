import { useMemo } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";


const COLORS = {
  data: "#3b82f6",
  target: "#f59e0b",
  concept: "#10b981",
  threshold: "#ef4444",
};


export default function DriftCharts({ runs }) {
  const history = useMemo(() => (
    [...runs].reverse().slice(-20).map((run) => {
      const details = run.details;
      const timestamp = run.created_at
        ? new Date(run.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
        : run.id;
      return {
        timestamp,
        psi: details?.data_drift?.score != null ? +details.data_drift.score.toFixed(3) : null,
        target: details?.target_drift?.score != null ? +details.target_drift.score.toFixed(4) : null,
        concept: details?.concept_drift?.score != null ? +details.concept_drift.score.toFixed(4) : null,
      };
    })
  ), [runs]);

  const barData = useMemo(() => {
    const details = runs[0]?.details;
    return [
      {
        name: "Data Drift",
        score: details?.data_drift?.score != null ? +details.data_drift.score.toFixed(3) : 0,
        threshold: details?.data_drift?.threshold ?? 0.2,
        fill: COLORS.data,
      },
      {
        name: "Target Drift",
        score: details?.target_drift?.score != null ? +details.target_drift.score.toFixed(4) : 0,
        threshold: details?.target_drift?.threshold ?? 0.02,
        fill: COLORS.target,
      },
      {
        name: "Concept Drift",
        score: details?.concept_drift?.score != null ? +details.concept_drift.score.toFixed(4) : 0,
        threshold: details?.concept_drift?.threshold ?? 0.05,
        fill: COLORS.concept,
      },
    ];
  }, [runs]);

  if (!runs.length) return null;

  return (
    <div className="drift-charts">
      <div className="drift-chart-block">
        <h4>Drift Score History (last 20 runs)</h4>
        <ResponsiveContainer width="100%" height={200}>
          <AreaChart data={history} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="gPsi" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={COLORS.data} stopOpacity={0.3} />
                <stop offset="95%" stopColor={COLORS.data} stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey="timestamp" tick={{ fontSize: 11 }} interval="preserveStartEnd" />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip formatter={(value, name) => [value ?? "—", name]} />
            <Legend />
            <ReferenceLine
              y={0.2}
              stroke={COLORS.threshold}
              strokeDasharray="4 2"
              label={{ value: "PSI threshold", fontSize: 10, fill: COLORS.threshold }}
            />
            <Area
              type="monotone"
              dataKey="psi"
              name="Data PSI"
              stroke={COLORS.data}
              fill="url(#gPsi)"
              dot={false}
              connectNulls
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className="drift-chart-block">
        <h4>Current Score vs Threshold</h4>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={barData} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey="name" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip formatter={(value) => Number(value).toFixed(4)} />
            <Legend />
            <Bar dataKey="score" name="Score" radius={[4, 4, 0, 0]}>
              {barData.map((entry) => <Cell key={entry.name} fill={entry.fill} />)}
            </Bar>
            <Bar
              dataKey="threshold"
              name="Threshold"
              fill={COLORS.threshold}
              opacity={0.4}
              radius={[4, 4, 0, 0]}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
