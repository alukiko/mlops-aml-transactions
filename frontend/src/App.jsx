import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { Activity, AlertTriangle, BarChart3, FlaskConical, Play, RefreshCw, Search } from "lucide-react";
import "./styles.css";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

const sampleTransaction = {
  Timestamp: "2022/09/01 00:20",
  "From Bank": "010",
  "From Account": "8000EBD30",
  "To Bank": "010",
  "To Account": "8000EBD30",
  "Amount Received": 3697.34,
  "Receiving Currency": "US Dollar",
  "Amount Paid": 3697.34,
  "Payment Currency": "US Dollar",
  "Payment Format": "Reinvestment"
};

function api(path, options) {
  return fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options
  }).then(async (response) => {
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || response.statusText);
    return data;
  });
}

function StatusPill({ value }) {
  const normalized = String(value || "unknown").toLowerCase();
  return <span className={`pill ${normalized}`}>{value || "unknown"}</span>;
}

function Inference({ onRefresh }) {
  const [payload, setPayload] = useState(JSON.stringify(sampleTransaction, null, 2));
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  async function submit() {
    setError("");
    try {
      const parsed = JSON.parse(payload);
      const transactions = Array.isArray(parsed) ? parsed : [parsed];
      const data = await api("/predict", { method: "POST", body: JSON.stringify({ transactions }) });
      setResult(data.results);
      onRefresh();
    } catch (exc) {
      setError(exc.message);
    }
  }

  return (
    <section className="panel inference">
      <div className="panel-header">
        <h2>Inference</h2>
        <button onClick={submit}><Search size={16} /> Predict</button>
      </div>
      <textarea value={payload} onChange={(event) => setPayload(event.target.value)} spellCheck="false" />
      {error && <div className="notice danger">{error}</div>}
      {result && (
        <div className="result-grid">
          {result.map((item, index) => (
            <div className="metric" key={index}>
              <span>Probability</span>
              <strong>{(item.probability * 100).toFixed(2)}%</strong>
              <StatusPill value={item.anomaly_flag ? "anomaly" : "normal"} />
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function RecentPredictions({ items }) {
  return (
    <section className="panel">
      <div className="panel-header">
        <h2>Recent Predictions</h2>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr><th>ID</th><th>Time</th><th>Amount</th><th>Payment</th><th>Probability</th><th>Class</th><th>Flags</th></tr>
          </thead>
          <tbody>
            {items.map((item) => {
              const payload = item.payload || {};
              return (
                <tr key={item.id}>
                  <td>{item.id}</td>
                  <td>{item.created_at}</td>
                  <td>{payload["Amount Paid"] ?? "-"}</td>
                  <td>{payload["Payment Format"] ?? "-"}</td>
                  <td>{(item.probability * 100).toFixed(2)}%</td>
                  <td>{item.predicted_class}</td>
                  <td>{item.anomaly_flag ? <StatusPill value="anomaly" /> : <StatusPill value="normal" />}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function dataDriftStatus(dd) {
  if (!dd) return "unknown";
  return dd.score != null && dd.score > (dd.threshold ?? 0.2) ? "drift" : "ok";
}

function Drift({ runs, onRefresh }) {
  const [running, setRunning] = useState(false);
  async function run() {
    setRunning(true);
    try {
      await api("/drift/run", { method: "POST", body: JSON.stringify({ use_recent_predictions: false }) });
      onRefresh();
    } finally {
      setRunning(false);
    }
  }
  const latest = runs[0];
  const details = latest?.details;

  return (
    <section className="panel">
      <div className="panel-header">
        <h2>Drift Alerts</h2>
        <button onClick={run} disabled={running}><Activity size={16} /> {running ? "Running" : "Run drift"}</button>
      </div>

      {latest ? (
        <>
          <div className="drift-cards">
            <div className="drift-card">
              <span className="drift-card-label">Data Drift</span>
              <StatusPill value={dataDriftStatus(details?.data_drift)} />
              <span className="drift-card-score">PSI {details?.data_drift?.score?.toFixed(3) ?? "—"} / threshold {details?.data_drift?.threshold ?? 0.2}</span>
            </div>
            <div className="drift-card">
              <span className="drift-card-label">Target Drift</span>
              <StatusPill value={details?.target_drift?.status ?? "unknown"} />
              <span className="drift-card-score">score {details?.target_drift?.score?.toFixed(4) ?? "—"} / threshold {details?.target_drift?.threshold ?? 0.02}</span>
            </div>
            <div className="drift-card">
              <span className="drift-card-label">Concept Drift</span>
              <StatusPill value={details?.concept_drift?.status ?? "unknown"} />
              <span className="drift-card-score">F1-drop {details?.concept_drift?.score?.toFixed(4) ?? "—"} / threshold {details?.concept_drift?.threshold ?? 0.05}</span>
            </div>
          </div>

          <div className="alert-line">
            <AlertTriangle size={18} />
            <span>Combined</span>
            <StatusPill value={latest.status} />
            <span>score {Number(latest.score || 0).toFixed(4)}</span>
            <span>rows current {details?.rows?.current ?? "—"} / ref {details?.rows?.reference ?? "—"}</span>
            <span>{details?.source || "unknown source"}</span>
          </div>
        </>
      ) : <div className="empty">No drift reports yet</div>}

      <div className="table-wrap compact">
        <table>
          <thead>
            <tr>
              <th>ID</th><th>Created</th><th>Combined</th>
              <th>Data Drift</th><th>Target Drift</th><th>Concept Drift</th>
              <th>Rows</th><th>Report</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((run) => {
              const d = run.details;
              return (
                <tr key={run.id}>
                  <td>{run.id}</td>
                  <td>{run.created_at}</td>
                  <td><StatusPill value={run.status} /></td>
                  <td>
                    <StatusPill value={dataDriftStatus(d?.data_drift)} />
                    {" "}{d?.data_drift?.score?.toFixed(2) ?? "—"}
                  </td>
                  <td><StatusPill value={d?.target_drift?.status ?? "unknown"} /></td>
                  <td><StatusPill value={d?.concept_drift?.status ?? "unknown"} /></td>
                  <td>{d?.rows?.current ?? "—"}</td>
                  <td>{run.report_html || run.report_json || "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Experiments({ items }) {
  return (
    <section className="panel">
      <div className="panel-header"><h2>Experiments</h2><FlaskConical size={18} /></div>
      <div className="table-wrap compact">
        <table>
          <thead><tr><th>Run</th><th>Name</th><th>ROC-AUC</th><th>F1</th><th>Recall</th><th>Rows</th></tr></thead>
          <tbody>
            {items.map((run) => (
              <tr key={run.run_id}>
                <td>{run.run_id.slice(0, 8)}</td>
                <td>{run.run_name || "-"}</td>
                <td>{run.metrics.oof_roc_auc?.toFixed?.(4) ?? run.metrics.roc_auc?.toFixed?.(4) ?? "-"}</td>
                <td>{run.metrics.oof_f1?.toFixed?.(4) ?? run.metrics.f1?.toFixed?.(4) ?? "-"}</td>
                <td>{run.metrics.oof_recall?.toFixed?.(4) ?? run.metrics.recall?.toFixed?.(4) ?? "-"}</td>
                <td>{run.params.train_rows || "-"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Operations({ jobs, onRefresh }) {
  async function retrain() {
    await api("/retrain", { method: "POST" });
    onRefresh();
  }
  return (
    <section className="panel">
      <div className="panel-header">
        <h2>Operations</h2>
        <button onClick={retrain}><Play size={16} /> Start retraining</button>
      </div>
      <div className="table-wrap compact">
        <table>
          <thead><tr><th>ID</th><th>Status</th><th>Created</th><th>Finished</th><th>MLflow run</th><th>Error</th></tr></thead>
          <tbody>
            {jobs.map((job) => (
              <tr key={job.id}>
                <td>{job.id}</td>
                <td><StatusPill value={job.status} /></td>
                <td>{job.created_at}</td>
                <td>{job.finished_at || "-"}</td>
                <td>{job.mlflow_run_id || "-"}</td>
                <td>{job.error || "-"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function App() {
  const [predictions, setPredictions] = useState([]);
  const [driftRuns, setDriftRuns] = useState([]);
  const [experiments, setExperiments] = useState([]);
  const [jobs, setJobs] = useState([]);

  async function refresh() {
    const [p, d, e, j] = await Promise.all([
      api("/predictions/recent?limit=50"),
      api("/drift/status"),
      api("/experiments"),
      api("/retrain/jobs")
    ]);
    setPredictions(p.items || []);
    setDriftRuns(d.items || []);
    setExperiments(e.items || []);
    setJobs(j.items || []);
  }

  useEffect(() => { refresh(); }, []);

  const stats = useMemo(() => {
    const anomalies = predictions.filter((item) => item.anomaly_flag).length;
    return { total: predictions.length, anomalies, drift: driftRuns[0]?.status || "none", jobs: jobs[0]?.status || "none" };
  }, [predictions, driftRuns, jobs]);

  return (
    <main>
      <header>
        <div>
          <h1>AML Monitoring</h1>
          <p>Inference, drift reports, experiments and retraining control</p>
        </div>
        <button className="ghost" onClick={refresh}><RefreshCw size={16} /> Refresh</button>
      </header>
      <div className="stats">
        <div className="stat"><BarChart3 size={18} /><span>Predictions</span><strong>{stats.total}</strong></div>
        <div className="stat"><AlertTriangle size={18} /><span>Anomalies</span><strong>{stats.anomalies}</strong></div>
        <div className="stat"><Activity size={18} /><span>Drift</span><strong>{stats.drift}</strong></div>
        <div className="stat"><Play size={18} /><span>Retraining</span><strong>{stats.jobs}</strong></div>
      </div>
      <div className="grid">
        <Inference onRefresh={refresh} />
        <Drift runs={driftRuns} onRefresh={refresh} />
        <RecentPredictions items={predictions} />
        <Experiments items={experiments} />
        <Operations jobs={jobs} onRefresh={refresh} />
      </div>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
