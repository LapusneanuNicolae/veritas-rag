"use client";

import { useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Citation = { document: string; location: string; snippet: string };
type Step = { step: string; detail: string };

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [citations, setCitations] = useState<Citation[]>([]);
  const [steps, setSteps] = useState<Step[]>([]);
  const [agentic, setAgentic] = useState(false);
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(false);

  async function upload() {
    if (!file) return;
    setStatus("Uploading & indexing…");
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(`${API}/api/documents/upload`, { method: "POST", body: form });
    const data = await res.json();
    setStatus(res.ok ? `Indexed "${data.filename}" (${data.chunks} chunks)` : `Error: ${data.detail}`);
  }

  async function ask() {
    if (!question.trim()) return;
    setLoading(true);
    setAnswer("");
    setCitations([]);
    setSteps([]);
    const endpoint = agentic ? "/api/ask/agentic" : "/api/ask";
    const res = await fetch(`${API}${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    setAnswer(data.answer || "");
    setCitations(data.citations || []);
    if (data.steps) setSteps(data.steps);
    setLoading(false);
  }

  return (
    <div className="wrap">
      <h1>🔎 Veritas RAG</h1>
      <p className="sub">Ask questions over your documents — with verifiable citations.</p>

      <div className="panel">
        <div className="row">
          <input type="file" onChange={(e) => setFile(e.target.files?.[0] || null)} />
          <button onClick={upload} disabled={!file}>Upload</button>
        </div>
        {status && <div className="status">{status}</div>}
      </div>

      <div className="panel">
        <div className="row">
          <input
            type="text"
            placeholder="Ask a question about your documents…"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && ask()}
          />
          <button onClick={ask} disabled={loading}>{loading ? "…" : "Ask"}</button>
        </div>
        <label className="toggle" style={{ marginTop: 10 }}>
          <input type="checkbox" checked={agentic} onChange={(e) => setAgentic(e.target.checked)} />
          Agentic mode (decompose + self-correct)
        </label>
      </div>

      {answer && (
        <div className="panel">
          <div className="tag">ANSWER</div>
          <div className="answer">{answer}</div>

          {steps.length > 0 && (
            <>
              <div className="tag" style={{ marginTop: 14 }}>AGENT STEPS</div>
              <ul className="steps">
                {steps.map((s, i) => (
                  <li key={i}><b>{s.step}</b>: {s.detail}</li>
                ))}
              </ul>
            </>
          )}

          {citations.length > 0 && (
            <>
              <div className="tag" style={{ marginTop: 14 }}>SOURCES</div>
              {citations.map((c, i) => (
                <div className="cite" key={i}>
                  [{i + 1}] {c.document} @ {c.location} — {c.snippet}
                </div>
              ))}
            </>
          )}
        </div>
      )}
    </div>
  );
}
