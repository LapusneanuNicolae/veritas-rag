"use client";

import { useEffect, useRef, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Citation = { document: string; location: string; snippet: string };
type Step = { step: string; detail: string };
type Message = {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  steps?: Step[];
  faithfulness?: number;
  cached?: boolean;
  streaming?: boolean;
};

const EXAMPLES = [
  "What is the refund window?",
  "Summarize the document",
  "What does it say about pricing?",
];

export default function Home() {
  const [docs, setDocs] = useState<string[]>([]);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [agentic, setAgentic] = useState(false);
  const [busy, setBusy] = useState(false);
  const [uploadMsg, setUploadMsg] = useState("");
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function upload(file: File) {
    setUploadMsg(`Indexing "${file.name}"…`);
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch(`${API}/api/documents/upload`, { method: "POST", body: form });
      const data = await res.json();
      if (res.ok) {
        setDocs((d) => Array.from(new Set([...d, data.filename])));
        setUploadMsg(`Indexed "${data.filename}" · ${data.chunks} chunks`);
      } else {
        setUploadMsg(`Error: ${data.detail}`);
      }
    } catch {
      setUploadMsg("Upload failed — is the backend running?");
    }
    setTimeout(() => setUploadMsg(""), 4000);
  }

  function push(msg: Message) {
    setMessages((m) => [...m, msg]);
  }
  function updateLast(patch: Partial<Message>) {
    setMessages((m) => {
      const copy = [...m];
      copy[copy.length - 1] = { ...copy[copy.length - 1], ...patch };
      return copy;
    });
  }

  async function ask(q?: string) {
    const text = (q ?? question).trim();
    if (!text || busy) return;
    setQuestion("");
    setBusy(true);
    push({ role: "user", content: text });

    if (agentic) {
      push({ role: "assistant", content: "", streaming: true });
      try {
        const res = await fetch(`${API}/api/ask/agentic`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: text }),
        });
        const data = await res.json();
        updateLast({
          content: data.answer || "",
          citations: data.citations,
          steps: data.steps,
          faithfulness: data.faithfulness,
          streaming: false,
        });
      } catch {
        updateLast({ content: "Request failed — is the backend running?", streaming: false });
      }
    } else {
      push({ role: "assistant", content: "", streaming: true });
      try {
        const res = await fetch(`${API}/api/ask/stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: text }),
        });
        const reader = res.body!.getReader();
        const decoder = new TextDecoder();
        let acc = "";
        let buffer = "";
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const parts = buffer.split("\n\n");
          buffer = parts.pop() || "";
          for (const part of parts) {
            const line = part.replace(/^data:\s?/, "");
            if (line === "[DONE]") continue;
            acc += line;
            updateLast({ content: acc });
          }
        }
        // fetch citations separately (stream endpoint returns text only)
        const meta = await fetch(`${API}/api/ask`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: text }),
        }).then((r) => r.json());
        updateLast({ content: acc || meta.answer, citations: meta.citations, cached: meta.cached, streaming: false });
      } catch {
        updateLast({ content: "Request failed — is the backend running?", streaming: false });
      }
    }
    setBusy(false);
  }

  function faithBadge(f: number) {
    const cls = f >= 0.8 ? "good" : f >= 0.5 ? "mid" : "low";
    return <span className={`badge ${cls}`}>faithfulness {f.toFixed(2)}</span>;
  }

  return (
    <div className="app">
      <header className="top">
        <div className="container top-inner">
          <div className="brand">
            <div className="logo">🔎</div>
            <div>
              <h1>Veritas RAG</h1>
              <p>Grounded answers with verifiable citations</p>
            </div>
          </div>
          <a className="gh" href="https://github.com/LapusneanuNicolae/veritas-rag" target="_blank" rel="noreferrer">
            GitHub ↗
          </a>
        </div>
      </header>

      <div className="container">
        <div className="docbar">
          <label className="upload-label">
            ＋ Upload document
            <input
              type="file"
              accept=".pdf,.docx,.md,.txt"
              onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])}
            />
          </label>
          {docs.map((d) => (
            <span className="chip" key={d}><span className="dot" />{d}</span>
          ))}
          {uploadMsg && <span className="hint">{uploadMsg}</span>}
          {docs.length === 0 && !uploadMsg && (
            <span className="hint">Upload a .pdf / .docx / .md / .txt to get started</span>
          )}
        </div>
      </div>

      <main className="container chat">
        {messages.length === 0 ? (
          <div className="empty">
            <div className="big">🔎</div>
            <h2>Ask anything about your documents</h2>
            <p>Hybrid retrieval · reranking · agentic mode · verifiable citations</p>
            <div className="examples">
              {EXAMPLES.map((ex) => (
                <span className="example" key={ex} onClick={() => ask(ex)}>{ex}</span>
              ))}
            </div>
          </div>
        ) : (
          <div className="msgs">
            {messages.map((m, i) => (
              <div className={`row ${m.role === "user" ? "user" : "ai"}`} key={i}>
                <div className={`avatar ${m.role === "user" ? "me" : "ai"}`}>
                  {m.role === "user" ? "🧑" : "🔎"}
                </div>
                <div style={{ maxWidth: "76%" }}>
                  <div className="bubble">
                    {m.streaming && !m.content ? (
                      <span className="typing"><span /><span /><span /></span>
                    ) : (
                      m.content
                    )}
                  </div>

                  {m.role === "assistant" && !m.streaming && (
                    <div className="meta">
                      {(m.cached || m.faithfulness !== undefined) && (
                        <div className="badges">
                          {m.cached && <span className="badge cached">cached</span>}
                          {m.faithfulness !== undefined && faithBadge(m.faithfulness)}
                        </div>
                      )}

                      {m.steps && m.steps.length > 0 && (
                        <div>
                          <div className="steps-title">Agent steps</div>
                          <div className="steps">
                            {m.steps.map((s, j) => (
                              <div className="step" key={j}><b>{s.step}</b> — {s.detail}</div>
                            ))}
                          </div>
                        </div>
                      )}

                      {m.citations && m.citations.length > 0 && (
                        <div>
                          <div className="sources-title">Sources</div>
                          {m.citations.map((c, j) => (
                            <div className="source" key={j} style={{ marginTop: 6 }}>
                              <span className="num">{j + 1}</span>
                              <span><b>{c.document}</b> · {c.location}<br />{c.snippet}</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            ))}
            <div ref={endRef} />
          </div>
        )}
      </main>

      <div className="composer">
        <div className="container">
          <div className="composer-box">
            <input
              type="text"
              placeholder="Ask a question about your documents…"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && ask()}
            />
            <button className="send" onClick={() => ask()} disabled={busy || !question.trim()}>
              {busy ? "…" : "Ask"}
            </button>
          </div>
          <div className="composer-controls">
            <div className="toggle" onClick={() => setAgentic((a) => !a)}>
              <span className={`switch ${agentic ? "on" : ""}`} />
              Agentic mode — decompose &amp; self-correct
            </div>
            <span className="hint">Enter to send</span>
          </div>
        </div>
      </div>
    </div>
  );
}
