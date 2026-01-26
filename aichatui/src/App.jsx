import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { UI_CONFIG } from "./config.js";

const STORAGE_KEY = "aichatui.settings.v1";

const defaultSettings = {
  theme: UI_CONFIG.theme,
};

function loadSettings() {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return defaultSettings;
  try {
    return { ...defaultSettings, ...JSON.parse(raw) };
  } catch (err) {
    console.warn("Failed to parse settings", err);
    return defaultSettings;
  }
}

function saveSettings(settings) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
}

export default function App() {
  const [settings, setSettings] = useState(loadSettings);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "This investment assistant uses AI and built-in knowledge to help draft portfolio ideas. This is not investment advice; it is only a guide. Please consult a finance professional, such as a Certified Financial Planner, for advice. [Read more about the motivation and implementation details.](https://claude.ai/public/artifacts/00b6f21a-baae-4018-ae1e-ea7cb8ff36f0)",
    },
  ]);
  const [isSending, setIsSending] = useState(false);

  useEffect(() => {
    document.documentElement.dataset.theme = settings.theme;
    saveSettings(settings);
  }, [settings]);

  const updateSettings = (next) => setSettings((prev) => ({ ...prev, ...next }));

  const handleSend = async (event) => {
    event.preventDefault();
    const message = input.trim();
    if (!message || isSending) return;

    setMessages((prev) => [...prev, { role: "user", content: message }]);
    setInput("");
    setIsSending(true);

    try {
      const resp = await fetch(`${UI_CONFIG.agentUrl.replace(/\/$/, "")}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message,
          provider: null,
          api_key: null,
          use_configured_key: true,
        }),
      });
      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(text || "Request failed");
      }
      const data = await resp.json();
      const reply = data.reply || "No response returned.";
      setMessages((prev) => [...prev, { role: "assistant", content: reply }]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Could not reach the agent: ${err.message}`,
        },
      ]);
    } finally {
      setIsSending(false);
    }
  };

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-dot" />
          <div>
            <p className="brand-kicker">InvestAssistant</p>
            <h1>Research Console</h1>
          </div>
        </div>

        <div className="panel subtle">
          <h2>Quick Prompts</h2>
          <button
            type="button"
            onClick={() =>
              setInput("Build a low cost, high return portfolio.")
            }
          >
            Build a low cost, high return portfolio
          </button>
          <button
            type="button"
            onClick={() =>
              setInput("Build a low risk, high return portfolio.")
            }
          >
            Build a low risk, high return portfolio
          </button>
        </div>
      </aside>

      <main className="chat">
        <header className="chat-header">
          <div>
            <h2>AI Investment assistant</h2>
          </div>
          <div className="status">Ready</div>
        </header>

        <section className="chat-body">
          {messages.map((msg, index) => (
            <div key={`${msg.role}-${index}`} className={`bubble ${msg.role}`}>
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                className="markdown"
                components={{
                  table: ({ children, ...props }) => (
                    <div className="table-wrap">
                      <table className="table" {...props}>
                        {children}
                      </table>
                    </div>
                  ),
                  thead: ({ children, ...props }) => (
                    <thead className="table-head" {...props}>
                      {children}
                    </thead>
                  ),
                  tbody: ({ children, ...props }) => (
                    <tbody className="table-body" {...props}>
                      {children}
                    </tbody>
                  ),
                  tr: ({ children, ...props }) => (
                    <tr className="table-row" {...props}>
                      {children}
                    </tr>
                  ),
                  th: ({ children, ...props }) => (
                    <th className="table-header" {...props}>
                      {children}
                    </th>
                  ),
                  td: ({ children, ...props }) => (
                    <td className="table-cell" {...props}>
                      {children}
                    </td>
                  ),
                }}
              >
                {msg.content}
              </ReactMarkdown>
            </div>
          ))}
        </section>

        <form className="composer" onSubmit={handleSend}>
          <textarea
            placeholder="Ask anything about funds, risk, or portfolios..."
            value={input}
            onChange={(event) => setInput(event.target.value)}
          />
          <button type="submit" disabled={isSending}>
            {isSending ? "Thinking..." : "Send"}
          </button>
        </form>
      </main>
    </div>
  );
}
