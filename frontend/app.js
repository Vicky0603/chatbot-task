const messagesEl = document.getElementById("messages");
const formEl = document.getElementById("chat-form");
const inputEl = document.getElementById("message-input");
const errorEl = document.getElementById("error-message");

let isSending = false;

function appendMessage(role, text) {
    const row = document.createElement("div");
    row.classList.add("message-row", role);

    if (role === "system") {
        const bubble = document.createElement("div");
        bubble.classList.add("message-bubble");
        bubble.textContent = text;
        row.appendChild(bubble);
        messagesEl.appendChild(row);
        messagesEl.scrollTop = messagesEl.scrollHeight;
        return;
    }

    const avatar = document.createElement("div");
    avatar.classList.add("avatar", role);
    avatar.textContent = role === "user" ? "You" : "AI";

    const bubble = document.createElement("div");
    bubble.classList.add("message-bubble");
    bubble.textContent = text;

    if (role === "user") {
        row.appendChild(bubble);
    row.appendChild(avatar);
    } else {
        row.appendChild(avatar);
        row.appendChild(bubble);
    }
    messagesEl.appendChild(row);
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

function setTyping(isTyping) {
    const existing = document.querySelector(".message-row.system.typing");
    if (isTyping) {
        if (!existing) {
        const row = document.createElement("div");
        row.classList.add("message-row", "system", "typing");

        const bubble = document.createElement("div");
        bubble.classList.add("message-bubble");
        bubble.textContent = "Assistant is typing...";

        row.appendChild(bubble);
        messagesEl.appendChild(row);
        }
    } else if (existing) {
        existing.remove();
    }
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

function renderErrorFromPayload(payload, requestId, where = "") {
  const parts = [];
  const err = payload && payload.error ? payload.error : null;
  if (err) {
    parts.push(`${err.type || 'error'} (${err.status || 'n/a'}): ${err.message || 'Unknown error'}`);
    if (err.detail) {
      if (Array.isArray(err.detail)) {
        parts.push(err.detail.map(d => (d.msg || JSON.stringify(d))).join("; "));
      } else if (typeof err.detail === 'string') {
        parts.push(err.detail);
      } else {
        parts.push(JSON.stringify(err.detail));
      }
    }
    if (err.request_id || requestId) {
      parts.push(`request_id=${err.request_id || requestId}`);
    }
  } else {
    parts.push("Unexpected error format");
  }
  const msg = where ? `${where}: ${parts.join(" | ")}` : parts.join(" | ");
  errorEl.textContent = msg;
}

async function sendMessage(text) {
    if (!text.trim()) return;
    if (isSending) return;

    isSending = true;
    errorEl.textContent = "";
    inputEl.value = "";
    inputEl.disabled = true;
    formEl.querySelector("button[type='submit']").disabled = true;

    appendMessage("user", text);
    setTyping(true);

    try {
        // Try streaming partial tokens first
        const controller = new AbortController();
        const signal = controller.signal;
        const res = await fetch("/promtior-rag/stream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ input: text }),
            signal,
        });
        if (!res.ok || !res.body) {
            // Try to extract structured error
            let payload = null;
            let rid = res.headers.get('X-Request-ID') || undefined;
            try { payload = await res.json(); } catch(_) {}
            if (payload) {
              renderErrorFromPayload(payload, rid, 'stream');
            }
            throw new Error(`Stream error: ${res.status}`);
        }

        // Prepare assistant bubble to update progressively
        setTyping(false);
        appendMessage("bot", "");
        const assistantRow = messagesEl.lastChild;
        const bubble = assistantRow.querySelector(".message-bubble");

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            // SSE-style: split by newlines and parse 'data: ' lines when possible
            const parts = buffer.split(/\n\n/);
            buffer = parts.pop() || "";
            for (const chunk of parts) {
                const line = chunk.split("\n").find(l => l.startsWith("data:"));
                if (!line) continue;
                const json = line.replace(/^data:\s*/, "");
                try {
                    const evt = JSON.parse(json);
                    const token = typeof evt === "string" ? evt : (evt?.data ?? evt?.token ?? evt?.content ?? "");
                    if (token) bubble.textContent += token;
                } catch (_) {
                    // Fallback: append raw text
                    bubble.textContent += json;
                }
            }
            messagesEl.scrollTop = messagesEl.scrollHeight;
        }

        // After stream ends, fetch final structured answer (to get citations)
        const finalRes = await fetch("/promtior-rag/invoke", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ input: text }),
        });
        if (!finalRes.ok) {
            let rid = finalRes.headers.get('X-Request-ID') || undefined;
            try {
              const payload = await finalRes.json();
              renderErrorFromPayload(payload, rid, 'invoke');
            } catch(_) {
              const txt = await finalRes.text().catch(() => null);
              errorEl.textContent = `Invoke error: ${finalRes.status} ${txt ?? ''}`;
            }
            throw new Error(`Invoke error: ${finalRes.status}`);
        }
        const finalData = await finalRes.json();
        const payload = finalData?.output ?? finalData;
        if (payload && payload.answer) {
            bubble.textContent = payload.answer;
            // Render citations if available
            if (Array.isArray(payload.sources) && payload.sources.length) {
                const cites = document.createElement("div");
                cites.style.marginTop = "6px";
                cites.style.fontSize = "0.8rem";
                cites.style.opacity = "0.85";
                cites.innerHTML = "Sources:";
                const ul = document.createElement("ul");
                for (const s of payload.sources.slice(0, 3)) {
                    const li = document.createElement("li");
                    li.textContent = `${s.source}`;
                    ul.appendChild(li);
                }
                cites.appendChild(ul);
                bubble.appendChild(document.createElement("br"));
                bubble.appendChild(cites);
            }
        }
    } catch (err) {
        console.error(err);
        setTyping(false);
        errorEl.textContent =
        "There was a problem talking to the assistant. Check the console or configuration.";
    } finally {
        isSending = false;
        inputEl.disabled = false;
        formEl.querySelector("button[type='submit']").disabled = false;
        inputEl.focus();
    }
}

formEl.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = inputEl.value;
    sendMessage(text);
    });

    inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        formEl.dispatchEvent(new Event("submit"));
    }
});

// Initial message
appendMessage(
    "bot",
    "Hi, I'm the Promtior assistant. You can ask me about services, when the company was founded, or other information based on its content."
);
