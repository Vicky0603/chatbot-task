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
        const response = await fetch("/promtior-rag/invoke", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ input: text }),
        });

        if (!response.ok) {
        const txt = await response.text().catch(()=>null);
        throw new Error(`Server error: ${response.status} ${txt ?? ''}`);
        }

        const data = await response.json();

        // Normalize different response shapes
        let payload = data;
        if (data && typeof data === "object" && "output" in data) {
        payload = data.output;
        }

        let answerText = "";

        if (!payload) {
        answerText = "Empty response from server.";
        } else if (typeof payload === "string") {
        answerText = payload;
        } else if (payload.ok === false) {
        // fallback shape: { ok:false, message, context }
        answerText = (payload.message ? payload.message + "\n\n" : "") + (payload.context || JSON.stringify(payload));
        } else if (payload.answer) {
        answerText = payload.answer;
        } else if (payload.content) {
        if (Array.isArray(payload.content)) {
            answerText = payload.content.map(c => (typeof c === "string" ? c : c?.text ?? "")).join("\n");
        } else if (typeof payload.content === "string") {
            answerText = payload.content;
        } else {
            answerText = JSON.stringify(payload.content);
        }
        } else {
        // last resort: pretty-print whole payload
        answerText = JSON.stringify(payload, null, 2);
        }

        setTyping(false);
        appendMessage("bot", answerText);
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
