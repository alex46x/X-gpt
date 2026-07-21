"""Self-contained browser chat interface."""

CHAT_UI = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="dark">
  <title>Project Genesis Chat</title>
  <style>
    :root {
      --background: #07090d;
      --sidebar: #0d1016;
      --surface: #171b23;
      --surface-hover: #202632;
      --border: #2d3440;
      --text: #f5f7fa;
      --muted: #9ba6b5;
      --accent: #78d6c6;
      --accent-strong: #a2efe2;
      --danger: #ff8d8d;
      --focus: #9ce8dc;
      --sidebar-width: 272px;
    }

    * { box-sizing: border-box; }

    html, body { height: 100%; }

    body {
      margin: 0;
      overflow: hidden;
      color: var(--text);
      background: var(--background);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
      font-size: 16px;
      line-height: 1.5;
    }

    button, textarea { font: inherit; }

    button {
      color: inherit;
      cursor: pointer;
    }

    button:focus-visible,
    textarea:focus-visible {
      outline: 3px solid var(--focus);
      outline-offset: 2px;
    }

    .app {
      display: grid;
      grid-template-columns: var(--sidebar-width) minmax(0, 1fr);
      height: 100dvh;
    }

    .sidebar {
      position: relative;
      z-index: 20;
      display: flex;
      min-height: 100dvh;
      flex-direction: column;
      gap: 24px;
      padding: 18px 14px;
      border-right: 1px solid var(--border);
      background: var(--sidebar);
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 11px;
      padding: 4px 8px;
      font-weight: 650;
      letter-spacing: -0.02em;
    }

    .brand-mark {
      display: grid;
      width: 34px;
      height: 34px;
      place-items: center;
      border: 1px solid #44625f;
      border-radius: 11px;
      color: var(--accent-strong);
      background: #12201f;
      font-family: ui-monospace, "Cascadia Code", monospace;
      font-size: 14px;
      box-shadow: inset 0 0 18px rgb(120 214 198 / 8%);
    }

    .new-chat {
      display: flex;
      min-height: 48px;
      align-items: center;
      gap: 10px;
      width: 100%;
      padding: 0 14px;
      border: 1px solid var(--border);
      border-radius: 13px;
      background: var(--surface);
      transition: background 180ms ease, border-color 180ms ease;
    }

    .new-chat:hover {
      border-color: #465162;
      background: var(--surface-hover);
    }

    .new-chat svg,
    .menu-button svg,
    .send-button svg {
      width: 20px;
      height: 20px;
      flex: none;
    }

    .model-card {
      margin-top: auto;
      padding: 14px;
      border: 1px solid var(--border);
      border-radius: 14px;
      color: var(--muted);
      background: #11151c;
      font-size: 13px;
    }

    .signal {
      display: flex;
      align-items: center;
      gap: 9px;
      margin-bottom: 8px;
      color: var(--text);
      font-weight: 600;
    }

    .signal-dot {
      width: 9px;
      height: 9px;
      border-radius: 999px;
      background: var(--muted);
      box-shadow: 0 0 0 4px rgb(155 166 181 / 10%);
    }

    .signal-dot.ready {
      background: var(--accent);
      box-shadow: 0 0 0 4px rgb(120 214 198 / 12%), 0 0 18px rgb(120 214 198 / 45%);
    }

    .fingerprint {
      display: block;
      overflow: hidden;
      margin-top: 5px;
      color: #788596;
      font-family: ui-monospace, "Cascadia Code", monospace;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .main {
      position: relative;
      display: grid;
      min-width: 0;
      min-height: 0;
      grid-template-rows: auto minmax(0, 1fr) auto;
    }

    .topbar {
      display: flex;
      min-height: 64px;
      align-items: center;
      gap: 12px;
      padding: 0 22px;
      border-bottom: 1px solid transparent;
    }

    .topbar-title {
      font-size: 15px;
      font-weight: 650;
    }

    .topbar-subtitle {
      color: var(--muted);
      font-size: 13px;
    }

    .menu-button {
      display: none;
      width: 44px;
      height: 44px;
      place-items: center;
      border: 0;
      border-radius: 11px;
      background: transparent;
    }

    .menu-button:hover { background: var(--surface); }

    .conversation {
      min-height: 0;
      overflow-y: auto;
      scroll-behavior: smooth;
      scrollbar-color: #343c48 transparent;
    }

    .conversation-inner {
      width: min(100%, 820px);
      min-height: 100%;
      margin: 0 auto;
      padding: 40px 24px 24px;
    }

    .welcome {
      display: grid;
      min-height: 62vh;
      grid-template-columns: minmax(0, 1fr);
      place-content: center;
      text-align: center;
    }

    .welcome[hidden] { display: none; }

    .welcome-kicker {
      margin: 0 0 10px;
      color: var(--accent);
      font-family: ui-monospace, "Cascadia Code", monospace;
      font-size: 12px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }

    .welcome h1 {
      max-width: 640px;
      margin: 0;
      font-size: clamp(30px, 5vw, 52px);
      font-weight: 560;
      letter-spacing: -0.045em;
      line-height: 1.08;
    }

    .welcome p:last-child {
      max-width: 520px;
      margin: 18px auto 0;
      color: var(--muted);
      font-size: 15px;
    }

    .message-list {
      display: grid;
      gap: 28px;
    }

    .message {
      display: grid;
      grid-template-columns: 34px minmax(0, 1fr);
      gap: 14px;
      align-items: start;
    }

    .message.user {
      grid-template-columns: minmax(0, 1fr);
      justify-items: end;
    }

    .avatar {
      display: grid;
      width: 34px;
      height: 34px;
      place-items: center;
      border: 1px solid var(--border);
      border-radius: 10px;
      color: var(--accent-strong);
      background: #12201f;
      font-family: ui-monospace, "Cascadia Code", monospace;
      font-size: 12px;
    }

    .message-text {
      max-width: 100%;
      margin: 3px 0 0;
      color: #e9edf2;
      overflow-wrap: anywhere;
      white-space: pre-wrap;
    }

    .user .message-text {
      max-width: min(78%, 620px);
      margin: 0;
      padding: 11px 15px;
      border: 1px solid #343b47;
      border-radius: 17px 17px 4px 17px;
      background: #1c212a;
    }

    .thinking {
      display: inline-flex;
      gap: 5px;
      padding-top: 10px;
    }

    .thinking span {
      width: 6px;
      height: 6px;
      border-radius: 999px;
      background: var(--accent);
      animation: pulse 1.1s ease-in-out infinite;
    }

    .thinking span:nth-child(2) { animation-delay: 120ms; }
    .thinking span:nth-child(3) { animation-delay: 240ms; }

    @keyframes pulse {
      0%, 70%, 100% { opacity: 0.25; transform: translateY(0); }
      35% { opacity: 1; transform: translateY(-3px); }
    }

    .error {
      color: var(--danger);
    }

    .composer-shell {
      padding: 12px 24px 24px;
      background: linear-gradient(180deg, transparent, var(--background) 28%);
    }

    .composer {
      display: grid;
      width: min(100%, 820px);
      margin: 0 auto;
      grid-template-columns: minmax(0, 1fr) 48px;
      gap: 10px;
      align-items: end;
      padding: 9px 9px 9px 18px;
      border: 1px solid #363e4b;
      border-radius: 22px;
      background: #181d25;
      box-shadow: 0 18px 60px rgb(0 0 0 / 24%);
    }

    .composer:focus-within {
      border-color: #5d706f;
      box-shadow: 0 0 0 3px rgb(120 214 198 / 8%), 0 18px 60px rgb(0 0 0 / 24%);
    }

    .composer textarea {
      width: 100%;
      max-height: 180px;
      resize: none;
      overflow-y: auto;
      padding: 10px 0;
      border: 0;
      outline: 0;
      color: var(--text);
      background: transparent;
      line-height: 1.5;
    }

    .composer textarea::placeholder { color: #7f8a99; }

    .send-button {
      display: grid;
      width: 48px;
      height: 48px;
      place-items: center;
      border: 0;
      border-radius: 15px;
      color: #06110f;
      background: var(--accent);
      transition: background 180ms ease, opacity 180ms ease;
    }

    .send-button:hover { background: var(--accent-strong); }

    .send-button:disabled {
      cursor: not-allowed;
      opacity: 0.38;
    }

    .composer-hint {
      width: min(100%, 820px);
      margin: 8px auto 0;
      color: #707b89;
      font-size: 12px;
      text-align: center;
    }

    .sr-only {
      position: absolute;
      width: 1px;
      height: 1px;
      padding: 0;
      margin: -1px;
      overflow: hidden;
      clip: rect(0, 0, 0, 0);
      white-space: nowrap;
      border: 0;
    }

    @media (max-width: 720px) {
      .app { grid-template-columns: 1fr; }

      .sidebar {
        position: fixed;
        inset: 0 auto 0 0;
        width: min(var(--sidebar-width), 86vw);
        transform: translateX(-102%);
        transition: transform 200ms ease;
        box-shadow: 24px 0 60px rgb(0 0 0 / 45%);
      }

      .sidebar.open { transform: translateX(0); }
      .menu-button { display: grid; }
      .topbar { padding: 0 12px; border-bottom-color: var(--border); }
      .topbar-subtitle { display: none; }
      .conversation-inner { padding: 26px 16px 18px; }
      .welcome { min-height: 56vh; }
      .composer-shell { padding: 10px 12px 14px; }
      .user .message-text { max-width: 90%; }
    }

    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after {
        scroll-behavior: auto !important;
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
      }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar" id="sidebar" aria-label="Chat navigation">
      <div class="brand">
        <span class="brand-mark" aria-hidden="true">G/</span>
        <span>Project Genesis</span>
      </div>
      <button class="new-chat" id="new-chat" type="button">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"
             aria-hidden="true">
          <path d="M12 5v14M5 12h14"/>
        </svg>
        New chat
      </button>
      <div class="model-card">
        <div class="signal">
          <span class="signal-dot" id="signal-dot" aria-hidden="true"></span>
          <span id="model-status">Checking model</span>
        </div>
        <span>Local inference · CPU</span>
        <span class="fingerprint" id="fingerprint">Bundle unavailable</span>
      </div>
    </aside>

    <main class="main">
      <header class="topbar">
        <button class="menu-button" id="menu-button" type="button"
                aria-label="Open navigation" aria-controls="sidebar" aria-expanded="false">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"
               aria-hidden="true">
            <path d="M4 7h16M4 12h16M4 17h16"/>
          </svg>
        </button>
        <div>
          <div class="topbar-title">Genesis Chat</div>
          <div class="topbar-subtitle">Your locally trained decoder</div>
        </div>
      </header>

      <section class="conversation" id="conversation" aria-label="Conversation">
        <div class="conversation-inner">
          <div class="welcome" id="welcome">
            <p class="welcome-kicker">Local model · experimental</p>
            <h1>What do you want to build?</h1>
            <p>This smoke model proves the pipeline works. Its answers will remain rough until
               it receives substantially more data and training.</p>
          </div>
          <div class="message-list" id="message-list" aria-live="polite"></div>
        </div>
      </section>

      <div class="composer-shell">
        <form class="composer" id="composer">
          <label class="sr-only" for="message-input">Message Project Genesis</label>
          <textarea id="message-input" rows="1" maxlength="32768"
                    placeholder="Message Project Genesis" required></textarea>
          <button class="send-button" id="send-button" type="submit"
                  aria-label="Send message" disabled>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
                 aria-hidden="true">
              <path d="M12 19V5M6 11l6-6 6 6"/>
            </svg>
          </button>
        </form>
        <p class="composer-hint">Enter to send · Shift+Enter for a new line</p>
      </div>
    </main>
  </div>

  <script>
    const messages = [];
    const sidebar = document.querySelector('#sidebar');
    const menuButton = document.querySelector('#menu-button');
    const newChatButton = document.querySelector('#new-chat');
    const conversation = document.querySelector('#conversation');
    const welcome = document.querySelector('#welcome');
    const messageList = document.querySelector('#message-list');
    const composer = document.querySelector('#composer');
    const input = document.querySelector('#message-input');
    const sendButton = document.querySelector('#send-button');
    const modelStatus = document.querySelector('#model-status');
    const signalDot = document.querySelector('#signal-dot');
    const fingerprint = document.querySelector('#fingerprint');
    let busy = false;

    function setBusy(value) {
      busy = value;
      composer.setAttribute('aria-busy', String(value));
      sendButton.disabled = value || !input.value.trim();
    }

    function resizeInput() {
      input.style.height = 'auto';
      input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
      sendButton.disabled = busy || !input.value.trim();
    }

    function addMessage(role, text, extraClass = '') {
      welcome.hidden = true;
      const article = document.createElement('article');
      article.className = `message ${role} ${extraClass}`.trim();
      if (extraClass === 'error') article.setAttribute('role', 'alert');
      if (role === 'assistant') {
        const avatar = document.createElement('span');
        avatar.className = 'avatar';
        avatar.textContent = 'G/';
        avatar.setAttribute('aria-hidden', 'true');
        article.append(avatar);
      }
      const content = document.createElement('p');
      content.className = 'message-text';
      content.textContent = text;
      article.append(content);
      messageList.append(article);
      conversation.scrollTop = conversation.scrollHeight;
      return article;
    }

    function addThinking() {
      welcome.hidden = true;
      const article = document.createElement('article');
      article.className = 'message assistant';
      article.innerHTML = '<span class="avatar" aria-hidden="true">G/</span>' +
        '<span class="thinking" aria-label="Generating response">' +
        '<span></span><span></span><span></span></span>';
      messageList.append(article);
      conversation.scrollTop = conversation.scrollHeight;
      return article;
    }

    async function sendMessage(text) {
      const historyLength = messages.length;
      messages.push({ role: 'user', content: text });
      addMessage('user', text);
      const thinking = addThinking();
      setBusy(true);
      try {
        const response = await fetch('/v1/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            messages: messages.slice(-1),
            max_new_tokens: 128,
            temperature: 1,
            top_k: 50
          })
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || `Request failed with status ${response.status}`);
        }
        messages.push({ role: 'assistant', content: payload.text });
        thinking.remove();
        addMessage('assistant', payload.text || '(The model returned an empty response.)');
      } catch (error) {
        messages.length = historyLength;
        thinking.remove();
        const detail = error instanceof Error ? error.message : 'The request failed.';
        addMessage('assistant', `${detail} Try a shorter message or start a new chat.`, 'error');
      } finally {
        setBusy(false);
        input.focus();
      }
    }

    composer.addEventListener('submit', (event) => {
      event.preventDefault();
      const text = input.value.trim();
      if (!text || busy) return;
      input.value = '';
      resizeInput();
      void sendMessage(text);
    });

    input.addEventListener('input', resizeInput);
    input.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        composer.requestSubmit();
      }
    });

    newChatButton.addEventListener('click', () => {
      messages.length = 0;
      messageList.replaceChildren();
      welcome.hidden = false;
      sidebar.classList.remove('open');
      menuButton.setAttribute('aria-expanded', 'false');
      input.focus();
    });

    menuButton.addEventListener('click', () => {
      const open = sidebar.classList.toggle('open');
      menuButton.setAttribute('aria-expanded', String(open));
    });

    fetch('/readyz')
      .then((response) => {
        if (!response.ok) throw new Error('Model unavailable');
        return response.json();
      })
      .then((status) => {
        modelStatus.textContent = 'Model ready';
        signalDot.classList.add('ready');
        fingerprint.textContent = status.bundle_fingerprint;
        fingerprint.title = status.bundle_fingerprint;
      })
      .catch(() => {
        modelStatus.textContent = 'Model unavailable';
      });

    resizeInput();
  </script>
</body>
</html>
"""
