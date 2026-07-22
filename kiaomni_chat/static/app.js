// KiaOmni Chat — production stress test UI
// Single ES module, no framework, no external CDN.

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

// Fallback article used only if /static/sample_data/article.txt fails to load.
// Shorter, just enough to test the side-by-side path.
const ARTICLE_FALLBACK = `The Transformer architecture was introduced in the 2017 paper
"Attention Is All You Need" by eight researchers at Google Brain: Ashish
Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan
Gomez, Łukasz Kaiser, and Illia Polosukhin. The paper replaced recurrence
with pure attention and became the foundation of modern LLMs. GPT-3,
released by OpenAI in 2020 with 175 billion parameters, was the watershed
moment. The open-source revolution began with Meta's LLaMA in February
2023 and continued with Mistral 7B in September 2023 and Alibaba's Qwen
family. The latest generation — Llama 3.1, Qwen2.5 — supports 128 K
contexts. As context windows grew, the cost of the KV cache grew
quadratically, motivating prompt-side eviction strategies like KiaOmni.`;

// The 9 test questions for the SpaceX Starship 2024–2025 article.
// Tier 1 = NIAH (single + multi-needle), Tier 2 = summarization, Tier 3 = reasoning.
const TEST_QUESTIONS = [
  { id: "q1", tier: 1, type: "NIAH",
    q: "On what date did the third integrated flight test (IFT-3) of Starship launch?" },
  { id: "q2", tier: 1, type: "NIAH",
    q: "What altitude did the Starship upper stage reach during IFT-3?" },
  { id: "q3", tier: 1, type: "NIAH",
    q: "What peak temperature did the heatshield experience during IFT-4's re-entry?" },
  { id: "q4", tier: 1, type: "NIAH",
    q: "On what date was the first successful booster catch achieved?" },
  { id: "q5", tier: 1, type: "NIAH multi-needle",
    q: "List the dates and outcomes of the five integrated flight tests (IFT-3 through IFT-7). Five facts: date + outcome for each." },
  { id: "q6", tier: 1, type: "NIAH multi-needle",
    q: "List the specifications of the Starship vehicle. Five facts needed: total height, total liftoff mass, total liftoff thrust, number of first-stage engines, number of upper-stage engines." },
  { id: "q7", tier: 3, type: "Multi-hop reasoning",
    q: "Why is the chopstick-catch approach more ambitious than leg-landing, and what engineering capabilities had to be proven before attempting the catch?" },
  { id: "q8", tier: 3, type: "Multi-hop reasoning",
    q: "Why did IFT-6 catch the booster successfully but lose the ship during the same mission, and what does the pattern of ship failures tell us about the engineering maturity of the two stages?" },
  { id: "q9", tier: 2, type: "Summarization",
    q: "Summarize the SpaceX Starship 2024–2025 test campaign in exactly 5 bullet points, covering: the vehicle, the early test flights, the catch breakthrough, the January 2025 milestone, and the broader industry impact." },
];

// ── VRAM formatting ────────────────────────────────────────────────────
// Two numbers come back from the engine:
//   vram_max_allocated_mb — true peak of `torch.cuda.memory_allocated()`
//                           since the last reset. THIS is the real
//                           "VRAM used by this call" number.
//   vram_reserved_mb      — size of the CUDA caching-allocator pool.
//                           Only grows, never shrinks (we don't call
//                           `empty_cache()` by design). For kiaomni, the
//                           saliency forward pass briefly pushes this up
//                           and the allocator holds the freed blocks, so
//                           this stays high across calls.
// We display the peak as the primary number and the pool as a secondary
// tag so the operator can see BOTH: "did this call OOM-risk" and "is the
// allocator pool growing toward a restart threshold".
function formatVram(stats) {
  if (!stats) return null;
  const peak = stats.vram_max_allocated_mb;
  const pool = stats.vram_reserved_mb;
  if (peak == null && pool == null) return null;
  const pStr = peak != null ? `${(peak / 1024).toFixed(2)} GB` : "?";
  if (pool == null) return pStr;
  // Show the pool only if it differs from the peak by > 64 MB.
  // Small deltas (≤64 MB) are just normal allocator rounding, not interesting.
  if (peak == null || Math.abs(peak - pool) < 64) return pStr;
  return `${pStr} peak / ${(pool / 1024).toFixed(2)} GB pool`;
}

// ── Chat save / copy / download ───────────────────────────────────────
function chatRecord(includeStats) {
  return {
    schema_version: 1,
    created_at: new Date().toISOString(),
    deploy_url: window.location.origin,
    model: "Qwen/Qwen2.5-7B-Instruct",
    policy: state.policy,
    budget: state.budget,
    max_new_tokens: state.maxNew,
    session_id: state.sessionId,
    message_count: state.messages.length,
    messages: state.messages.map((m) => {
      const out = { role: m.role, content: m.content, ts: m.ts || null };
      if (includeStats && m.stats) out.stats = m.stats;
      return out;
    }),
  };
}

function downloadBlob(content, filename, mime) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { document.body.removeChild(a); URL.revokeObjectURL(url); }, 0);
}

function timestampedFilename(prefix, ext) {
  const d = new Date();
  const z = (n) => String(n).padStart(2, "0");
  return `${prefix}_${d.getFullYear()}${z(d.getMonth()+1)}${z(d.getDate())}_${z(d.getHours())}${z(d.getMinutes())}${z(d.getSeconds())}.${ext}`;
}

function saveChatAsJson() {
  if (!state.messages.length) { alert("No messages to save yet."); return; }
  const rec = chatRecord(true);
  downloadBlob(JSON.stringify(rec, null, 2), timestampedFilename("kiaomni_chat", "json"), "application/json");
}

function chatAsText() {
  const rec = chatRecord(true);
  const lines = [];
  lines.push(`KiaOmni Chat Export`);
  lines.push(`Created:    ${rec.created_at}`);
  lines.push(`Model:      ${rec.model}`);
  lines.push(`Policy:     ${rec.policy}`);
  lines.push(`Budget:     ${rec.budget}`);
  lines.push(`Max new:    ${rec.max_new_tokens}`);
  lines.push(`Session:    ${rec.session_id || "(none)"}`);
  lines.push(`Messages:   ${rec.message_count}`);
  lines.push("=".repeat(72));
  rec.messages.forEach((m, i) => {
    const stamp = m.ts ? new Date(m.ts).toISOString() : "#" + (i + 1);
    if (m.role === "user") {
      lines.push("");
      lines.push(`[${stamp}] USER:`);
      lines.push(m.content);
    } else if (m.role === "assistant") {
      const s = m.stats || {};
      const meta = [
        `policy=${rec.policy}`,
        `B=${rec.budget}`,
        s.tokens_in != null ? `${s.tokens_in}→${s.tokens_kept} tok (${s.compression_pct ?? "?"}%)` : null,
        s.prefill_ms != null ? `prefill ${(s.prefill_ms/1000).toFixed(2)}s` : null,
        s.decode_ms != null ? `decode ${(s.decode_ms/1000).toFixed(2)}s` : null,
        s.tok_per_sec != null ? `${s.tok_per_sec.toFixed(1)} tok/s` : null,
        s.vram_max_mb != null ? `${(s.vram_max_mb/1024).toFixed(2)} GB peak${s.vram_reserved_mb && Math.abs(s.vram_max_mb - s.vram_reserved_mb) >= 64 ? ` / ${(s.vram_reserved_mb/1024).toFixed(2)} GB pool` : ""}` : null,
      ].filter(Boolean).join(" · ");
      lines.push("");
      lines.push(`[${stamp}] ASSISTANT (${meta}):`);
      lines.push(m.content);
    } else if (m.role === "system") {
      lines.push("");
      lines.push(`[${stamp}] SYSTEM:`);
      lines.push(m.content);
    }
  });
  return lines.join("\n");
}

function copyChatAsText() {
  if (!state.messages.length) { alert("No messages to copy yet."); return; }
  const text = chatAsText();
  navigator.clipboard.writeText(text).then(
    () => { flashSaveStatus("Copied!"); },
    () => { alert("Clipboard write failed — try Download instead."); }
  );
}

function downloadChatAsText() {
  if (!state.messages.length) { alert("No messages to download yet."); return; }
  downloadBlob(chatAsText(), timestampedFilename("kiaomni_chat", "txt"), "text/plain");
}

function flashSaveStatus(msg) {
  const el = $("#save-status");
  if (!el) return;
  const prev = el.textContent;
  el.textContent = msg;
  el.style.color = "var(--pass)";
  setTimeout(() => { el.textContent = prev; el.style.color = ""; }, 1500);
}

// ── State ────────────────────────────────────────────────────────────────
const state = {
  tab: "chat",
  policy: "kiaomni_gaussian",
  budget: 512,
  maxNew: 2048,
  sessionId: null,
  messages: [],  // {role, content, meta?}
  abortCtrl: null,
  oomCount: 0,
  // Populated from /api/health on first paint; used for the budget
  // slider range and the % label.
  modelContextWindow: 131072,  // Qwen2.5-7B default until health responds
  defaultBudget: 512,
};

// ── Tabs ────────────────────────────────────────────────────────────────
$$(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    const tab = btn.dataset.tab;
    state.tab = tab;
    $$(".tab").forEach((b) => b.setAttribute("aria-selected", String(b === btn)));
    renderPanel();
  });
});

// ── Settings sync ──────────────────────────────────────────────────────
$("#policy").addEventListener("change", (e) => {
  state.policy = e.target.value;
  $("#budget").disabled = state.policy === "fullcontext";
});
$("#budget").addEventListener("input", (e) => {
  state.budget = Number(e.target.value);
  $("#budget-hint").textContent = state.budget;
  updateBudgetPct();
});

function updateBudgetPct() {
  const ctx = state.modelContextWindow || 131072;
  const pct = (state.budget / ctx) * 100;
  // Format the context size for the label — 131072 → "128K", 32768 → "32K"
  const ctxLabel = ctx >= 1024 ? `${Math.round(ctx / 1024)}K` : `${ctx}`;
  $("#budget-pct").textContent = `~${pct.toFixed(2)}% of ${ctxLabel} context`;
}
$("#max-new").addEventListener("change", (e) => {
  state.maxNew = Number(e.target.value);
});
$("#new-session").addEventListener("click", () => createSession());
$("#restart").addEventListener("click", () => {
  if (confirm("Restart the container? All sessions and VRAM will be cleared.")) {
    fetch("/api/restart", { method: "POST" }).then(() => {
      document.body.innerHTML = "<div style='padding:40px;font-family:monospace'>Restarting… refresh in a few seconds.</div>";
    });
  }
});

// ── Status polling ────────────────────────────────────────────────────
async function pollHealth() {
  try {
    const r = await fetch("/api/health");
    if (!r.ok) throw new Error(`health ${r.status}`);
    const h = await r.json();
    if (h.context_window && h.context_window !== state.modelContextWindow) {
      state.modelContextWindow = h.context_window;
      // Rescale the slider to be a sane fraction of the new context
      const max = Math.min(8192, Math.max(1024, Math.floor(h.context_window / 16)));
      const def = h.default_budget || 512;
      const slider = $("#budget");
      slider.max = String(max);
      if (def <= Number(slider.max) && def >= Number(slider.min)) {
        slider.value = String(def);
        state.budget = def;
        $("#budget-hint").textContent = String(def);
      } else {
        $("#budget-hint").textContent = String(state.budget);
      }
      updateBudgetPct();
    }
    setStatus(h.ready ? "ready" : "warming", h);
  } catch (e) {
    setStatus("error", { error: String(e) });
  }
}
function setStatus(kind, h) {
  const dot = $("#status-dot");
  const text = $("#status-text");
  dot.className = "dot";
  if (kind === "ready") {
    dot.classList.add("dot-ready");
    const v = h.vram_allocated_mb ? `${(h.vram_allocated_mb/1024).toFixed(1)} GB` : "CPU";
    text.textContent = `${h.model.split("/").pop()} · ${h.gpu} · ${v}`;
  } else if (kind === "warming") {
    dot.classList.add("dot-init");
    text.textContent = "warming up…";
  } else {
    dot.classList.add("dot-bad");
    text.textContent = "disconnected";
  }
}

// ── Session ────────────────────────────────────────────────────────────
async function createSession(systemPrompt) {
  const r = await fetch("/api/session/create", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ system_prompt: systemPrompt ?? null }),
  });
  const s = await r.json();
  state.sessionId = s.session_id;
  state.messages = s.messages;
  $("#session-info").textContent = `Session ${state.sessionId.slice(0, 8)}…`;
  return s;
}
async function appendSession(role, content) {
  if (!state.sessionId) return;
  await fetch("/api/session/append", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ session_id: state.sessionId, role, content }),
  });
}

// ── SSE helper ─────────────────────────────────────────────────────────
// Posts JSON, reads an SSE stream, calls onEvent for each event object.
async function ssePost(path, body, onEvent, signal) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) {
    const text = await res.text();
    onEvent({ type: "error", error: "http", message: `${res.status}: ${text}` });
    return;
  }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      for (const line of frame.split("\n")) {
        if (!line.startsWith("data: ")) continue;
        try {
          const ev = JSON.parse(line.slice(6));
          onEvent(ev);
        } catch (e) { /* ignore */ }
      }
    }
  }
}

// ── Chat panel ────────────────────────────────────────────────────────
function renderChat() {
  return `
    <div class="scenario">
      <h2>Chat</h2>
      <p class="lede">Free-form conversation. Each turn is rebuilt with the
      model's chat template and passed through the patched <code>generate</code>;
      KiaOmni evicts if the rebuilt prompt exceeds the budget.</p>
      <div class="messages" id="msgs"></div>
      <form class="composer" id="composer">
        <div class="composer-card">
          <textarea id="user-input" placeholder="Type a message…  (Enter to send · Shift+Enter for newline)" required></textarea>
          <div class="row">
            <button type="submit" class="btn-primary" id="send-btn">Send</button>
            <button type="button" id="cancel-btn" class="btn-secondary" disabled>Cancel</button>
            <span class="muted" id="chat-status"></span>
          </div>
        </div>
      </form>
    </div>`;
}

let currentAssistantEl = null;
let currentText = "";

function setupChatHandlers() {
  const form = $("#composer");
  const input = $("#user-input");
  const cancelBtn = $("#cancel-btn");
  const sendBtn = $("#send-btn");
  const status = $("#chat-status");

  // Enter to send, Shift+Enter for newline
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      form.requestSubmit();
    }
  });
  // Auto-grow the textarea as the user types
  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 220) + "px";
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;

    if (!state.sessionId) await createSession();
    await appendSession("user", text);
    state.messages.push({ role: "user", content: text, ts: new Date().toISOString() });
    input.value = "";
    input.style.height = "auto";
    renderMessages();

    // Stream assistant reply
    sendBtn.disabled = true;
    cancelBtn.disabled = false;
    state.abortCtrl = new AbortController();
    currentText = "";
    currentAssistantEl = appendMessage("assistant", "▍");
    const msgsEl = $("#msgs");
    if (msgsEl) msgsEl.scrollTo({ top: msgsEl.scrollHeight, behavior: "smooth" });

    let stats = null;
    await ssePost("/api/chat", {
      messages: state.messages,
      policy: state.policy,
      budget: state.budget,
      max_new_tokens: state.maxNew,
      session_id: state.sessionId,
    }, (ev) => {
      if (ev.type === "token") {
        currentText += ev.text;
        if (currentAssistantEl) {
          const stream = currentAssistantEl.querySelector(".answer-stream");
          if (stream) stream.textContent = currentText + "▍";
        }
        if (msgsEl) msgsEl.scrollTo({ top: msgsEl.scrollHeight, behavior: "smooth" });
      } else if (ev.type === "status" && ev.phase === "prefill") {
        status.textContent = `compressing ${ev.tokens_in} → ${ev.tokens_kept} tokens…`;
      } else if (ev.type === "stats") {
        stats = ev.stats;
        if (currentAssistantEl) {
          const stream = currentAssistantEl.querySelector(".answer-stream");
          if (stream) stream.textContent = currentText;
          const meta = currentAssistantEl.querySelector(".msg-meta");
          if (meta) {
            meta.innerHTML = renderMetaTags({
              policy: state.policy,
              kept: stats.tokens_kept,
              in: stats.tokens_in,
              compression: (stats.tokens_kept / stats.tokens_in * 100).toFixed(1) + "%",
              prefill: (stats.prefill_ms / 1000).toFixed(2) + "s",
              decode: (stats.decode_ms / 1000).toFixed(2) + "s",
              tps: stats.tok_per_sec.toFixed(1),
              vram: formatVram(stats) || "?",
              ik: stats.keep_indices ? stats.keep_indices.length : null,
            });
          }
        }
        status.textContent = "";
      } else if (ev.type === "error") {
        if (ev.error === "oom") state.oomCount++;
        if (currentAssistantEl) {
          const stream = currentAssistantEl.querySelector(".answer-stream");
          if (stream) stream.innerHTML = `<span class="tag tag-fail">${ev.error}</span> ${ev.message}`;
        }
        status.textContent = "";
      }
    }, state.abortCtrl.signal);

    // Append assistant message to session
    if (currentText) {
      await appendSession("assistant", currentText);
      const assistantMsg = {
        role: "assistant",
        content: currentText,
        ts: new Date().toISOString(),
      };
      if (stats) {
        assistantMsg.stats = {
          tokens_in: stats.tokens_in,
          tokens_kept: stats.tokens_kept,
          compression_pct: (stats.tokens_kept / Math.max(1, stats.tokens_in) * 100),
          prefill_ms: stats.prefill_ms,
          decode_ms: stats.decode_ms,
          tok_per_sec: stats.tok_per_sec,
          vram_max_mb: stats.vram_max_allocated_mb,
          vram_reserved_mb: stats.vram_reserved_mb,
        };
      }
      state.messages.push(assistantMsg);
    }
    currentAssistantEl = null;
    sendBtn.disabled = false;
    cancelBtn.disabled = true;
  });

  cancelBtn.addEventListener("click", () => {
    if (state.abortCtrl) state.abortCtrl.abort();
  });

  // Save / Copy / Download
  $("#chat-save-json").addEventListener("click", () => saveChatAsJson());
  $("#chat-copy-text").addEventListener("click", () => copyChatAsText());
  $("#chat-download-text").addEventListener("click", () => downloadChatAsText());
}

function renderMessages() {
  const wrap = $("#msgs");
  if (!wrap) return;
  wrap.innerHTML = state.messages.map((m) => msgHTML(m)).join("");
  wrap.scrollTop = wrap.scrollHeight;
}

function appendMessage(role, placeholder) {
  const wrap = $("#msgs");
  if (!wrap) return null;
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.innerHTML = `
    <div class="role">${role}${role === "assistant" ? '<span class="cursor" aria-hidden="true"></span>' : ""}</div>
    <div class="answer-stream">${escapeHTML(placeholder)}</div>
    <div class="msg-meta"></div>`;
  wrap.appendChild(div);
  return div;
}

function msgHTML(m) {
  return `<div class="msg ${m.role}">
    <div class="role">${m.role}</div>
    <div>${escapeHTML(m.content)}</div>
  </div>`;
}

function renderMetaTags({ policy, kept, in: tin, compression, prefill, decode, tps, vram, ik }) {
  const cls = policy === "fullcontext" ? "tag-vanilla"
            : policy === "kiaomni_gaussian" ? "tag-kia"
            : "tag-kia";
  const els = [
    `<span class="tag ${cls}">${policy}</span>`,
    `<span class="tag">${kept}/${tin} (${compression})</span>`,
    `<span class="tag">prefill ${prefill}</span>`,
    `<span class="tag">decode ${decode}</span>`,
    `<span class="tag">${tps} tok/s</span>`,
    `<span class="tag">VRAM ${vram}</span>`,
  ];
  if (ik != null) els.push(`<span class="tag">ik ${ik}</span>`);
  return els.join("");
}

function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
}

// ── Demo Tasks panel ──────────────────────────────────────────────────
function renderDemo() {
  return `
    <div class="scenario">
      <h2>Demo Tasks</h2>
      <p class="lede">The four NIAH-style tasks from
      <code>notebook/demo/kiaomni_vs_snapkv_kaggle.ipynb</code>, run live on
      the deployed model with auto-grading. Reference numbers (Qwen2.5-7B NF4,
      B=512): single 100% · multi 100% · reason 40% · summary 100%.</p>
      <div class="composer">
        <div class="row">
          <div class="field" style="flex:1 1 140px">
            <label>Task</label>
            <select id="demo-task">
              <option value="single">Single-needle</option>
              <option value="multi">Multi-needle (3)</option>
              <option value="reason">Reasoning (4-hop)</option>
              <option value="summary">Summary (8 facts)</option>
              <option value="all">All 4</option>
            </select>
          </div>
          <div class="field" style="flex:1 1 100px">
            <label>Samples / task</label>
            <input id="demo-n" type="number" min="1" max="10" value="3" />
          </div>
          <div class="field" style="flex:0 0 auto; align-self:flex-end">
            <button id="demo-run" class="btn-primary">Run</button>
          </div>
        </div>
      </div>
      <div id="demo-out"></div>
    </div>`;
}

function setupDemoHandlers() {
  $("#demo-run").addEventListener("click", async () => {
    const task = $("#demo-task").value;
    const n = Number($("#demo-n").value);
    const out = $("#demo-out");
    out.innerHTML = `<p class="muted">Running…</p>`;
    try {
      const r = await fetch("/api/demo/run", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ task, policy: state.policy, budget: state.budget, n_samples: n }),
      });
      if (!r.ok) { out.innerHTML = `<div class="tag tag-fail">HTTP ${r.status}</div>`; return; }
      const data = await r.json();
      out.innerHTML = renderDemoTable(data);
    } catch (e) {
      out.innerHTML = `<div class="tag tag-fail">${escapeHTML(String(e))}</div>`;
    }
  });
}

function renderDemoTable(data) {
  const rows = data.results.map((taskRes) => {
    const pct = (taskRes.mean_score * 100).toFixed(0);
    const cls = taskRes.mean_score >= 0.9 ? "tag-pass"
              : taskRes.mean_score >= 0.3 ? "tag-warn" : "tag-fail";
    return `<tr>
      <td>${taskRes.task}</td>
      <td>${taskRes.n_samples}</td>
      <td>${taskRes.n_pass}</td>
      <td><span class="tag ${cls}">${pct}%</span></td>
      <td style="color:var(--ink-3)">${taskRes.samples.slice(0, 1).map((s) => s.detail ?? "—").join("; ")}</td>
    </tr>`;
  }).join("");
  const samples = data.results.flatMap((t) => t.samples.map((s) =>
    `<details><summary>${t.task} #${s.sid} · ${s.info ?? ""}</summary>
       <div style="margin-top:8px"><b>Answer:</b> <code>${escapeHTML(s.answer ?? "(no answer)")}</code></div>
       <div><b>Gold:</b> <code>${escapeHTML(s.gold ?? "")}</code></div>
       <div><b>Verdict:</b> <span class="tag ${s.score >= 0.99 ? "tag-pass" : "tag-fail"}">${s.detail ?? s.error ?? ""}</span></div>
       <div class="msg-meta" style="margin-top:6px">
         ${s.stats ? `<span class="tag">${s.stats.tokens_in}→${s.stats.tokens_kept}</span>
                     <span class="tag">${(s.stats.prefill_ms/1000).toFixed(2)}s prefill</span>
                     <span class="tag">${s.stats.tok_per_sec.toFixed(1)} tok/s</span>` : ""}
       </div>
     </details>`
  )).join("");
  return `
    <table class="demo-table">
      <thead><tr><th>Task</th><th>N</th><th>Pass</th><th>Mean</th><th>Detail</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <details style="margin-top:14px"><summary>Per-sample answers (${data.results.reduce((a, t) => a + t.samples.length, 0)})</summary>
      <div style="margin-top:8px">${samples}</div>
    </details>`;
}

// ── Document Q&A panel ───────────────────────────────────────────────
function renderDocQA() {
  return `
    <div class="scenario">
      <h2>Document Q&amp;A</h2>
      <p class="lede">Paste a document and one or more questions. The model is
      instructed to answer only from the document; the eviction policy keeps
      the relevant tokens under the configured budget.</p>
      <div class="composer">
        <div class="composer-card">
          <label class="muted" for="doc-text">Document</label>
          <textarea id="doc-text" placeholder="Paste your document here…" style="min-height:160px"></textarea>
          <div class="row">
            <button id="docqa-load-sample" class="btn-secondary" type="button">Load sample article</button>
          </div>
          <label class="muted" for="doc-qs">Questions (one per line)</label>
          <textarea id="doc-qs" placeholder="What is the main thesis?&#10;Who are the key stakeholders?"></textarea>
          <div class="row">
            <button id="docqa-run" class="btn-primary">Run</button>
            <span class="muted" id="docqa-status"></span>
          </div>
        </div>
      </div>
      <div id="docqa-out"></div>
    </div>`;
}

function setupDocQAHandlers() {
  $("#docqa-load-sample").addEventListener("click", async () => {
    const ta = $("#doc-text");
    if (!ta) return;
    let article = "";
    try {
      const r = await fetch("/static/sample_data/article.txt");
      if (r.ok) article = (await r.text()).trim();
    } catch (_) { /* ignore */ }
    if (!article) article = ARTICLE_FALLBACK.trim();
    ta.value = article;
    ta.dispatchEvent(new Event("input"));
    const qs = $("#doc-qs");
    if (qs && !qs.value.trim()) {
      qs.value = TEST_QUESTIONS.map((t, i) => `(Q${i + 1}) ${t.q}`).join("\n");
      qs.dispatchEvent(new Event("input"));
    }
  });

  $("#docqa-run").addEventListener("click", async () => {
    const doc = $("#doc-text").value.trim();
    const qs = $("#doc-qs").value.split("\n").map((s) => s.trim()).filter(Boolean);
    if (!doc || !qs.length) return;
    const out = $("#docqa-out");
    out.innerHTML = `<p class="muted">Running…</p>`;
    try {
      const r = await fetch("/api/docqa", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ document: doc, questions: qs, policy: state.policy, budget: state.budget, max_new_tokens: state.maxNew }),
      });
      if (!r.ok) { out.innerHTML = `<div class="tag tag-fail">HTTP ${r.status}</div>`; return; }
      const data = await r.json();
      out.innerHTML = data.answers.map((a) => `
        <div class="msg assistant">
          <div class="role">question</div>
          <div>${escapeHTML(a.question)}</div>
          <div class="role" style="margin-top:10px">answer</div>
          <div>${escapeHTML(a.answer ?? a.error ?? "(no answer)")}</div>
          ${a.stats ? `<div class="msg-meta">${renderMetaTags({ policy: state.policy, kept: a.stats.tokens_kept, in: a.stats.tokens_in, compression: (a.stats.tokens_kept / a.stats.tokens_in * 100).toFixed(1) + "%", prefill: (a.stats.prefill_ms / 1000).toFixed(2) + "s", decode: (a.stats.decode_ms / 1000).toFixed(2) + "s", tps: a.stats.tok_per_sec.toFixed(1), vram: formatVram(a.stats) || "?", ik: a.stats.keep_indices ? a.stats.keep_indices.length : null })}</div>` : ""}
        </div>`).join("");
    } catch (e) {
      out.innerHTML = `<div class="tag tag-fail">${escapeHTML(String(e))}</div>`;
    }
  });
}

// ── Compare panel (multi-turn, side-by-side, real chat) ─────────────
function renderCompare() {
  return `
    <div class="scenario">
      <h2>Side-by-Side Compare</h2>
      <p class="lede">One composer, four columns. Every turn runs all
      policies against the full context — each evicts independently.
      FullContext is the baseline; SnapKV and KiaOmni show how different
      compression strategies perform.</p>
      <div class="compare-grid" id="cmp-grid">
        ${["fullcontext", "kiaomni_s8", "kiaomni_gaussian", "snapkv"].map((p) => {
          const tagCls = p === "fullcontext" ? "tag-vanilla"
                       : p === "snapkv" ? "tag-snapkv"
                       : "tag-kia";
          return `
          <div class="compare-col" data-policy="${p}">
            <div class="compare-col-header">
              <span class="tag ${tagCls}">${p}</span>
              <span class="muted compare-col-stats" data-stats-for="${p}">—</span>
            </div>
            <div class="compare-col-messages" data-messages-for="${p}">
              <div class="compare-msg-empty">waiting for first turn…</div>
            </div>
          </div>`;
        }).join("")}
      </div>
      <div class="compare-toolbar">
        <button id="cmp-load-sample" class="btn-secondary" type="button">Load sample article</button>
        <button id="cmp-save-json" class="btn-secondary" type="button">Save chat (JSON)</button>
        <button id="cmp-copy-text" class="btn-secondary" type="button">Copy text</button>
        <button id="cmp-download-text" class="btn-secondary" type="button">Download .txt</button>
        <button id="cmp-reset" class="btn-secondary" type="button">Reset all</button>
        <span class="muted" id="cmp-status"></span>
      </div>
      <form class="composer" id="cmp-composer">
        <div class="composer-card">
          <textarea id="cmp-input" placeholder="Type a turn…  (Enter to send · Shift+Enter for newline)" required></textarea>
          <div class="row">
            <button type="submit" class="btn-primary" id="cmp-send">Send turn</button>
            <button type="button" id="cmp-cancel" class="btn-secondary" disabled>Cancel</button>
          </div>
        </div>
      </form>
    </div>`;
}

const POLICIES = ["fullcontext", "kiaomni_s8", "kiaomni_gaussian", "snapkv"];
const compareState = { turns: [] };   // [{ts, user, responses: {policy: {text, stats}}}]

function renderCmpMsg(role, text, stats, policy) {
  const cls = role === "user" ? "user" : "assistant";
  const safeText = escapeHTML(text ?? "");
  let meta = "";
  if (role === "assistant" && stats) {
    const pct = stats.tokens_in ? (stats.tokens_kept / stats.tokens_in * 100).toFixed(1) : "—";
    meta = `<div class="compare-msg-meta">
      <span class="tag">${stats.tokens_in}→${stats.tokens_kept} (${pct}%)</span>
      <span class="tag">prefill ${(stats.prefill_ms/1000).toFixed(2)}s</span>
      <span class="tag">decode ${(stats.decode_ms/1000).toFixed(2)}s</span>
      <span class="tag">${stats.tok_per_sec.toFixed(1)} tok/s</span>
      <span class="tag">${formatVram(stats) || "?"}</span>
    </div>`;
  }
  return `<div class="compare-msg ${cls}">${safeText}${meta}</div>`;
}

function appendCmpTurn(userText, responses) {
  // Append a user bubble to every column, then an assistant bubble to the
  // matching column. Auto-scroll each column to the bottom.
  const ts = new Date().toISOString();
  for (const p of POLICIES) {
    const msgs = document.querySelector(`[data-messages-for="${p}"]`);
    if (!msgs) continue;
    // Remove the empty-state placeholder on first turn
    const empty = msgs.querySelector(".compare-msg-empty");
    if (empty) empty.remove();
    msgs.insertAdjacentHTML("beforeend", renderCmpMsg("user", userText, null, p));
    const r = responses[p];
    if (r && !r.error) {
      msgs.insertAdjacentHTML("beforeend", renderCmpMsg("assistant", r.text, r.stats, p));
      // Update column header stats
      const statsEl = document.querySelector(`[data-stats-for="${p}"]`);
      if (statsEl && r.stats) {
        const pct = r.stats.tokens_in ? (r.stats.tokens_kept / r.stats.tokens_in * 100).toFixed(1) : "—";
        statsEl.textContent = `${r.stats.tokens_in}→${r.stats.tokens_kept} (${pct}%) · ${(r.stats.tok_per_sec).toFixed(1)} tok/s`;
      }
    } else if (r && r.error) {
      msgs.insertAdjacentHTML("beforeend", `<div class="compare-msg assistant"><span class="tag tag-fail">${r.error}</span> ${escapeHTML(r.message ?? "")}</div>`);
    }
    // Auto-scroll to bottom
    msgs.scrollTo({ top: msgs.scrollHeight, behavior: "smooth" });
  }
  // Persist in JS state
  compareState.turns.push({ ts, user: userText, responses });
}

function compareHistoryAsMessages() {
  // Flatten the 3-column history into a single messages[] list. We use
  // the FullContext column as the canonical thread (its messages are the
  // best-quality by construction).
  const msgs = [];
  for (const turn of compareState.turns) {
    msgs.push({ role: "user", content: turn.user });
    const r = turn.responses.fullcontext;
    if (r && r.text) msgs.push({ role: "assistant", content: r.text });
  }
  return msgs;
}

async function loadSampleIntoCmp() {
  let article = "";
  try {
    const r = await fetch("/static/sample_data/article.txt");
    if (r.ok) article = (await r.text()).trim();
  } catch (_) { /* ignore */ }
  if (!article) article = ARTICLE_FALLBACK.trim();
  const questionsBlock = TEST_QUESTIONS.map((t, i) =>
    `(Q${i + 1}) ${t.q}`).join("\n\n");
  const ta = $("#cmp-input");
  if (ta) {
    ta.value = article +
      "\n\n---\n\n" +
      "Use the document above to answer each of the following 9 questions. " +
      "Number your answers Q1, Q2, ..., Q9 in order. " +
      "For factual questions, give the exact date, name, or number. " +
      "For summarization and reasoning, be concise but complete.\n\n" +
      questionsBlock;
    ta.dispatchEvent(new Event("input"));
    ta.focus();
  }
}

function setupCompareHandlers() {
  const form = $("#cmp-composer");
  const input = $("#cmp-input");
  const sendBtn = $("#cmp-send");
  const cancelBtn = $("#cmp-cancel");
  const status = $("#cmp-status");

  // Enter to send, Shift+Enter for newline
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      form.requestSubmit();
    }
  });
  // Auto-grow
  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 220) + "px";
  });

  state.compareAbort = null;

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    // Render the user bubble in all 4 columns immediately
    appendCmpTurn(text, { fullcontext: { empty: true }, kiaomni_s8: { empty: true }, kiaomni_gaussian: { empty: true }, snapkv: { empty: true } });
    input.value = "";
    input.style.height = "auto";

    sendBtn.disabled = true;
    cancelBtn.disabled = false;
    state.compareAbort = new AbortController();
    status.textContent = "running 4 policies…";

    // Build the history we'll send. Use the FullContext thread as canonical.
    // For the first turn, history is just this user turn.
    const hist = compareHistoryAsMessages();

    try {
      const r = await fetch("/api/compare/turn", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          history: hist,
          budget: state.budget,
          max_new_tokens: state.maxNew,
          policies: POLICIES,
        }),
        signal: state.compareAbort.signal,
      });
      if (!r.ok) {
        status.innerHTML = `<span class="tag tag-fail">HTTP ${r.status}</span>`;
        sendBtn.disabled = false;
        cancelBtn.disabled = true;
        return;
      }
      const data = await r.json();
      // Replace the placeholder user bubble + append assistant bubbles
      // for each policy. We do this by popping the last "empty" user
      // bubble per column and inserting the assistant response.
      const responses = {};
      for (const res of data.results) {
        responses[res.policy] = res.error
          ? { error: res.error, message: res.message }
          : { text: res.text, stats: res.stats, wall_ms: res.wall_ms };
      }
      // Remove the empty placeholders we inserted
      for (const p of POLICIES) {
        const msgs = document.querySelector(`[data-messages-for="${p}"]`);
        if (msgs) {
          const empties = msgs.querySelectorAll(".compare-msg.user");
          // The last user bubble is our just-inserted placeholder
          if (empties.length) empties[empties.length - 1].remove();
        }
      }
      // Now insert the real user bubble (without empty) + assistant reply
      for (const p of POLICIES) {
        const msgs = document.querySelector(`[data-messages-for="${p}"]`);
        if (!msgs) continue;
        msgs.insertAdjacentHTML("beforeend", renderCmpMsg("user", text, null, p));
        const r2 = responses[p];
        if (r2 && !r2.error) {
          msgs.insertAdjacentHTML("beforeend", renderCmpMsg("assistant", r2.text, r2.stats, p));
          const statsEl = document.querySelector(`[data-stats-for="${p}"]`);
          if (statsEl && r2.stats) {
            const pct = r2.stats.tokens_in ? (r2.stats.tokens_kept / r2.stats.tokens_in * 100).toFixed(1) : "—";
            statsEl.textContent = `${r2.stats.tokens_in}→${r2.stats.tokens_kept} (${pct}%) · ${(r2.stats.tok_per_sec).toFixed(1)} tok/s`;
          }
        } else if (r2 && r2.error) {
          msgs.insertAdjacentHTML("beforeend", `<div class="compare-msg assistant"><span class="tag tag-fail">${r2.error}</span> ${escapeHTML(r2.message ?? "")}</div>`);
        }
        msgs.scrollTo({ top: msgs.scrollHeight, behavior: "smooth" });
      }
      // Persist
      const ts = new Date().toISOString();
      compareState.turns[compareState.turns.length - 1] = { ts, user: text, responses };
      status.textContent = "";
    } catch (err) {
      if (err.name === "AbortError") {
        status.textContent = "cancelled";
      } else {
        status.innerHTML = `<span class="tag tag-fail">${escapeHTML(String(err))}</span>`;
      }
    } finally {
      sendBtn.disabled = false;
      cancelBtn.disabled = true;
      state.compareAbort = null;
    }
  });

  cancelBtn.addEventListener("click", () => {
    if (state.compareAbort) state.compareAbort.abort();
  });

  $("#cmp-load-sample").addEventListener("click", loadSampleIntoCmp);
  $("#cmp-reset").addEventListener("click", () => {
    compareState.turns = [];
    for (const p of POLICIES) {
      const msgs = document.querySelector(`[data-messages-for="${p}"]`);
      if (msgs) msgs.innerHTML = `<div class="compare-msg-empty">waiting for first turn…</div>`;
      const statsEl = document.querySelector(`[data-stats-for="${p}"]`);
      if (statsEl) statsEl.textContent = "—";
    }
    status.textContent = "reset";
    setTimeout(() => { status.textContent = ""; }, 1200);
  });

  // Save / Copy / Download
  $("#cmp-save-json").addEventListener("click", () => saveCompareAsJson());
  $("#cmp-copy-text").addEventListener("click", () => copyCompareAsText());
  $("#cmp-download-text").addEventListener("click", () => downloadCompareAsText());
}

function compareRecord() {
  return {
    schema_version: 1,
    created_at: new Date().toISOString(),
    deploy_url: window.location.origin,
    model: "Qwen/Qwen2.5-7B-Instruct",
    mode: "side_by_side",
    policies: POLICIES,
    budget: state.budget,
    max_new_tokens: state.maxNew,
    turn_count: compareState.turns.length,
    turns: compareState.turns.map((t) => ({
      ts: t.ts,
      user: t.user,
      responses: Object.fromEntries(
        Object.entries(t.responses).map(([k, v]) => [k, v && !v.error ? { text: v.text, stats: v.stats, wall_ms: v.wall_ms } : v])
      ),
    })),
  };
}

function compareAsText() {
  const rec = compareRecord();
  const lines = [];
  lines.push(`KiaOmni Side-by-Side Chat Export`);
  lines.push(`Created:   ${rec.created_at}`);
  lines.push(`Model:     ${rec.model}`);
  lines.push(`Mode:      ${rec.mode}`);
  lines.push(`Budget:    ${rec.budget}`);
  lines.push(`Max new:   ${rec.max_new_tokens}`);
  lines.push(`Policies:  ${rec.policies.join(", ")}`);
  lines.push(`Turns:     ${rec.turn_count}`);
  lines.push("=".repeat(72));
  rec.turns.forEach((t, i) => {
    const stamp = t.ts ? new Date(t.ts).toISOString() : `Turn ${i+1}`;
    lines.push("");
    lines.push(`--- ${stamp} ---`);
    lines.push(`[USER]`);
    lines.push(t.user);
    lines.push("");
    for (const p of rec.policies) {
      const r = t.responses[p];
      lines.push(`[${p}]`);
      if (r.error) {
        lines.push(`  ERROR ${r.error}: ${r.message ?? ""}`);
      } else if (r.stats) {
        const pct = r.stats.tokens_in ? (r.stats.tokens_kept / r.stats.tokens_in * 100).toFixed(1) : "—";
        lines.push(`  [${r.stats.tokens_in}→${r.stats.tokens_kept} tok (${pct}%) · prefill ${(r.stats.prefill_ms/1000).toFixed(2)}s · decode ${(r.stats.decode_ms/1000).toFixed(2)}s · ${r.stats.tok_per_sec.toFixed(1)} tok/s · ${formatVram(r.stats) || "?"}]`);
      }
      lines.push(r.text || "");
      lines.push("");
    }
  });
  return lines.join("\n");
}

function saveCompareAsJson() {
  if (!compareState.turns.length) { alert("No turns to save yet."); return; }
  downloadBlob(JSON.stringify(compareRecord(), null, 2), timestampedFilename("kiaomni_compare", "json"), "application/json");
}
function copyCompareAsText() {
  if (!compareState.turns.length) { alert("No turns to copy yet."); return; }
  navigator.clipboard.writeText(compareAsText()).then(
    () => flashSaveStatus("Copied!"),
    () => alert("Clipboard write failed — try Download instead.")
  );
}
function downloadCompareAsText() {
  if (!compareState.turns.length) { alert("No turns to download yet."); return; }
  downloadBlob(compareAsText(), timestampedFilename("kiaomni_compare", "txt"), "text/plain");
}

// ── Multi-Turn panel ──────────────────────────────────────────────────
function renderMultiTurn() {
  return `
    <div class="scenario">
      <h2>Multi-Turn (rolling context)</h2>
      <p class="lede">Send many turns in a row. The conversation history grows
      past the budget and KiaOmni evicts on every turn. The VRAM chart at the
      bottom shows whether memory stays flat as the chat extends.</p>
      <div id="msgs-multi" class="messages"></div>
      <form class="composer" id="composer-multi">
        <div class="composer-card">
          <textarea id="user-input-multi" placeholder="Message turn…  (Enter to send · Shift+Enter for newline)"></textarea>
          <div class="row">
            <button type="submit" class="btn-primary">Send turn</button>
            <button type="button" id="reset-multi" class="btn-secondary">Reset</button>
            <span class="muted" id="multi-status"></span>
          </div>
        </div>
      </form>
    </div>`;
}

function setupMultiTurnHandlers() {
  const input = $("#user-input-multi");
  const form = $("#composer-multi");

  // Enter to send, Shift+Enter for newline
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      form.requestSubmit();
    }
  });
  // Auto-grow
  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 220) + "px";
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    if (!state.sessionId) await createSession();
    await appendSession("user", text);
    state.messages.push({ role: "user", content: text });
    renderMultiMessages();
    input.value = "";
    input.style.height = "auto";

    const status = $("#multi-status");
    status.textContent = "compressing + generating…";
    state.abortCtrl = new AbortController();
    currentText = "";
    currentAssistantEl = appendMultiMessage("assistant", "▍");
    const msgsEl = $("#msgs-multi");
    if (msgsEl) msgsEl.scrollTo({ top: msgsEl.scrollHeight, behavior: "smooth" });

    let stats = null;
    await ssePost("/api/chat", {
      messages: state.messages,
      policy: state.policy,
      budget: state.budget,
      max_new_tokens: state.maxNew,
      session_id: state.sessionId,
    }, (ev) => {
      if (ev.type === "token") {
        currentText += ev.text;
        if (currentAssistantEl) {
          const stream = currentAssistantEl.querySelector(".answer-stream");
          if (stream) stream.textContent = currentText + "▍";
        }
        if (msgsEl) msgsEl.scrollTo({ top: msgsEl.scrollHeight, behavior: "smooth" });
      } else if (ev.type === "stats") {
        stats = ev.stats;
        if (currentAssistantEl) {
          const stream = currentAssistantEl.querySelector(".answer-stream");
          if (stream) stream.textContent = currentText;
          const meta = currentAssistantEl.querySelector(".msg-meta");
          if (meta) meta.innerHTML = renderMetaTags({
            policy: state.policy, kept: stats.tokens_kept, in: stats.tokens_in,
            compression: (stats.tokens_kept / stats.tokens_in * 100).toFixed(1) + "%",
            prefill: (stats.prefill_ms / 1000).toFixed(2) + "s",
            decode: (stats.decode_ms / 1000).toFixed(2) + "s",
            tps: stats.tok_per_sec.toFixed(1),
            vram: formatVram(stats) || "?",
            ik: stats.keep_indices ? stats.keep_indices.length : null,
          });
        }
        status.textContent = "";
      } else if (ev.type === "error") {
        if (ev.error === "oom") state.oomCount++;
        status.textContent = ev.message;
      }
    }, state.abortCtrl.signal);

    if (currentText) {
      await appendSession("assistant", currentText);
      state.messages.push({ role: "assistant", content: currentText });
    }
    currentAssistantEl = null;
  });
  $("#reset-multi").addEventListener("click", async () => {
    if (state.sessionId) await fetch(`/api/session/${state.sessionId}`, { method: "DELETE" });
    state.sessionId = null;
    state.messages = [];
    renderMultiMessages();
  });
}

function renderMultiMessages() {
  const wrap = $("#msgs-multi");
  if (!wrap) return;
  wrap.innerHTML = state.messages.map((m) => msgHTML(m)).join("");
  wrap.scrollTop = wrap.scrollHeight;
}

function appendMultiMessage(role, placeholder) {
  const wrap = $("#msgs-multi");
  if (!wrap) return null;
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.innerHTML = `<div class="role">${role}${role === "assistant" ? '<span class="cursor" aria-hidden="true"></span>' : ""}</div>
    <div class="answer-stream">${escapeHTML(placeholder)}</div>
    <div class="msg-meta"></div>`;
  wrap.appendChild(div);
  return div;
}

// ── Render dispatch ───────────────────────────────────────────────────
function renderPanel() {
  const panel = $("#panel");
  switch (state.tab) {
    case "chat":     panel.innerHTML = renderChat();     setupChatHandlers();      renderMessages(); break;
    case "demo":     panel.innerHTML = renderDemo();     setupDemoHandlers();     break;
    case "docqa":    panel.innerHTML = renderDocQA();    setupDocQAHandlers();    break;
    case "compare":  panel.innerHTML = renderCompare();  setupCompareHandlers();  break;
    case "multiturn":panel.innerHTML = renderMultiTurn();setupMultiTurnHandlers();renderMultiMessages(); break;
  }
}

// ── Telemetry chart ───────────────────────────────────────────────────
const vramChart = {
  canvas: null, ctx: null, w: 0, h: 0, dpr: 1,
  data: [],  // {t, allocated, reserved, max}
  maxPoints: 200,
};
function initChart() {
  vramChart.canvas = $("#vram-chart");
  vramChart.ctx = vramChart.canvas.getContext("2d");
  vramChart.dpr = window.devicePixelRatio || 1;
  resizeChart();
  window.addEventListener("resize", resizeChart);
}
function resizeChart() {
  const c = vramChart.canvas;
  const rect = c.getBoundingClientRect();
  vramChart.w = rect.width;
  vramChart.h = rect.height;
  c.width = vramChart.w * vramChart.dpr;
  c.height = vramChart.h * vramChart.dpr;
  vramChart.ctx.scale(vramChart.dpr, vramChart.dpr);
  drawChart();
}
function drawChart() {
  const ctx = vramChart.ctx, w = vramChart.w, h = vramChart.h;
  ctx.clearRect(0, 0, w, h);
  if (vramChart.data.length < 2) {
    ctx.fillStyle = "oklch(0.66 0.012 240)";
    ctx.font = "11px ui-monospace, monospace";
    ctx.textAlign = "center";
    ctx.fillText("waiting for telemetry…", w / 2, h / 2);
    return;
  }
  const xs = vramChart.data.map((d) => d.t);
  const xmin = xs[0], xmax = xs[xs.length - 1];
  const ymax = Math.max(...vramChart.data.map((d) => d.reserved)) * 1.1 || 1;
  const xpad = 36, ypad = 14;
  const plotW = w - xpad - 4, plotH = h - ypad - 4;
  const xToPx = (x) => xpad + ((x - xmin) / (xmax - xmin || 1)) * plotW;
  const yToPx = (y) => h - ypad - (y / ymax) * plotH;

  // Grid
  ctx.strokeStyle = "oklch(0.28 0.010 240)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = ypad + (i / 4) * plotH;
    ctx.beginPath(); ctx.moveTo(xpad, y); ctx.lineTo(w - 4, y); ctx.stroke();
  }

  // Reserved (filled)
  ctx.fillStyle = "oklch(0.29 0.018 240 / 0.5)";
  ctx.beginPath();
  ctx.moveTo(xToPx(xs[0]), h - ypad);
  vramChart.data.forEach((d) => ctx.lineTo(xToPx(d.t), yToPx(d.reserved)));
  ctx.lineTo(xToPx(xs[xs.length - 1]), h - ypad);
  ctx.closePath(); ctx.fill();

  // Lines
  const lines = [
    { key: "max",       color: "oklch(0.78 0.16 195)", dash: [] },
    { key: "reserved",  color: "oklch(0.66 0.012 240)", dash: [] },
    { key: "allocated", color: "oklch(0.74 0.13 230)", dash: [] },
  ];
  for (const ln of lines) {
    ctx.strokeStyle = ln.color;
    ctx.lineWidth = 1.5;
    ctx.setLineDash(ln.dash);
    ctx.beginPath();
    vramChart.data.forEach((d, i) => {
      const x = xToPx(d.t), y = yToPx(d[ln.key]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
  ctx.setLineDash([]);

  // Y-axis label
  ctx.fillStyle = "oklch(0.66 0.012 240)";
  ctx.font = "10px ui-monospace, monospace";
  ctx.textAlign = "right";
  ctx.fillText(`${(ymax / 1024).toFixed(1)} GB`, xpad - 4, ypad + 8);
  ctx.fillText("0", xpad - 4, h - ypad);
}

async function pollTelemetry() {
  try {
    const r = await fetch("/api/telemetry");
    if (!r.ok) return;
    const t = await r.json();
    vramChart.data = t.snapshots.map((s) => ({
      t: s.t, allocated: s.allocated_mb, reserved: s.reserved_mb, max: s.max_allocated_mb,
    })).slice(-vramChart.maxPoints);
    drawChart();
    // Stats grid
    const last = t.snapshots[t.snapshots.length - 1];
    const lastReq = t.requests[t.requests.length - 1];
    const grid = $("#stats-grid");
    if (last && grid) {
      grid.innerHTML = `
        <div class="stat"><div class="label">VRAM alloc</div><div class="value">${(last.allocated_mb / 1024).toFixed(2)} GB</div></div>
        <div class="stat"><div class="label">VRAM reserved</div><div class="value">${(last.reserved_mb / 1024).toFixed(2)} GB</div></div>
        <div class="stat"><div class="label">VRAM peak</div><div class="value">${(last.max_allocated_mb / 1024).toFixed(2)} GB</div></div>
        <div class="stat"><div class="label">Fragmentation</div><div class="value">${last.fragmentation_pct.toFixed(1)}%</div></div>
        <div class="stat"><div class="label">Uptime</div><div class="value">${Math.floor(t.uptime_s)}s</div></div>
        <div class="stat"><div class="label">Requests</div><div class="value">${t.stats.total_requests}</div></div>
        <div class="stat"><div class="label">OOM count</div><div class="value">${t.oom_count}</div></div>
        ${lastReq ? `<div class="stat"><div class="label">Last tok/s</div><div class="value">${lastReq.tok_per_sec.toFixed(1)}</div></div>` : ""}
      `;
    }
    const oomBadge = $("#oom-badge");
    if (oomBadge) {
      oomBadge.innerHTML = t.oom_count > 0 ? `<span class="badge-oom">OOM ×${t.oom_count}</span>` : "";
    }
  } catch (e) { /* ignore */ }
}

// ── Init ─────────────────────────────────────────────────────────────
renderPanel();
initChart();
pollHealth();
pollTelemetry();   // populate the chart immediately
setInterval(pollHealth, 10_000);
setInterval(pollTelemetry, 2_000);
updateBudgetPct();  // make sure the label is correct on first paint
