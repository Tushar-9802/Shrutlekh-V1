"use strict";

const $ = (id) => document.getElementById(id);
const API = "/api";

let recordings = [];
let selected = null;
let pollTimer = null;
let transcriptOpen = false;

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const mmss = (t) => `${String(Math.floor(t / 60)).padStart(2, "0")}:${String(Math.floor(t % 60)).padStart(2, "0")}`;

function when(ts) {
    const d = new Date(ts * 1000);
    const today = new Date();
    const sameDay = d.toDateString() === today.toDateString();
    return sameDay
        ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
        : d.toLocaleDateString([], { day: "numeric", month: "short" });
}

async function api(path, options) {
    const res = await fetch(API + path, options);
    if (res.status === 204) {
        return null;
    }
    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
        throw new Error(body.detail || `${res.status} ${res.statusText}`);
    }
    return body;
}

// --- theme ---------------------------------------------------------------

function initTheme() {
    const saved = localStorage.getItem("shrutlekh-theme");
    if (saved) {
        document.documentElement.dataset.theme = saved;
    }
    $("theme-toggle").addEventListener("click", () => {
        const dark = document.documentElement.dataset.theme === "dark"
            || (!document.documentElement.dataset.theme
                && matchMedia("(prefers-color-scheme: dark)").matches);
        const next = dark ? "light" : "dark";
        document.documentElement.dataset.theme = next;
        localStorage.setItem("shrutlekh-theme", next);
    });
}

// --- config --------------------------------------------------------------

async function loadConfig() {
    const cfg = await api("/config");
    const labels = { hindi: "Hindi", hinglish: "Hinglish (code-mixed)" };
    $("mode-input").innerHTML = cfg.modes
        .map((m) => `<option value="${esc(m)}">${esc(labels[m] || m)}</option>`).join("");
    $("template-input").innerHTML = cfg.templates
        .map((t) => `<option value="${esc(t)}">${esc(t[0].toUpperCase() + t.slice(1))}</option>`).join("");
    const model = cfg.asr_model.split("/").pop();
    $("engine-note").textContent =
        `${cfg.tier.toUpperCase()} tier · ${model} · notes by ${cfg.ollama_model} · up to ${cfg.max_upload_mb} MB`;
}

// --- recordings list -----------------------------------------------------

function renderList() {
    const ul = $("recordings-list");
    ul.innerHTML = recordings.map((r) => {
        const dur = r.duration_s ? `${Math.round(r.duration_s)}s` : "";
        const bits = [when(r.created_at), r.mode, dur].filter(Boolean).join(" · ");
        return `<li><button type="button" data-id="${esc(r.id)}" aria-current="${r.id === selected}">
            <span class="rec-title"><span class="dot ${esc(r.status)}"></span>${esc(r.title)}</span>
            <span class="rec-meta">${esc(bits)}</span>
        </button></li>`;
    }).join("");
    $("recordings-empty").hidden = recordings.length > 0;
    for (const b of ul.querySelectorAll("button")) {
        b.addEventListener("click", () => select(b.dataset.id));
    }
}

async function refreshList() {
    recordings = await api("/recordings?limit=100");
    renderList();
    schedulePoll();
}

function schedulePoll() {
    clearTimeout(pollTimer);
    const busy = recordings.some((r) => r.status === "pending" || r.status === "processing");
    if (busy) {
        pollTimer = setTimeout(async () => {
            await refreshList();
            if (selected) {
                await showRecording(selected, true);
            }
        }, 2000);
    }
}

// --- one recording -------------------------------------------------------

function show(view) {
    for (const id of ["empty-state", "search-view", "recording-view"]) {
        $(id).hidden = id !== view;
    }
}

async function select(id) {
    selected = id;
    renderList();
    show("recording-view");
    await showRecording(id);
}

async function showRecording(id, quiet) {
    let rec;
    try {
        rec = await api(`/recordings/${id}`);
    } catch {
        selected = null;
        show("empty-state");
        return;
    }
    $("recording-title").textContent = rec.title;
    const model = rec.asr_model ? rec.asr_model.split("/").pop() : null;
    $("recording-meta").textContent = [
        rec.mode, rec.template,
        rec.duration_s ? `${Math.round(rec.duration_s)}s` : null,
        rec.tier && model ? `${rec.tier}/${model}` : null,
        `${rec.n_segments} segments`,
    ].filter(Boolean).join(" · ");

    const busy = rec.status === "pending" || rec.status === "processing";
    $("recording-progress").hidden = !busy;
    $("recording-progress").textContent = rec.status === "pending"
        ? "Queued…" : "Transcribing and summarising…";
    $("recording-error").hidden = !rec.error;
    $("recording-error").textContent = rec.error || "";

    if (rec.status !== "ready") {
        $("notes-section").hidden = true;
        $("transcript-section").hidden = true;
        return;
    }
    if (quiet && !$("notes-section").hidden) {
        return; // already rendered; a poll tick shouldn't reflow the page
    }

    const notes = await api(`/recordings/${id}/notes`);
    const note = notes[0];
    $("notes-section").hidden = !note;
    if (note) {
        $("notes-summary").innerHTML = note.summary.split(/\n{2,}/)
            .map((p) => `<p>${esc(p).replace(/\n/g, "<br>")}</p>`).join("");
        const items = note.action_items || [];
        $("action-items").hidden = items.length === 0;
        $("action-list").innerHTML = items.map((a) => {
            const who = [a.who, a.when].filter(Boolean).join(" · ");
            return `<li>${esc(a.what)}${who ? ` <span class="action-who">${esc(who)}</span>` : ""}</li>`;
        }).join("");
    }

    const segments = await api(`/recordings/${id}/segments`);
    $("transcript-section").hidden = segments.length === 0;
    $("transcript").innerHTML = segments.map((s) =>
        `<li><time>${mmss(s.start)}</time><span>${esc(s.text)}</span></li>`).join("");
    $("transcript").hidden = !transcriptOpen;
    $("transcript-toggle").textContent = transcriptOpen ? "hide" : "show";
}

// --- search --------------------------------------------------------------

async function runSearch(q) {
    const hits = await api(`/search?q=${encodeURIComponent(q)}&limit=30`);
    show("search-view");
    $("search-heading").textContent = `${hits.length} match${hits.length === 1 ? "" : "es"} for “${q}”`;
    $("search-empty").hidden = hits.length > 0;
    $("search-results").innerHTML = hits.map((h) =>
        `<li>
            <p class="hit-head">
                <button type="button" class="link-button" data-id="${esc(h.recording_id)}">${esc(h.title)}</button>
                &nbsp;·&nbsp;${mmss(h.t_start)}
            </p>
            <p>${esc(h.snippet)}</p>
        </li>`).join("");
    for (const b of $("search-results").querySelectorAll("button[data-id]")) {
        b.addEventListener("click", () => select(b.dataset.id));
    }
}

// --- wiring --------------------------------------------------------------

function initForms() {
    $("add-form").addEventListener("submit", async (e) => {
        e.preventDefault();
        const file = $("file-input").files[0];
        if (!file) {
            return;
        }
        const body = new FormData();
        body.append("file", file);
        body.append("mode", $("mode-input").value);
        body.append("template", $("template-input").value);
        if ($("title-input").value.trim()) {
            body.append("title", $("title-input").value.trim());
        }
        $("add-error").hidden = true;
        $("add-submit").disabled = true;
        $("add-submit").textContent = "Uploading…";
        try {
            const res = await api("/recordings", { method: "POST", body });
            $("add-form").reset();
            await refreshList();
            await select(res.id);
        } catch (err) {
            $("add-error").hidden = false;
            $("add-error").textContent = err.message;
        } finally {
            $("add-submit").disabled = false;
            $("add-submit").textContent = "Transcribe";
        }
    });

    $("search-form").addEventListener("submit", (e) => {
        e.preventDefault();
        const q = $("search-input").value.trim();
        if (q) {
            runSearch(q);
        }
    });

    $("search-clear").addEventListener("click", () => {
        $("search-input").value = "";
        show(selected ? "recording-view" : "empty-state");
    });

    $("transcript-toggle").addEventListener("click", () => {
        transcriptOpen = !transcriptOpen;
        $("transcript").hidden = !transcriptOpen;
        $("transcript-toggle").textContent = transcriptOpen ? "hide" : "show";
    });

    $("delete-recording").addEventListener("click", async () => {
        if (!selected || !confirm("Remove this recording, its transcript and its notes?")) {
            return;
        }
        try {
            await api(`/recordings/${selected}`, { method: "DELETE" });
        } catch (err) {
            $("recording-error").hidden = false;
            $("recording-error").textContent = err.message;
            return;
        }
        selected = null;
        show("empty-state");
        await refreshList();
    });
}

(async function start() {
    initTheme();
    initForms();
    try {
        await loadConfig();
        await refreshList();
    } catch (err) {
        $("add-error").hidden = false;
        $("add-error").textContent = `Cannot reach the server: ${err.message}`;
    }
})();
