# Shrutlekh — working plan (2026-08-24)

Successor to the original blueprint PDF; folds in the 2026-08-24 bake-off results and the
three-stream research (Meetily teardown, Indic ASR landscape, competitor survey).

## Strategic frame

The gap — offline Hinglish/code-mix voice notes — is open. Estimated window: 6–12 months
(Meetily adding Indic ASR backends, or Sarvam Edge going public, closes it). The defensible
position is not "Meetily but Hindi"; it is the Indic-specific hard parts: code-mix ASR,
mixed-script output, summaries in the meeting's language, Indian meeting idioms.
Meetily's non-English issue cluster (#150, #370, #581, #584) doubles as our feature list.

## Bake-off verdict (measured, 2026-08-24)

| model | WER (FLEURS avg) | CPU RTF (4 thr) | role |
|---|---|---|---|
| vaani-small (244M) | 0.217 | 0.36–0.89 | low tier |
| vaani-medium (769M) | 0.199 | 0.98–2.23 | mid tier |
| vaani-large-v3 (1.5B) | 0.187 | 1.8–4.2 | GPU tier |
| vanilla large-v3 (1.5B) | 0.296 | 1.8–3.3 | baseline only |

vaani-small beats vanilla large-v3 at 1/6 the size; Indic fine-tuning = 37% relative WER cut
at equal architecture. Greedy decode fully deterministic; CPU/GPU transcripts identical.
Full rows: `results/asr_bench.jsonl`. Harness: `scripts/benchmarks/bench_one.py` + `run_grid.py`.

**Hinglish (2026-08-31, hand-corrected mixed-script reference, 90s real podcast audio):**

| model | WER | CER |
|---|---|---|
| vaani-large-v3 | 0.569 | 0.534 |
| vaani-medium | 0.651 | 0.588 |
| vaani-small-int8 (shipping CPU config) | 0.687 | 0.625 |
| vanilla whisper-large-v3 | 0.911 | 0.703 |

Every available model fails code-mixed speech. Vaani's errors are script (English words rendered
in Devanagari, meaning mostly intact); vanilla Whisper's are comprehension. This is the number
v0.2 exists to move. Scorer: `scripts/benchmarks/score_wer.py` → `results/wer.jsonl`.

**Code-mix models (2026-09-14, same reference, cuda):**

| model | Hinglish WER | FLEURS mean | notes |
|---|---|---|---|
| Trelis whisper-hinglish-preview, `<|mixedcode|>` on | **0.164** | 0.272 | FLEURS penalty is script convention, not quality |
| Trelis, token off | 0.256 | 0.194 | Hindi quality ≈ vaani-large-v3 (0.187) |
| Srota qwen3-asr-0.6b-hinglish | 0.199 | 0.364 | drops leading audio (teentaal −10 s, fleurs_1842 −40%) |
| vaani-large-v3 (previous best) | 0.569 | 0.187 | |

Trelis is the v0.2 GPU-tier engine: Vaani-level Hindi plus a 4.2× Hinglish WER cut over the
shipping config. The token is a per-recording mode (Hindi vs Hinglish) — the app needs a toggle
or a first-pass script-ratio auto-detect. Srota's deletions disqualify it until the cause is
found; it stays the 0.6B CPU-tier candidate on paper. Trelis on CPU = CT2-int8 conversion with
a generated `tokenizer.json` and a patched `sot_sequence` (unverified). Reference set is n=1 for
Hinglish — grow it before the launch claim. Pin the Trelis weight hash (research preview).

## Phase 0 — extend the bake-off (this week)

- [ ] Commit benchmark work.
- [ ] CT2-int8 convert vaani-small + vaani-medium (`ct2-transformers-converter --quantization int8`);
      re-run CPU rows via faster-whisper. Target: small ≤ ~0.25 RTF @ 4 threads.
- [ ] Hand-correct the two podcast references (mixed-script convention for the Hinglish clip —
      it becomes the v0.2 acceptance format).
- [ ] Add Hinglish shortlist to the grid (both Apache-2.0, both native mixed-script output):
      `Trelis/whisper-hinglish-preview` (large-v3 FT on the Vaani checkpoint, `<|mixedcode|>` token;
      research preview → pin weights + hash) and `moorlee/qwen3-asr-0.6b-hinglish` "Srota"
      (Qwen3-ASR FT; ggml-org GGUF path for CPU).
- [ ] Benchmark `trysem/indicconformer-120m-onnx` (plain onnxruntime, CC-BY-4.0) as low-tier
      dark horse; earns a slot only if it beats int8-vaani-small on our clips.
- Skip: NeMo-fork .nemo checkpoints on Windows; `atharva-again` 600M int8 port (WER doubled);
  Sarvam (no open ASR weights).

## Phase 1 — v0.1 spine (1–2 weeks, quiet GitHub ship)

Upload → tiered ASR (int8-vaani-small CPU / vaani-medium+ GPU) → hindi-normalize →
gemma3n:e4b (meeting/lecture/brief templates, MedScribe patterns) → llmclean → SQLite →
Streamlit list/keyword-search → markdown export.

- Summaries in the transcript's language (Hindi in → Hindi summary) — Meetily's documented gap.
- Keep the original upload untouched on disk; transcribe from a processed copy
  (Meetily's two-path idea → re-transcribe-later comes free).
- Eval harness re-scores the 5-clip set on every pipeline change.
- README leads with the bake-off table. No launch thread.

## Phase 1.5 — RAG query over past notes ("what was discussed where")

Natural-language questions across the meeting archive: "SSD wale project ka budget kis meeting
mein discuss hua tha?" → answer + the meetings it came from. Scope verdict: fits local-first
exactly (every component already local); depends on the v0.1 SQLite store existing; NOT
launch-gating — retention feature, not acquisition. Est. 1–2 days for hybrid search,
3–5 days for grounded answering + eval.

Design:
- **Store**: `sqlite-vec` virtual table + FTS5 table in the SAME SQLite db as the notes.
  Zero new infrastructure; ships as part of the app db file.
- **Chunking**: whisper segments merged to ~200–400-token windows (1-segment overlap),
  each row: meeting_id, ts_start, ts_end, text, embedding. Per-meeting summary embedded too
  (coarse retrieval level).
- **Embedder**: `intfloat/multilingual-e5-small` (118M, 384-dim, Hindi-capable, query:/passage:
  prefixes) — CPU-tier friendly; `bge-m3` optional on GPU tier. Hinglish embedding quality is
  the open risk → eval before committing (see below).
- **Retrieval**: hybrid — FTS5 keyword + vector cosine, RRF fusion, top-k≈8, dedupe by meeting.
  Hybrid because Devanagari morphology weakens pure keyword search and code-mix weakens
  pure vectors; each covers the other.
- **Answering**: gemma3n grounded prompt over retrieved chunks; answer MUST cite meeting
  title/date; if max similarity < threshold → explicit "not found in your notes" (no
  default-fallthrough — the router lesson from the KB). llmclean on output.
- **Eval**: 15–20 seed queries with known correct meetings; metrics: correct-meeting@k and
  groundedness (does every claim trace to a retrieved chunk). Extends the existing harness.
- **UI**: one query box in Streamlit above the notes list; answer card + links to source meetings.

Risks: multilingual embedding quality on Hinglish (measure first — if e5-small fails on
code-mix queries, try bge-m3 small-batch or transliterate-to-single-script before embedding);
sqlite-vec on Windows (ships wheels — verify early); scale is trivial (personal archives =
hundreds of meetings, brute-force cosine would honestly suffice).

### Marginalia port (2026-09-14) — Phase 1.5 is mostly already written

Marginalia (Fic Search MVP, separate private repo) is a working hybrid-RAG stack over long
documents: paragraph-aware chunker → sqlite-vec dense + rank-bm25 lexical → RRF → ordinal /
structural query routing → neighbour expansion → grounded answer with citation validation and
cited-first source panel; FastAPI + vanilla-JS shell, background indexer with status polling,
tested to ~680K words. It is the Phase 1.5 design, built. The fanfic community's 2026 AI
climate (mass AI-accusation doc, harassment waves, AO3 no-policy) makes a public fic-tool launch
a poor bet; the stack moves here, the fic repo stays private for personal use.

Reuse map (chapters → meetings, paragraphs → timestamped ASR segments):

| Marginalia module | Shrutlekh use | change |
|---|---|---|
| `ingest/chunk.py` | segment-aware chunker | paragraphs = whisper segments; keep 300–500 tok target |
| `search/retrieve.py` | hybrid retrieval | `\w+` tokenizer already handles Devanagari; ordinal cues need Hindi/Hinglish forms (पहले, पहली बार, kab, sabse pehle); structural routing → "last meeting" / date / title |
| `search/answer.py` | grounded answer + citation validation | OpenAI → Ollama gemma3n; citation label → `[<meeting>, mm:ss–mm:ss]`; keep the strict-constraint prompt and hallucinated-citation strip |
| `search/pipeline.py` | query orchestration | drop budget caps; keep query log |
| `ingest/embed.py` | embeddings | OpenAI 1536-d → local `multilingual-e5-small` 384-d (`vec0 FLOAT[384]`), `query:`/`passage:` prefixes |
| `db.py` + `workers/indexer.py` | store + background jobs | works→meetings, chapters→recordings, chunks keyed by (meeting, t_start, t_end) |
| `ingest/entities.py`, `search/entity_facts.py` | people/projects/decisions index across meetings | later; strong fit for "what did X commit to" |
| `main.py`, `static/*` | app shell | candidate replacement for the Streamlit MVP (see decision below) |
| `ingest/fetch.py`, `ingest/parse.py`, `mail.py`, `routes/suggestions.py` | — | not ported (AO3/epub/boilerplate/email) |

Port rules: copy files, never merge repos (Marginalia is under a separate identity — its
history stays there); scan every ported file for fic-derived strings before it lands here;
Hinglish embedding quality is measured before the embedder is committed.

**Open decision — app shell.** Streamlit was picked for speed. Marginalia's FastAPI +
vanilla-JS shell already exists, is tested, and is what a Tauri build wraps later; Streamlit
would be thrown away at that point. Adopting the shell now means v0.1 replaces the works
ingestion with audio ingestion (upload → ASR → normalize → summarize) inside a proven app,
and Phase 1.5 comes along for free. Cost: a few days more than a Streamlit page.

## Phase 2 — v0.2 Hinglish (the loud launch)

Integrate the Phase-0 code-mix winner; acceptance test = hand-corrected Hinglish reference.
Mixed-script normalization likely lands in hindi-normalize (library credibility compounding).
Launch on r/LocalLLaMA: measured claims, benchmarks from CPU-class hardware.
IndicXlit transliteration = fallback path only.

## Phase 3 — v0.3+ (after v0.2 traction)

Mic-only live capture (sounddevice + RNNoise→loudness-normalize→Silero-VAD chain);
WASAPI loopback deferred to Tauri phase (half of Meetily's bug tracker). Diarization = Meetily's
paid feature; shipping it free later is a strategic option. sherpa-onnx = candidate unified
runtime if conformer-CTC wins the low tier. Tauri packaging last.

## Throughout

Extract `indic-asr-pipeline` once the pipeline is used twice. Eval harness doubles as the
publishable artifact (per-stage attribution across Sakhi/MedScribe/Shrutlekh).

## Execution shape (from the Gemma 4 Good winners, 2026-08-31)

What placed: narrow scope, a fine-tune on a purpose-built dataset, one hard metric up front,
finished build with on-device benchmarks. Applied here:
- The one number: code-mix WER on real Hinglish audio, measured on CPU-class hardware.
  README and launch lead with it; the feature list comes after.
- Fine-tune only if Trelis/Srota fail the Hinglish acceptance set — a base model that wins the
  A/B ships; the hand-corrected mixed-script references double as the seed dataset if a
  fine-tune is needed.
- Sakhi-v2 (clinical Hindi ASR, entity-WER) extends this harness — the second use that
  triggers the `indic-asr-pipeline` extraction. Shared contract until then: the
  `results/*.jsonl` row schema and the normalize-then-jiwer scoring path.
