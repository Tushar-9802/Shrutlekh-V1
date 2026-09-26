# Shrutlekh

श्रुतलेख — *what is heard, written down.*

Offline voice-to-notes for Hindi and Hinglish. Drop in a recording; get a transcript, a summary
in the language that was actually spoken, and the action items that were actually committed to.
Nothing leaves the machine — the whole archive is one SQLite file.

## Why

The local notes space is crowded, and the tools in it share a blind spot: they run vanilla
Whisper with automatic language detection. Measured on real audio, here is what that costs.

**Hindi** (FLEURS hi_in, 3 clips, WER after normalisation, lower is better):

| model | params | WER |
|---|---|---|
| Vaani whisper-large-v3 | 1.5B | **0.187** |
| Vaani whisper-medium | 769M | 0.199 |
| Vaani whisper-small | 244M | 0.217 |
| OpenAI whisper-large-v3 | 1.5B | 0.296 |

An Indic-tuned 244M model reads Hindi better than a 1.5B general one. Holding the architecture
fixed (Vaani large-v3 against vanilla large-v3) isolates the fine-tune at a 37% relative
reduction.

**Hinglish** (90s of real code-switched podcast speech, against a hand-corrected mixed-script
reference):

| model | WER | CER |
|---|---|---|
| Trelis whisper-hinglish-preview, `<\|mixedcode\|>` | **0.164** | 0.109 |
| Srota qwen3-asr-0.6b-hinglish | 0.199 | 0.151 |
| Trelis, token off | 0.256 | 0.225 |
| Vaani whisper-large-v3 | 0.569 | 0.534 |
| OpenAI whisper-large-v3 | 0.911 | 0.703 |

Vanilla Whisper gets 91% of the words wrong on Indian code-switched speech. The two failures are
not the same kind: Vaani's errors are mostly script — English words spelled out in Devanagari,
meaning intact — while vanilla Whisper's are comprehension.

## Tiers

Chosen by hardware at startup, override with `--tier`.

| tier | engine | Hindi WER | CPU RTF (4 threads) |
|---|---|---|---|
| CPU | Vaani small, int8 CTranslate2 | 0.257 | 0.12–0.18 |
| GPU | Trelis whisper-hinglish-preview | 0.194 | — |

int8 costs about four WER points against fp32 and buys roughly 5× the speed. A 90-second
recording transcribes in about 15 seconds on four CPU threads.

## Pipeline

```
audio ─→ ASR ─→ hindi-normalize ─→ gemma3n ─→ llmclean ─→ SQLite
         │                         │
         │                         └─ summary + action items
         └─ 15s windows, greedy, language pinned
```

- **ASR** — [faster-whisper](https://github.com/SYSTRAN/faster-whisper) on the CPU tier,
  transformers on the GPU tier. Decoding is pinned (`temperature=0`, greedy, no conditioning on
  previous text) so the same audio always gives the same transcript.
- **[hindi-normalize](https://pypi.org/project/hindi-normalize/)** — Devanagari normalisation,
  number words to digits.
- **gemma3n via [Ollama](https://ollama.com)** — summary and action items, local.
- **[llmclean](https://pypi.org/project/llmclean/)** — strips model-output artefacts.
- Action items are extracted as JSON and then validated in Python; anything malformed is dropped
  rather than guessed at.

The uploaded file is never modified. A copy is what gets processed, so re-transcribing with a
better model later stays possible.

## Running it

Requires [uv](https://docs.astral.sh/uv/), [ffmpeg](https://ffmpeg.org), and
[Ollama](https://ollama.com) with `gemma3n:e4b` pulled.

```
uv sync
uv run uvicorn shrutlekh.main:app --host 127.0.0.1 --port 8000
```

Then open <http://127.0.0.1:8000>.

There is a CLI for the same pipeline:

```
uv run python -m scripts.cli add meeting.m4a --mode hinglish --template meeting
uv run python -m scripts.cli list
uv run python -m scripts.cli search "budget"
```

`--mode hinglish` switches the GPU engine into mixed-script output. It is a per-recording choice:
on pure Hindi audio the Devanagari-only mode scores better.

## Reproducing the numbers

```
uv run python scripts/benchmarks/run_grid.py        # models x clips x devices
uv run python scripts/benchmarks/bench_int8_ct2.py  # int8 CPU rows
uv run python scripts/benchmarks/score_wer.py       # WER/CER against references
```

Rows land in `results/*.jsonl`. Scoring normalises both sides identically and strips punctuation,
since the Vaani models emit none.

## What is not here yet

- The Hinglish figure rests on **one** 90-second clip. It is a demonstration, not a benchmark.
- CPU timings come from a Ryzen 7 7800X3D capped to 4 threads, which flatters a budget laptop.
  Expect 1.5–2.5× slower on the hardware this is aimed at.
- Live capture, diarization, and languages beyond Hindi are not implemented.
- Trelis whisper-hinglish-preview is a research preview; its weights may change.
- Srota reaches 0.199 at 0.6B but drops leading audio on some clips, so it is not wired in.

## Built on

[AI4Bharat](https://ai4bharat.org) and [ARTPARK-IISc](https://huggingface.co/ARTPARK-IISc) for the
Vaani Whisper models, [Trelis](https://huggingface.co/Trelis/whisper-hinglish-preview) for the
code-mix fine-tune, and Google's Gemma for the summaries.

## Licence

MIT.
