import re

import ollama
from llmclean import clean_text, load_json

from shrutlekh.asr import Segment
from shrutlekh.config import Settings

TEMPLATES = {
    "meeting": ("Summarize this meeting transcript in 5-10 bullet points: what was discussed, any decisions "
                "reached, any questions left open. Cover only the categories the transcript actually contains."),
    "lecture": "Summarize this lecture transcript as study notes: key concepts, definitions, examples. Use headings.",
    "brief": ("Summarize this transcript in 3-4 sentences: what was discussed and the main points made. "
              "Do not quote or restate the transcript; condense it."),
}

# The model invents to fill a template's categories: a meeting summary grew a funding decision
# the transcript never mentioned. State the grounding rule separately from the format.
_GROUND_RULE = (
    "Use only what this transcript says. Never introduce a topic, decision, number or name that is not in it. "
    "If the transcript has nothing for part of the format, leave that part out instead of inventing one."
)

# Script is decided in code, not by the model: told to "keep the mix", a 4B model picks one side.
# Telling it only what to do is not enough either — it needs the wrong answer shown too.
_LANG_RULES = {
    "hindi": "Write in Hindi, in Devanagari script. Do not translate to English. No preamble.",
    "english": "Write in English. No preamble.",
    "hinglish": (
        "Write Hinglish exactly as the transcript spells it: Hindi words in Devanagari script, English words "
        "in Latin script. Never transliterate Hindi into Latin letters. "
        "Correct: 'Director ने story अच्छे से नहीं बताई, film boring थी।' "
        "Wrong: 'Director ne story achche se nahi bataai, film boring thi.' "
        "No preamble."
    ),
}
_DEVA = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[A-Za-z]")


def detect_language(text: str) -> str:
    deva, latin = len(_DEVA.findall(text)), len(_LATIN.findall(text))
    if deva + latin == 0:
        return "hindi"
    share = latin / (deva + latin)
    return "english" if share > 0.85 else "hindi" if share < 0.15 else "hinglish"
_ACTIONS_PROMPT = (
    "From this transcript, list action items as JSON: "
    '{"items":[{"what": str, "who": str|null, "when": str|null}]}. '
    'Only include commitments actually stated. If there are none, return {"items":[]}.'
)


def _chat(s: Settings, system: str, user: str, json_mode: bool = False) -> str:
    r = ollama.chat(
        model=s.ollama_model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        think=False,  # answer lands in .content, never in .thinking
        options={"temperature": 0},
        keep_alive=s.keep_alive,
        format="json" if json_mode else "",
    )
    return r.message.content or ""


def transcript_text(segments: list[Segment]) -> str:
    return "\n".join(s.text for s in segments)


def summarize(segments: list[Segment], s: Settings, template: str = "meeting") -> str:
    text = transcript_text(segments)
    system = f"{TEMPLATES[template]}\n\n{_GROUND_RULE}\n\n{_LANG_RULES[detect_language(text)]}"
    return clean_text(_chat(s, system, text), repetition=True)


def action_items(segments: list[Segment], s: Settings) -> list[dict]:
    raw = _chat(s, _ACTIONS_PROMPT, transcript_text(segments), json_mode=True)
    data = load_json(raw, default={}) or {}
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    kept = []
    for it in items:  # model proposes, Python decides: only well-typed items with a non-empty `what` survive
        if isinstance(it, dict) and isinstance(it.get("what"), str) and it["what"].strip():
            kept.append({
                "what": it["what"].strip(),
                "who": it["who"] if isinstance(it.get("who"), str) else None,
                "when": it["when"] if isinstance(it.get("when"), str) else None,
            })
    return kept
