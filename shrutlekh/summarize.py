import ollama
from llmclean import clean_text, load_json

from shrutlekh.asr import Segment
from shrutlekh.config import Settings

TEMPLATES = {
    "meeting": "Summarize this meeting transcript: decisions made, topics discussed, open questions. 5-10 bullet points.",
    "lecture": "Summarize this lecture transcript as study notes: key concepts, definitions, examples. Use headings.",
    "brief": "Summarize this transcript in one short paragraph.",
}
_LANG_RULE = ("Write in the same language and script mix as the transcript itself "
              "(Hindi stays Hindi, Hinglish stays Hinglish). Do not translate.")
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
    raw = _chat(s, f"{TEMPLATES[template]} {_LANG_RULE}", transcript_text(segments))
    return clean_text(raw, repetition=True)


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
