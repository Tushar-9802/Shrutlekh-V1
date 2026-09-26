from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import librosa
import numpy as np
from shrutlekh.config import Settings

SR = 16000
WINDOW_S = 15  # 448-token decoder cap x ~2 tok/char in Devanagari: 30 s windows truncate

@dataclass
class Segment:
    start: float
    end: float
    text: str

def load_audio(file_path: Path) -> np.ndarray:
    audio, _ = librosa.load(file_path, sr=SR, mono=True) #all formats for ffmpeg
    return audio.astype(np.float32)

def audio_duration(path: Path) -> float:
    return float(librosa.get_duration(path=path))


def transcribe(path: Path, settings: Settings, mode: str = "hindi") -> list[Segment]:
    audio = load_audio(path)
    if settings.tier == "gpu":
        return _transcribe_trelis(audio, settings, mode)
    return _transcribe_ct2(audio, settings)

#-----------------CPU-----------------


@lru_cache(maxsize=1)
def _ct2_model(model_dir: str, threads: int):
    from faster_whisper import WhisperModel
    return WhisperModel(model_dir, device="cpu", compute_type="int8", cpu_threads=threads)


def _transcribe_ct2(audio: np.ndarray, s: Settings) -> list[Segment]:
    model = _ct2_model(s.asr_model, s.cpu_threads)
    raw, _ = model.transcribe(
        audio, language="hi", temperature=0.0, beam_size=1,
        condition_on_previous_text=False, chunk_length=int(WINDOW_S),
    )
    return [Segment(seg.start, seg.end, seg.text.strip()) for seg in raw if seg.text.strip()]

#-----------------GPU-----------------

@lru_cache(maxsize=1)
def _trelis(repo: str):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    processor = WhisperProcessor.from_pretrained(repo)
    model = WhisperForConditionalGeneration.from_pretrained(repo, dtype=torch.bfloat16).to("cuda").eval()
    return processor, model

def _prefix(tok, mode: str) -> list[int]:
    ids = tok.convert_tokens_to_ids
    p = [ids("<|startoftranscript|>"), ids("<|hi|>")]
    if mode == "hinglish":
        p += tok("<|mixedcode|>", add_special_tokens=False).input_ids  # id 51866; pipeline() cannot emit it
    p += [ids("<|transcribe|>"), ids("<|notimestamps|>")]
    return p

def _transcribe_trelis(audio: np.ndarray, s: Settings, mode: str) -> list[Segment]:
    import torch
    proc, model = _trelis(s.asr_model)
    prefix = torch.tensor([_prefix(proc.tokenizer, mode)], device="cuda")
    step = int(WINDOW_S * SR)
    out: list[Segment] = []
    for start in range(0, len(audio), step):
        chunk = audio[start:start + step]
        feat = proc.feature_extractor(chunk, sampling_rate=SR, return_tensors="pt").input_features
        with torch.inference_mode():
            ids = model.generate(input_features=feat.to("cuda", torch.bfloat16), decoder_input_ids=prefix,
                                 max_new_tokens=440, num_beams=1, do_sample=False)
        text = proc.tokenizer.decode(ids[0], skip_special_tokens=True).strip()
        if text:
            out.append(Segment(start / SR, min(start + step, len(audio)) / SR, text))
    return out