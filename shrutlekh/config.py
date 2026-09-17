import os
from dataclasses import dataclass
import torch

MODELS = { 
    "cpu": "models/vaani-small-int8",
    "gpu": "Trelis/whisper-hinglish-preview",
}

@dataclass(frozen=True)
class Settings:
    tier: str
    asr_model: str
    cpu_threads: int = 4
    ollama_model: str = "gemma3n:e4b"
    keep_alive: str = os.environ.get("OLLAMA_KEEP_ALIVE", "10m")  # call-site value overrides the daemon's

def load(tier: str | None = None) -> Settings:
    tier = tier or ("gpu" if torch.cuda.is_available() else "cpu")
    return Settings(tier=tier, asr_model=MODELS[tier])