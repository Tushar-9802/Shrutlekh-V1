from hindi_normalize import normalize_transcript

from shrutlekh.asr import Segment


def normalize(segments: list[Segment]) -> list[Segment]:
    return [Segment(s.start, s.end, normalize_transcript(s.text)) for s in segments]
