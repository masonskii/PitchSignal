from dataclasses import dataclass


SAMPLE = 8000


@dataclass
class Pitching:
    pitch: float
    range: tuple[int, int]
    amplitude: float
    sampling: float = 1 / SAMPLE


@dataclass
class Autcorr:
    lag: float
    peak: float
    range: tuple[int, int]
    SFS: int
    FS: int
