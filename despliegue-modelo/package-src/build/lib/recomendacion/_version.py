from pathlib import Path

with open(Path(__file__).resolve().parent / "VERSION") as _f:
    __version__ = _f.read().strip()
