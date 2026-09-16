"""Offline measurement tables for Figures 7–14."""
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data" / "appendix_early"


def load_data(name):
    """Read numeric inputs; missing inputs are errors, never a network fallback."""
    return json.loads((DATA / f"{name}.json").read_text())
