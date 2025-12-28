# src/ai_chess_coach/charlie.py
"""
Public API facade for Charlie.

Streamlit (app/streamlit_app.py) imports from here, so this module re-exports:
- run_pipeline
- emit_markdown / emit_key_positions_csv
- dataclass_to_jsonable
- CharlieReport
- make_engine / build_limit
"""

from __future__ import annotations

from .analysis.core import make_engine, build_limit
from .coach.pipeline import run_pipeline
from .storage.emit import emit_json, emit_markdown, emit_key_positions_csv
from .utils.serialize import dataclass_to_jsonable
from .utils.types import MoveRecord, GameSummary, AggregatedStats, StudyPlan, CharlieReport

__all__ = [
    "make_engine",
    "build_limit",
    "run_pipeline",
    "emit_json",
    "emit_markdown",
    "emit_key_positions_csv",
    "dataclass_to_jsonable",
    "MoveRecord",
    "GameSummary",
    "AggregatedStats",
    "StudyPlan",
    "CharlieReport",
]
