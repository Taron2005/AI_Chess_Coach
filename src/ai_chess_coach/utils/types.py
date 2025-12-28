from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MoveRecord:
    """One move’s evaluation and classification."""
    ply: int
    side_to_move: str                 # "white" or "black" before the move
    fen_before: str
    san_played: str
    san_best: Optional[str]
    cp_best: Optional[int]            # POV = side-to-move (mate mapped to large cp)
    cp_played: Optional[int]          # "
    delta_cp: Optional[int]           # cp_best - cp_played (>=0 means worse than best)
    label: Optional[str]              # "inaccuracy" | "mistake" | "blunder" | None
    phase_label: str                  # "opening" | "middlegame" | "endgame"
    phase_score: float                # 1.0 opening … 0.0 endgame
    tags: List[str] = field(default_factory=list)


@dataclass
class GameSummary:
    """Aggregated info for a single PGN game."""
    headers: Dict[str, Any]
    white_acpl: float
    black_acpl: float
    inaccuracies: Dict[str, int]
    mistakes: Dict[str, int]
    blunders: Dict[str, int]
    phase_errors: Dict[str, Dict[str, int]]
    key_positions: List[MoveRecord]
    opening: Dict[str, Any]
    coach_message: str


@dataclass
class AggregatedStats:
    """Totals across all games in one run."""
    games: int
    total_by_label: Dict[str, int]
    total_by_phase: Dict[str, Dict[str, int]]
    openings_count: Dict[str, int]
    weakness_tags_count: Dict[str, int]


@dataclass
class StudyPlan:
    """
    Study plan derived from weaknesses; tasks are mapped from tags.
    Optionally includes LLM-generated narrative (llm_text) and a source marker.
    """
    days: List[Dict[str, Any]]
    llm_text: Optional[str] = None
    llm_source: Optional[str] = None  # e.g., "groq" if llm_text came from Groq callback


@dataclass
class CharlieReport:
    """Top-level container returned to the UI."""
    games: List[GameSummary]
    aggregated: AggregatedStats
    study_plan: StudyPlan
