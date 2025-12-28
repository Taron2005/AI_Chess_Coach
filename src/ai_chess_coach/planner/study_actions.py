from __future__ import annotations

import os
from typing import Dict

try:
    import yaml  # optional dependency
except Exception:
    yaml = None


STUDY_ACTIONS: Dict[str, str] = {
    # Tactics
    "tactics:forks": "Solve 10 fork puzzles",
    "tactics:pins": "Practice pin and skewer drills (10 puzzles)",
    "tactics:hanging": "Do 10 'hanging piece' puzzles (avoid undefended pieces)",
    "tactics:discovered": "Study discovered attacks (short video + 10 puzzles)",
    "tactics:missed_capture": "Tactics on winning captures (10 puzzles)",
    "tactics:missed_check": "Tactics on forcing moves (checks) (10 puzzles)",
    "tactics:self_pin": "Drills: avoid self-pins and alignments (8 puzzles)",
    "tactics:en_prise": "Drills: stop leaving pieces en prise (10 puzzles)",
    "tactics:back_rank": "Back-rank checkmate patterns (10 puzzles)",
    "tactics:blunder_mate": "Spot mate threats / avoid getting mated (10 puzzles)",

    # Openings
    "opening:Sicilian": "Watch Sicilian basics (placeholder) & annotate 1 model game",
    "opening:Caro-Kann": "Review Caro-Kann plans & annotate 1 model game",
    "opening:French": "Review French Defense plans & annotate 1 model game",

    # Phases
    "phase:opening": "Opening fundamentals (development, center, king safety)",
    "phase:middlegame": "Middlegame plans (pawn structure, weak squares)",
    "phase:endgame": "Rook endgames: Lucena/Philidor, technique",

    # Other
    "king_safety": "King-safety drills (create luft, recognize open-file dangers)",

    # Habits
    "warmup:puzzles": "5-minute warm-up: 3 easy tactics",
    "review:game": "Review 1 of your games and note 3 takeaways",
}


def maybe_override_actions(config_dir: str = "configs", filename: str = "study_map.yaml") -> None:
    """
    If configs/study_map.yaml exists, override default STUDY_ACTIONS
    with user-provided mappings.
    """
    path = os.path.join(config_dir, filename)
    if not yaml or not os.path.exists(path):
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        if isinstance(d, dict):
            for k, v in d.items():
                STUDY_ACTIONS[str(k)] = str(v)
    except Exception:
        # silent fail
        return
