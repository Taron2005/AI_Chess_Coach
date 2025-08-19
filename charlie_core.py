#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
charlie_core.py — Core analysis engine for Charlie (AI Chess Coach)

Responsibilities:
- Parse PGNs and iterate through real moves.
- For each move, use Stockfish to:
  * Evaluate best lines (MultiPV) and the played move.
  * Compute eval loss Δcp = cp_best − cp_played.
  * Label the move as inaccuracy/mistake/blunder using dynamic thresholds that
    consider MultiPV gap, advantage size, and game phase; also handle mates.
  * Assign lightweight tactical tags (en-prise, self-pin, missed forcing, back-rank risk, forks, king safety).
- Aggregate results across games.
- Build a study plan (heuristic), and if an LLM callback is provided, ask it to
  produce an additional human-friendly plan (marked in the UI as fully LLM-generated).
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Any, Iterable, Callable

import chess
import chess.pgn
import chess.engine

try:
    import yaml  # optional mapping override for study actions
except Exception:
    yaml = None

# On Windows, certain Python builds need the Proactor loop policy for subprocesses
if os.name == "nt":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass


# =========================
# Config & thresholds
# =========================

# Base centipawn thresholds; final decision uses dynamic scaling by context
BASE_THRESHOLDS_CP = {
    "inaccuracy": 100,
    "mistake": 250,
    "blunder": 500,
}

# Tapered-material weights used to compute a phase metric (opening→endgame)
PHASE_WEIGHTS = {
    chess.PAWN: 0,    # pawns usually excluded from phase computation
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
}
PHASE_MAX = 2 * (2*PHASE_WEIGHTS[chess.KNIGHT] + 2*PHASE_WEIGHTS[chess.BISHOP]
                 + 2*PHASE_WEIGHTS[chess.ROOK] + 1*PHASE_WEIGHTS[chess.QUEEN])
# = 2*(2+2+4+4) = 24

DEFAULT_DEPTH = 14
DEFAULT_NODES = None
DEFAULT_MOVE_TIME = None  # seconds


# A small catalog of study actions tied to tags (can be overridden by YAML)
STUDY_ACTIONS = {
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

    # Warmups / reflection
    "warmup:puzzles": "5-minute warm-up: 3 easy tactics",
    "review:game": "Review 1 of your games and note 3 takeaways",
}

def _maybe_override_actions():
    """
    If configs/study_map.yaml exists, override the default STUDY_ACTIONS
    with user-provided mappings. This lets users customize tasks without code edits.
    """
    path = os.path.join("configs", "study_map.yaml")
    if yaml and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = yaml.safe_load(f) or {}
            if isinstance(d, dict):
                for k, v in d.items():
                    STUDY_ACTIONS[str(k)] = str(v)
        except Exception:
            # Silent fail: use default STUDY_ACTIONS
            pass

_maybe_override_actions()


# =========================
# Data classes (report schema)
# =========================

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


# =========================
# Engine utilities
# =========================

def make_engine(stockfish_path: str) -> chess.engine.SimpleEngine:
    """
    Launch the Stockfish binary via UCI and return the engine handle.
    Raises FileNotFoundError when the path is bad.
    """
    if not os.path.exists(stockfish_path):
        raise FileNotFoundError(f"Stockfish not found at: {stockfish_path}")
    try:
        return chess.engine.SimpleEngine.popen_uci(stockfish_path)
    except NotImplementedError:
        # Some Windows envs need this event loop policy for subprocess support
        if os.name == "nt":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            return chess.engine.SimpleEngine.popen_uci(stockfish_path)
        raise


def build_limit(depth: Optional[int] = None, nodes: Optional[int] = None, movetime: Optional[float] = None) -> chess.engine.Limit:
    """
    Build a chess.engine.Limit from the mutually exclusive inputs.
    Priority: nodes -> movetime -> depth -> DEFAULT_DEPTH
    """
    if nodes is not None:
        return chess.engine.Limit(nodes=nodes)
    if movetime is not None:
        return chess.engine.Limit(time=movetime)
    return chess.engine.Limit(depth=(depth if depth is not None else DEFAULT_DEPTH))


def _score_cp_and_mate(info: chess.engine.InfoDict, pov: chess.Color, mate_cp: int = 100000) -> Tuple[Optional[int], Optional[int]]:
    """
    Convert an engine score (which can be mate or centipawns) into:
    - cp: centipawns, mapping mate to a large "cp" value via `mate_score=mate_cp`
    - mate: mate distance (plys to mate) or None
    """
    if "score" not in info or info["score"] is None:
        return None, None
    s = info["score"].pov(pov)
    cp = s.score(mate_score=mate_cp)
    mate = s.mate()
    return cp, mate


def analyze_multipv(engine: chess.engine.SimpleEngine, board: chess.Board, limit: chess.engine.Limit, n: int = 2):
    """
    Query the engine for up to `n` principal variations and return a list
    sorted by MultiPV rank (best first).
    """
    if n <= 1:
        res = engine.analyse(board, limit)
        return [res]
    res = engine.analyse(board, limit, multipv=n)
    if isinstance(res, list):
        return sorted(res, key=lambda d: int(d.get("multipv", 1)))
    else:
        return [res]


# =========================
# Phase computation
# =========================

def phase_score(board: chess.Board) -> float:
    """
    Compute a normalized phase score in [0, 1]: 1≈opening, 0≈endgame.
    Uses a tapered-material approach (counts major/minor pieces on board).
    """
    phase = 0
    for pt, w in PHASE_WEIGHTS.items():
        if w == 0:
            continue
        phase += (len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK))) * w
    return max(0.0, min(1.0, phase / PHASE_MAX))


def phase_label(phase_value: float) -> str:
    """Map the numeric phase to a coarse label."""
    if phase_value >= 0.66:
        return "opening"
    if phase_value <= 0.25:
        return "endgame"
    return "middlegame"


# =========================
# Tactic tagging helpers
# =========================

PIECE_VALUE = {chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}

def _is_self_pinned(board: chess.Board, color: chess.Color, square: chess.Square) -> bool:
    """True if the piece on `square` is pinned to our king (python-chess helper)."""
    return board.is_pinned(color, square)


def _en_prise_after(board_after: chess.Board, color: chess.Color) -> bool:
    """
    Detects if any of our non-pawn pieces are hanging after the move:
    crude attackers vs defenders count.
    """
    for pt in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
        for sq in board_after.pieces(pt, color):
            defenders = sum(1 for _ in board_after.attackers(color, sq))
            attackers = sum(1 for _ in board_after.attackers(not color, sq))
            if attackers > defenders:
                return True
    return False


def classify_tactic_theme(board_before: chess.Board, move: chess.Move, best_move: Optional[chess.Move], delta_cp: Optional[int]) -> List[str]:
    """
    Lightweight engine-aided tactical classification.

    Currently detects:
    - tactics:en_prise          → you left something hanging after your move
    - tactics:self_pin          → your moved piece becomes pinned
    - tactics:missed_capture    → best move was a capture and Δcp meaningful
    - tactics:missed_check      → best move gave a check/mate and Δcp meaningful
    - tactics:back_rank         → rough luft heuristic: king stuck on back rank
    - tactics:forks             → knight move creates >=2 big-target attacks
    - king_safety               → pawn moves near own king (simple hint)
    """
    tags: List[str] = []
    color = board_before.turn

    # Generate the after-move board to inspect consequences
    after = board_before.copy(stack=False)
    after.push(move)

    # En prise? (hanging piece after move)
    if _en_prise_after(after, color):
        tags.append("tactics:en_prise")

    # Self-pin? (moved piece ends up pinned)
    moved_piece = after.piece_at(move.to_square)
    if moved_piece and _is_self_pinned(after, color, move.to_square):
        tags.append("tactics:self_pin")

    # Missed forcing move (capture/check) only if eval swing is big enough
    if best_move and delta_cp is not None and delta_cp >= 120:
        bm_san = board_before.san(best_move)
        if "x" in bm_san:
            tags.append("tactics:missed_capture")
        if "+" in bm_san or "#" in bm_san:
            tags.append("tactics:missed_check")

    # Back-rank risk: crude luft test via unmoved f/g/h pawns and king on back rank
    king_sq = after.king(color)
    if king_sq is not None:
        rank = chess.square_rank(king_sq)
        back_rank = 0 if color == chess.WHITE else 7
        if rank == back_rank:
            files = [5, 6, 7]  # f, g, h files
            pawns_back = 0
            for f in files:
                sq = chess.square(f, 1 if color == chess.WHITE else 6)
                pc = after.piece_at(sq)
                if pc and pc.piece_type == chess.PAWN and pc.color == color:
                    pawns_back += 1
            if pawns_back >= 2:
                tags.append("tactics:back_rank")

    # Knight fork detector: Knight move that attacks >=2 valuable targets
    moved_from = board_before.piece_at(move.from_square)
    if moved_from and moved_from.piece_type == chess.KNIGHT:
        attacked = list(after.attacks(move.to_square))
        big_targets = 0
        for sq in attacked:
            pc = after.piece_at(sq)
            if pc and pc.color != color and pc.piece_type in (chess.QUEEN, chess.ROOK, chess.KING):
                big_targets += 1
        if big_targets >= 2:
            tags.append("tactics:forks")

    # Simple king safety nudge: pushing pawns near your king weakens cover
    if moved_from and moved_from.piece_type == chess.PAWN and king_sq is not None:
        if abs(chess.square_file(move.from_square) - chess.square_file(king_sq)) <= 1:
            tags.append("king_safety")

    return sorted(set(tags))


# =========================
# Dynamic labeling
# =========================

def _clamp(a, lo, hi):
    """Clamp a numeric value into [lo, hi]."""
    return max(lo, min(hi, a))


def dynamic_thresholds(cp_best: int, multipv_gap: int, phase_val: float) -> Tuple[int, int, int]:
    """
    Adjust inaccuracy/mistake/blunder thresholds based on context:

    - multipv_gap: if only one good move exists (big gap), be stricter.
    - cp_best (advantage): if you're winning big, throwing eval is worse (stricter).
    - phase_val: endgames demand more precision (stricter); openings are looser.
    """
    inc, mist, bl = BASE_THRESHOLDS_CP["inaccuracy"], BASE_THRESHOLDS_CP["mistake"], BASE_THRESHOLDS_CP["blunder"]

    # MultiPV gap in [0..800] → map to a scale factor ~[0.6..1.3]
    gap = _clamp(abs(multipv_gap), 0, 800)
    f_gap = _clamp(1.3 - (gap / 800.0), 0.6, 1.3)

    # Advantage magnitude in [0..2000] → map to ~[0.7..1.2]
    adv = _clamp(abs(cp_best), 0, 2000)
    f_adv = _clamp(1.2 - (adv / 2000.0), 0.7, 1.2)

    # Phase scaling: opening 1.0→~1.3 (looser), endgame 0.0→~0.7 (stricter)
    f_phase = _clamp(0.7 + 0.6 * phase_val, 0.7, 1.3)

    scale = f_gap * f_adv * f_phase
    return (int(inc * scale), int(mist * scale), int(bl * scale))


def label_from_engine(best_info: Dict[str, Any], played_cp: Optional[int], phase_val: float, second_info: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    """
    Decide label using mate awareness + dynamic thresholds.

    Returns:
        label: "inaccuracy" | "mistake" | "blunder" | None
        cp_best: the best line CP (POV = side-to-move)
        multipv_gap: |best_cp - second_cp| (0 if second is missing)
    """
    if played_cp is None or best_info is None:
        return None, None, None

    cp_best, mate_best = best_info.get("_cp"), best_info.get("_mate")
    if cp_best is None:
        return None, None, None

    delta = cp_best - played_cp

    # If best mates and our move loses it → blunder by policy
    if mate_best is not None:
        if mate_best > 0 and delta > 0:
            return "blunder", cp_best, 0

    # If our move is a near-forced mate against us (mapped to huge negative cp), call blunder
    if played_cp <= -90000:
        return "blunder", cp_best, 0

    # MultiPV gap if second PV exists
    gap = 0
    if second_info and second_info.get("_cp") is not None and cp_best is not None:
        gap = abs(cp_best - second_info["_cp"])

    # Dynamic thresholds find a context-aware boundary between inacc/mistake/blunder
    inc_thr, mist_thr, bl_thr = dynamic_thresholds(cp_best, gap, phase_val)

    if delta >= bl_thr:
        return "blunder", cp_best, gap
    if delta >= mist_thr:
        return "mistake", cp_best, gap
    if delta >= inc_thr:
        return "inaccuracy", cp_best, gap
    return None, cp_best, gap


# =========================
# Per-game analysis
# =========================

def analyze_game(engine: chess.engine.SimpleEngine, game: chess.pgn.Game, limit: chess.engine.Limit) -> GameSummary:
    """
    Analyze a single PGN with Stockfish.

    For each played move:
      - Evaluate MultiPV=2 from the current node (best + second).
      - Evaluate the played move via root_moves=[move].
      - Compute Δcp and decide label with dynamic thresholds and mate awareness.
      - Assign tactic tags from board-before/best/after context.
    """
    board = game.board()
    headers = dict(game.headers)
    opening_info = {
        "eco": headers.get("ECO"),
        "opening": headers.get("Opening"),
        "variation": headers.get("Variation"),
    }

    # ACPL components (sum of eval losses per side)
    white_deltas: List[int] = []
    black_deltas: List[int] = []

    # Counts by side and by phase
    label_counts = {
        "white": {"inaccuracy": 0, "mistake": 0, "blunder": 0},
        "black": {"inaccuracy": 0, "mistake": 0, "blunder": 0},
    }
    phase_error_counts = {
        "opening": {"inaccuracy": 0, "mistake": 0, "blunder": 0},
        "middlegame": {"inaccuracy": 0, "mistake": 0, "blunder": 0},
        "endgame": {"inaccuracy": 0, "mistake": 0, "blunder": 0},
    }

    key_positions: List[MoveRecord] = []
    ply = 0

    for move in game.mainline_moves():
        fen_before = board.fen()
        side_before = "white" if board.turn == chess.WHITE else "black"

        # Compute phase at the current node (before move)
        ph_score = phase_score(board)
        ph_label = phase_label(ph_score)

        # Ask the engine for best and second-best lines (MultiPV=2)
        infos = analyze_multipv(engine, board, limit, n=2)

        # Pack scores (centipawns) + mate into compact dicts
        def pack(info: Dict[str, Any]) -> Dict[str, Any]:
            cp, mate = _score_cp_and_mate(info, pov=board.turn)
            m = info.get("pv", [None])
            mv = m[0] if m else None
            return {"_cp": cp, "_mate": mate, "_move": mv, "raw": info}

        best = pack(infos[0]) if infos else {"_cp": None, "_mate": None, "_move": None}
        second = pack(infos[1]) if len(infos) >= 2 else None

        # Evaluate the actually played move from the root (single-PV)
        played_info = engine.analyse(board, limit, root_moves=[move])
        played_cp, played_mate = _score_cp_and_mate(played_info, pov=board.turn)

        # Produce the label using dynamic thresholds
        label, cp_best, gap = label_from_engine(best, played_cp, ph_score, second)

        # Δcp for ACPL: how much worse than best we played (>=0 by construction)
        delta = None
        if cp_best is not None and played_cp is not None:
            delta = cp_best - played_cp

        # Heuristic tactic tags (for plan building and UX hints)
        tags = classify_tactic_theme(board, move, best.get("_move"), delta)

        # Build a record for this move (include SANs and FEN)
        san_played = board.san(move)
        san_best = chess.Board(fen_before).san(best.get("_move")) if best.get("_move") else None

        key_positions.append(MoveRecord(
            ply=ply,
            side_to_move=side_before,
            fen_before=fen_before,
            san_played=san_played,
            san_best=san_best,
            cp_best=cp_best,
            cp_played=played_cp,
            delta_cp=delta,
            label=label,
            phase_label=ph_label,
            phase_score=round(ph_score, 3),
            tags=tags,
        ))

        # Aggregate for ACPL and error counts
        if delta is not None:
            if side_before == "white":
                white_deltas.append(max(0, delta))
            else:
                black_deltas.append(max(0, delta))
        if label:
            label_counts[side_before][label] += 1
            phase_error_counts[ph_label][label] += 1

        # Play the move on the board and continue
        board.push(move)
        ply += 1

    # Compute ACPL for both sides (mean of eval losses)
    white_acpl = float(sum(white_deltas) / len(white_deltas)) if white_deltas else 0.0
    black_acpl = float(sum(black_deltas) / len(black_deltas)) if black_deltas else 0.0

    # Build a short coach message that mentions totals and dominant error phase
    total_blunders = label_counts["white"]["blunder"] + label_counts["black"]["blunder"]
    total_mistakes = label_counts["white"]["mistake"] + label_counts["black"]["mistake"]
    total_inacc = label_counts["white"]["inaccuracy"] + label_counts["black"]["inaccuracy"]
    top_phase = max(phase_error_counts.items(), key=lambda kv: sum(kv[1].values()))[0] if any(
        sum(v.values()) for v in phase_error_counts.values()
    ) else "opening"
    if total_blunders or total_mistakes or total_inacc:
        coach_message = (
            f"You had {total_blunders} blunder(s), {total_mistakes} mistake(s), and {total_inacc} inaccuracy(ies). "
            f"Most issues happened in the {top_phase}."
        )
    else:
        coach_message = "Clean game—few significant errors. Nice!"

    # Return a compact game summary
    return GameSummary(
        headers=headers,
        white_acpl=round(white_acpl, 1),
        black_acpl=round(black_acpl, 1),
        inaccuracies={"white": label_counts["white"]["inaccuracy"], "black": label_counts["black"]["inaccuracy"]},
        mistakes={"white": label_counts["white"]["mistake"], "black": label_counts["black"]["mistake"]},
        blunders={"white": label_counts["white"]["blunder"], "black": label_counts["black"]["blunder"]},
        phase_errors=phase_error_counts,
        key_positions=key_positions,
        opening=opening_info,
        coach_message=coach_message,
    )


# =========================
# Aggregation & study plan
# =========================

def tally_openings(games: List[GameSummary]) -> Dict[str, int]:
    """
    Count how many times each opening (name or ECO) appears across games.
    Used for UI "Top Openings" and for optional plan hints.
    """
    counts: Dict[str, int] = {}
    for g in games:
        name = g.opening.get("opening") or g.opening.get("eco") or "Unknown"
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))


def aggregate_stats(games: List[GameSummary]) -> AggregatedStats:
    """
    Merge per-game summaries:
      - total_by_label: sum of inaccuracy/mistake/blunder across all games
      - total_by_phase: same counts split by opening/middlegame/endgame
      - openings_count: frequency of openings reached
      - weakness_tags_count: histogram from MoveRecord.tags (for plan)
    """
    total_by_label = {"inaccuracy": 0, "mistake": 0, "blunder": 0}
    total_by_phase = {
        "opening": {"inaccuracy": 0, "mistake": 0, "blunder": 0},
        "middlegame": {"inaccuracy": 0, "mistake": 0, "blunder": 0},
        "endgame": {"inaccuracy": 0, "mistake": 0, "blunder": 0},
    }
    weakness_tags: Dict[str, int] = {}

    for g in games:
        # Add side-wise counts to global totals
        inc = getattr(g, "inaccuracies", {}) or {}
        mst = getattr(g, "mistakes", {}) or {}
        bld = getattr(g, "blunders", {}) or {}
        total_by_label["inaccuracy"] += inc.get("white", 0) + inc.get("black", 0)
        total_by_label["mistake"] += mst.get("white", 0) + mst.get("black", 0)
        total_by_label["blunder"] += bld.get("white", 0) + bld.get("black", 0)

        # Add per-phase splits
        for ph in total_by_phase:
            for lab in total_by_phase[ph]:
                total_by_phase[ph][lab] += getattr(g, "phase_errors", {}).get(ph, {}).get(lab, 0)

        # Gather tactic tags from labeled positions (and tag the phase too)
        for mr in getattr(g, "key_positions", []) or []:
            if getattr(mr, "label", None):
                for t in getattr(mr, "tags", []) or []:
                    weakness_tags[t] = weakness_tags.get(t, 0) + 1
                phase_tag = f"phase:{getattr(mr, 'phase_label', 'unknown')}"
                weakness_tags[phase_tag] = weakness_tags.get(phase_tag, 0) + 1

        # Optional opening-family “hints” to bias the plan (very coarse)
        op = (getattr(g, "opening", {}) or {}).get("opening")
        if op:
            if "Sicilian" in op:
                weakness_tags["opening:Sicilian"] = weakness_tags.get("opening:Sicilian", 0) + 1
            if "Caro-Kann" in op:
                weakness_tags["opening:Caro-Kann"] = weakness_tags.get("opening:Caro-Kann", 0) + 1
            if "French" in op:
                weakness_tags["opening:French"] = weakness_tags.get("opening:French", 0) + 1

    openings_count = tally_openings(games)

    return AggregatedStats(
        games=len(games),
        total_by_label=total_by_label,
        total_by_phase=total_by_phase,
        openings_count=openings_count,
        weakness_tags_count=dict(sorted(weakness_tags.items(), key=lambda kv: kv[1], reverse=True)),
    )


def _default_reason(tag: str, counts: Dict[str, int]) -> str:
    """
    Short, human-friendly reasons attached to tasks in the plan.
    """
    n = counts.get(tag, 0)
    if tag.startswith("phase:"):
        phase = tag.split(":", 1)[1]
        return f"Frequent errors in the {phase} (count {n})."
    if tag.startswith("opening:"):
        return f"Often reached this opening family (count {n})."
    if tag.startswith("tactics:"):
        return f"Recurring tactical issue detected (count {n})."
    if tag == "king_safety":
        return f"King safety issues recurred (count {n})."
    return f"Recurring theme detected (count {n})."


def make_study_plan(
    weakness_tags_count: Dict[str, int],
    days: int = 7,
    per_day: int = 3,
    minutes_per_day: int = 45,
    include_warmups: bool = True,
    llm_callback: Optional[Callable[[str], str]] = None,   # Optional LLM hook (Groq)
    games: Optional[List[GameSummary]] = None,              # Context for LLM prompt (optional)
) -> StudyPlan:
    """
    Build a practical, compact study plan:
      - Sort tags by frequency (most common issues first).
      - Map tags → concrete actions (from STUDY_ACTIONS with fallbacks).
      - Add light spaced repetition for top issues.
      - Optionally add warmups and a review habit.
      - Allocate tasks per day using minutes_per_day and per_day.
      - If llm_callback is provided, ask the LLM to write a narrative plan (markdown)
        and mark its provenance so UI can label it clearly.
    """
    # 1) Order tags by frequency (descending)
    sorted_tags = [t for t, _ in sorted(weakness_tags_count.items(), key=lambda kv: kv[1], reverse=True)]

    # 2) Map tags → actions; use sensible fallbacks if not defined
    tasks_pool: List[Dict[str, Any]] = []
    for t in sorted_tags:
        action = STUDY_ACTIONS.get(t)
        if not action:
            if t.startswith("opening:"):
                action = "Study basic plans & traps; annotate one model game"
            elif t.startswith("phase:"):
                action = "Review principles; solve 5 themed puzzles"
            elif t.startswith("tactics:"):
                action = "Solve 10 themed puzzles"
            elif t == "king_safety":
                action = "King-safety puzzles; practice creating luft"
        if action:
            tasks_pool.append({"tag": t, "action": action, "reason": _default_reason(t, weakness_tags_count)})

    # 3) Light spaced repetition for first couple of items
    spaced = []
    for i, item in enumerate(tasks_pool):
        spaced.append(item)
        if i < 2:  # duplicate the top items as a quick review
            spaced.append({**item, "action": item["action"] + " (quick review)"})
    tasks_pool = spaced

    # 4) Warmups and review habit if requested
    if include_warmups:
        tasks_pool.insert(0, {"tag": "warmup:puzzles", "action": STUDY_ACTIONS["warmup:puzzles"], "reason": "Activate pattern recognition."})
        tasks_pool.append({"tag": "review:game", "action": STUDY_ACTIONS["review:game"], "reason": "Consolidate from real play."})

    # 5) Allocate tasks across days
    total_min = max(10, minutes_per_day)
    per_task_min = max(5, round(total_min / max(1, per_day)))
    days_out: List[Dict[str, Any]] = []
    idx = 0
    for d in range(1, days + 1):
        day_tasks = tasks_pool[idx:idx+per_day]
        if not day_tasks:
            break
        for t in day_tasks:
            t.setdefault("duration_min", per_task_min)
        days_out.append({"day": d, "total_minutes": min(total_min, per_task_min * len(day_tasks)), "tasks": day_tasks})
        idx += per_day

    plan = StudyPlan(days=days_out)

    # 6) Optional: ask LLM (Groq) to produce a compact narrative plan
    if llm_callback:
        # Provide a small, useful context for the LLM (top weaknesses and notable swings)
        def top_k(d: Dict[str, int], k=6):
            return ", ".join([f"{k1}×{v}" for k1, v in list(sorted(d.items(), key=lambda kv: kv[1], reverse=True))[:k]])
        context = [
            f"Total games: {len(games) if games else '?'}",
            f"Top weaknesses: {top_k(weakness_tags_count, k=8)}",
        ]
        if games:
            spots = []
            for gi, g in enumerate(games[:2], 1):
                for mr in g.key_positions:
                    if mr.label and mr.delta_cp and mr.delta_cp >= 300:
                        spots.append(f"G{gi} ply {mr.ply}: played {mr.san_played}, best {mr.san_best or '?'} (Δ{mr.delta_cp}cp), tags={','.join(mr.tags)}")
                        if len(spots) >= 5:
                            break
                if len(spots) >= 5:
                    break
            if spots:
                context.append("Notable spots: " + " | ".join(spots))

        prompt = f"""
You are a chess study coach. Create a concise, practical, 7-day study plan based on the user's recurring weaknesses.
Constraints: 45 minutes per day, 3 tasks per day. Use short bullets, include a reason after each task in parentheses.

Context:
{os.linesep.join(context)}

Output format:
Day 1:
- task (reason)
- task (reason)
- task (reason)

Day 2:
- ...

Keep it under 180 words total.
""".strip()
        try:
            plan.llm_text = llm_callback(prompt)
            plan.llm_source = "groq"  # provenance marker for UI
        except Exception:
            # If the LLM call fails, we simply skip llm_text and ship the heuristic plan
            plan.llm_text = None
            plan.llm_source = None

    return plan


# =========================
# IO & Reporting
# =========================

def iter_games_from_pgn_paths(pgn_paths: List[str]) -> Iterable[chess.pgn.Game]:
    """Yield games from a list of PGN file paths (supports multiple games per file)."""
    for path in pgn_paths:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                yield game


def dataclass_to_jsonable(obj):
    """
    Recursively convert dataclasses/lists/dicts to JSON-serializable structures.
    Used by emit_json and the UI to export a complete report dump.
    """
    if isinstance(obj, list):
        return [dataclass_to_jsonable(x) for x in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: dataclass_to_jsonable(v) for k, v in obj.items()}
    return obj


def emit_json(report: CharlieReport, path: str):
    """Write the whole report as pretty-printed JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataclass_to_jsonable(report), f, ensure_ascii=False, indent=2)


def emit_markdown(report: CharlieReport, path: str):
    """
    Produce a readable Markdown report:
      - Totals, openings.
      - Per-game summaries + key positions (top deltas).
      - Study plan and optional LLM plan.
    """
    lines: List[str] = []
    lines.append("# Charlie — AI Chess Coach Report\n")
    lines.append(f"- Total games: **{report.aggregated.games}**\n")
    tb = report.aggregated.total_by_label
    lines.append(f"- Errors across all games: Inacc **{tb['inaccuracy']}**, Mistakes **{tb['mistake']}**, Blunders **{tb['blunder']}**\n")

    lines.append("## Openings Played (top)\n")
    for name, cnt in list(report.aggregated.openings_count.items())[:10]:
        lines.append(f"- {name}: {cnt}\n")

    lines.append("## Per-Game Summaries\n")
    for i, g in enumerate(report.games, 1):
        lines.append(f"### Game {i}: {g.headers.get('White','?')} vs {g.headers.get('Black','?')} ({g.headers.get('Date','?')})\n")
        lines.append(f"- Result: {g.headers.get('Result','?')}\n")
        lines.append(f"- Opening: {g.opening.get('eco','?')} — {g.opening.get('opening','?')}\n")
        lines.append(f"- ACPL: White {g.white_acpl}, Black {g.black_acpl}\n")
        lines.append(f"- Errors: W inacc {g.inaccuracies['white']}, mist {g.mistakes['white']}, bl {g.blunders['white']}; "
                     f"B inacc {g.inaccuracies['black']}, mist {g.mistakes['black']}, bl {g.blunders['black']}\n")
        lines.append(f"- Coach note: {g.coach_message}\n")

        # Show top N labeled positions (by delta magnitude)
        kp_sorted = sorted(
            [mr for mr in g.key_positions if mr.label],
            key=lambda mr: (mr.delta_cp or 0),
            reverse=True
        )[:8]
        if kp_sorted:
            lines.append("\n#### Key Positions\n")
            lines.append("| Ply | Side | Phase | Played | Best | Δcp | Label | Tags | FEN |\n")
            lines.append("|---:|:-----|:------|:------|:-----|---:|:-----|:-----|:----|\n")
            for mr in kp_sorted:
                lines.append(
                    f"| {mr.ply} | {mr.side_to_move} | {mr.phase_label} | {mr.san_played} | {mr.san_best or ''} | "
                    f"{mr.delta_cp if mr.delta_cp is not None else ''} | {mr.label or ''} | "
                    f"{', '.join(mr.tags)} | `{mr.fen_before}` |\n"
                )

    lines.append("\n## Study Plan\n")
    # Heuristic plan (engine-first; always present)
    if report.study_plan.days:
        for day in report.study_plan.days:
            lines.append(f"### Day {day['day']}  —  ~{day.get('total_minutes', '')} min\n")
            for t in day["tasks"]:
                dur = f" ({t.get('duration_min', '')} min)" if t.get("duration_min") else ""
                reason = f" — _{t.get('reason','')}_ " if t.get("reason") else ""
                lines.append(f"- **{t['tag']}** → {t['action']}{dur}{reason}\n")
    else:
        lines.append("_No study tasks detected. You played quite solidly!_\n")

    # LLM narrative (optional)
    if report.study_plan.llm_text:
        lines.append("\n### LLM-generated Coaching Plan (optional)\n")
        lines.append(report.study_plan.llm_text.strip() + "\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def emit_key_positions_csv(report: CharlieReport, path: str) -> None:
    """
    Export labeled key positions to CSV for downstream analysis (Excel/BI).
    Columns:
      game_index, white, black, date, ply, side, phase, played, best,
      delta_cp, label, tags, fen
    """
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["game_index", "white", "black", "date", "ply", "side", "phase",
                    "played", "best", "delta_cp", "label", "tags", "fen"])
        for gi, g in enumerate(report.games, 1):
            for mr in getattr(g, "key_positions", []) or []:
                if not getattr(mr, "label", None):
                    continue
                w.writerow([
                    gi,
                    g.headers.get("White", "?"),
                    g.headers.get("Black", "?"),
                    g.headers.get("Date", "?"),
                    getattr(mr, "ply", ""),
                    getattr(mr, "side_to_move", ""),
                    getattr(mr, "phase_label", ""),
                    getattr(mr, "san_played", ""),
                    getattr(mr, "san_best", "") or "",
                    getattr(mr, "delta_cp", "") if getattr(mr, "delta_cp", None) is not None else "",
                    getattr(mr, "label", "") or "",
                    ",".join(getattr(mr, "tags", []) or []),
                    getattr(mr, "fen_before", "")
                ])


# =========================
# Orchestration (pipeline)
# =========================

def run_pipeline(
    stockfish_path: str,
    pgn_paths: List[str],
    depth: Optional[int] = DEFAULT_DEPTH,
    nodes: Optional[int] = DEFAULT_NODES,
    movetime: Optional[float] = DEFAULT_MOVE_TIME,
    plan_days: int = 7,
    plan_per_day: int = 3,
    plan_minutes_per_day: int = 45,
    plan_include_warmups: bool = True,
    llm_callback: Optional[Callable[[str], str]] = None,
) -> CharlieReport:
    """
    End-to-end function called by the UI:
      - Open engine and build limits.
      - Iterate all PGN games and analyze them.
      - Aggregate global stats.
      - Build a study plan (and optionally attach LLM text).
      - Close engine and return a full report.
    """
    engine = make_engine(stockfish_path)
    limit = build_limit(depth=depth, nodes=nodes, movetime=movetime)

    games_out: List[GameSummary] = []
    try:
        for game in iter_games_from_pgn_paths(pgn_paths):
            games_out.append(analyze_game(engine, game, limit))
    finally:
        engine.quit()  # ensure we always release the engine process

    aggregated = aggregate_stats(games_out)
    study_plan = make_study_plan(
        aggregated.weakness_tags_count,
        days=plan_days,
        per_day=plan_per_day,
        minutes_per_day=plan_minutes_per_day,
        include_warmups=plan_include_warmups,
        llm_callback=llm_callback,
        games=games_out,
    )

    return CharlieReport(
        games=games_out,
        aggregated=aggregated,
        study_plan=study_plan,
    )