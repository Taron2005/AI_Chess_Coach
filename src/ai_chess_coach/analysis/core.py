from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional, Tuple

import chess
import chess.engine
import chess.pgn

from ..utils.platform import set_windows_event_loop_policy
from ..utils.types import GameSummary, MoveRecord

# =========================
# Config & thresholds
# =========================

BASE_THRESHOLDS_CP = {
    "inaccuracy": 100,
    "mistake": 250,
    "blunder": 500,
}

PHASE_WEIGHTS = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
}
PHASE_MAX = 2 * (2 * PHASE_WEIGHTS[chess.KNIGHT] + 2 * PHASE_WEIGHTS[chess.BISHOP]
                 + 2 * PHASE_WEIGHTS[chess.ROOK] + 1 * PHASE_WEIGHTS[chess.QUEEN])

DEFAULT_DEPTH = 14


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

    set_windows_event_loop_policy()

    try:
        return chess.engine.SimpleEngine.popen_uci(stockfish_path)
    except NotImplementedError:
        # Rare Windows subprocess edge cases
        set_windows_event_loop_policy()
        return chess.engine.SimpleEngine.popen_uci(stockfish_path)


def build_limit(
    depth: Optional[int] = None,
    nodes: Optional[int] = None,
    movetime: Optional[float] = None
) -> chess.engine.Limit:
    """
    Build a chess.engine.Limit from mutually exclusive inputs.
    Priority: nodes -> movetime -> depth -> DEFAULT_DEPTH
    """
    if nodes is not None:
        return chess.engine.Limit(nodes=nodes)
    if movetime is not None:
        return chess.engine.Limit(time=movetime)
    return chess.engine.Limit(depth=(depth if depth is not None else DEFAULT_DEPTH))


def _score_cp_and_mate(
    info: chess.engine.InfoDict,
    pov: chess.Color,
    mate_cp: int = 100000
) -> Tuple[Optional[int], Optional[int]]:
    """
    Convert an engine score (mate or centipawns) into:
    - cp: centipawns, mapping mate to large cp via mate_score=mate_cp
    - mate: mate distance (plies) or None
    """
    if "score" not in info or info["score"] is None:
        return None, None
    s = info["score"].pov(pov)
    cp = s.score(mate_score=mate_cp)
    mate = s.mate()
    return cp, mate


def analyze_multipv(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    limit: chess.engine.Limit,
    n: int = 2
):
    """
    Query engine for up to n PVs and return list sorted by MultiPV rank.
    """
    if n <= 1:
        res = engine.analyse(board, limit)
        return [res]

    res = engine.analyse(board, limit, multipv=n)
    if isinstance(res, list):
        return sorted(res, key=lambda d: int(d.get("multipv", 1)))
    return [res]


# =========================
# Phase computation
# =========================

def phase_score(board: chess.Board) -> float:
    """
    Normalized phase score in [0, 1]: 1≈opening, 0≈endgame.
    Counts major/minor pieces on the board.
    """
    phase = 0
    for pt, w in PHASE_WEIGHTS.items():
        if w == 0:
            continue
        phase += (len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK))) * w
    return max(0.0, min(1.0, phase / PHASE_MAX))


def phase_label(phase_value: float) -> str:
    if phase_value >= 0.66:
        return "opening"
    if phase_value <= 0.25:
        return "endgame"
    return "middlegame"


# =========================
# Tactic tagging helpers
# =========================

def _is_self_pinned(board: chess.Board, color: chess.Color, square: chess.Square) -> bool:
    return board.is_pinned(color, square)


def _en_prise_after(board_after: chess.Board, color: chess.Color) -> bool:
    """
    Detect if any of our non-pawn pieces are hanging after the move:
    crude attackers vs defenders count.
    """
    for pt in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
        for sq in board_after.pieces(pt, color):
            defenders = sum(1 for _ in board_after.attackers(color, sq))
            attackers = sum(1 for _ in board_after.attackers(not color, sq))
            if attackers > defenders:
                return True
    return False


def classify_tactic_theme(
    board_before: chess.Board,
    move: chess.Move,
    best_move: Optional[chess.Move],
    delta_cp: Optional[int]
) -> List[str]:
    """
    Lightweight tactical classification.
    """
    tags: List[str] = []
    color = board_before.turn

    after = board_before.copy(stack=False)
    after.push(move)

    if _en_prise_after(after, color):
        tags.append("tactics:en_prise")

    moved_piece = after.piece_at(move.to_square)
    if moved_piece and _is_self_pinned(after, color, move.to_square):
        tags.append("tactics:self_pin")

    if best_move and delta_cp is not None and delta_cp >= 120:
        bm_san = board_before.san(best_move)
        if "x" in bm_san:
            tags.append("tactics:missed_capture")
        if "+" in bm_san or "#" in bm_san:
            tags.append("tactics:missed_check")

    # Back-rank risk heuristic
    king_sq = after.king(color)
    if king_sq is not None:
        rank = chess.square_rank(king_sq)
        back_rank = 0 if color == chess.WHITE else 7
        if rank == back_rank:
            files = [5, 6, 7]  # f g h
            pawns_back = 0
            for f in files:
                sq = chess.square(f, 1 if color == chess.WHITE else 6)
                pc = after.piece_at(sq)
                if pc and pc.piece_type == chess.PAWN and pc.color == color:
                    pawns_back += 1
            if pawns_back >= 2:
                tags.append("tactics:back_rank")

    # Knight fork heuristic
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

    # Simple king safety nudge
    if moved_from and moved_from.piece_type == chess.PAWN and king_sq is not None:
        if abs(chess.square_file(move.from_square) - chess.square_file(king_sq)) <= 1:
            tags.append("king_safety")

    return sorted(set(tags))


# =========================
# Dynamic labeling
# =========================

def _clamp(a: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, a))


def dynamic_thresholds(cp_best: int, multipv_gap: int, phase_val: float) -> Tuple[int, int, int]:
    """
    Adjust thresholds based on multipv gap, advantage size, and phase.
    """
    inc = BASE_THRESHOLDS_CP["inaccuracy"]
    mist = BASE_THRESHOLDS_CP["mistake"]
    bl = BASE_THRESHOLDS_CP["blunder"]

    gap = _clamp(abs(multipv_gap), 0, 800)
    f_gap = _clamp(1.3 - (gap / 800.0), 0.6, 1.3)

    adv = _clamp(abs(cp_best), 0, 2000)
    f_adv = _clamp(1.2 - (adv / 2000.0), 0.7, 1.2)

    f_phase = _clamp(0.7 + 0.6 * phase_val, 0.7, 1.3)

    scale = f_gap * f_adv * f_phase
    return int(inc * scale), int(mist * scale), int(bl * scale)


def label_from_engine(
    best_info: Dict[str, Any],
    played_cp: Optional[int],
    phase_val: float,
    second_info: Optional[Dict[str, Any]]
) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    """
    Decide label with mate awareness + dynamic thresholds.
    Returns (label, cp_best, multipv_gap).
    """
    if played_cp is None or best_info is None:
        return None, None, None

    cp_best, mate_best = best_info.get("_cp"), best_info.get("_mate")
    if cp_best is None:
        return None, None, None

    delta = cp_best - played_cp

    if mate_best is not None:
        if mate_best > 0 and delta > 0:
            return "blunder", cp_best, 0

    if played_cp <= -90000:
        return "blunder", cp_best, 0

    gap = 0
    if second_info and second_info.get("_cp") is not None:
        gap = abs(cp_best - second_info["_cp"])

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

def analyze_game(
    engine: chess.engine.SimpleEngine,
    game: chess.pgn.Game,
    limit: chess.engine.Limit
) -> GameSummary:
    """
    Analyze a single PGN with Stockfish.
    """
    board = game.board()
    headers = dict(game.headers)
    opening_info = {
        "eco": headers.get("ECO"),
        "opening": headers.get("Opening"),
        "variation": headers.get("Variation"),
    }

    white_deltas: List[int] = []
    black_deltas: List[int] = []

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

        ph_score = phase_score(board)
        ph_label = phase_label(ph_score)

        infos = analyze_multipv(engine, board, limit, n=2)

        def pack(info: Dict[str, Any]) -> Dict[str, Any]:
            cp, mate = _score_cp_and_mate(info, pov=board.turn)
            pv = info.get("pv", [None])
            mv = pv[0] if pv else None
            return {"_cp": cp, "_mate": mate, "_move": mv, "raw": info}

        best = pack(infos[0]) if infos else {"_cp": None, "_mate": None, "_move": None}
        second = pack(infos[1]) if len(infos) >= 2 else None

        played_info = engine.analyse(board, limit, root_moves=[move])
        played_cp, _played_mate = _score_cp_and_mate(played_info, pov=board.turn)

        label, cp_best, _gap = label_from_engine(best, played_cp, ph_score, second)

        delta: Optional[int] = None
        if cp_best is not None and played_cp is not None:
            delta = cp_best - played_cp

        tags = classify_tactic_theme(board, move, best.get("_move"), delta)

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

        if delta is not None:
            if side_before == "white":
                white_deltas.append(max(0, delta))
            else:
                black_deltas.append(max(0, delta))

        if label:
            label_counts[side_before][label] += 1
            phase_error_counts[ph_label][label] += 1

        board.push(move)
        ply += 1

    white_acpl = float(sum(white_deltas) / len(white_deltas)) if white_deltas else 0.0
    black_acpl = float(sum(black_deltas) / len(black_deltas)) if black_deltas else 0.0

    total_blunders = label_counts["white"]["blunder"] + label_counts["black"]["blunder"]
    total_mistakes = label_counts["white"]["mistake"] + label_counts["black"]["mistake"]
    total_inacc = label_counts["white"]["inaccuracy"] + label_counts["black"]["inaccuracy"]

    top_phase = "opening"
    if any(sum(v.values()) for v in phase_error_counts.values()):
        top_phase = max(phase_error_counts.items(), key=lambda kv: sum(kv[1].values()))[0]

    if total_blunders or total_mistakes or total_inacc:
        coach_message = (
            f"You had {total_blunders} blunder(s), {total_mistakes} mistake(s), and {total_inacc} inaccuracy(ies). "
            f"Most issues happened in the {top_phase}."
        )
    else:
        coach_message = "Clean game—few significant errors. Nice!"

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
