from __future__ import annotations

from typing import Iterable, List

import chess.pgn


def iter_games_from_pgn_paths(pgn_paths: List[str]) -> Iterable[chess.pgn.Game]:
    """Yield games from a list of PGN file paths (supports multiple games per file)."""
    for path in pgn_paths:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                yield game
