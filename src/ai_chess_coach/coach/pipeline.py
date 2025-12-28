from __future__ import annotations

from typing import Callable, List, Optional

from ..analysis.core import analyze_game, build_limit, make_engine
from ..ingest.pgn import iter_games_from_pgn_paths
from ..planner.plan import make_study_plan
from ..profiling.aggregate import aggregate_stats
from ..utils.types import CharlieReport, GameSummary


def run_pipeline(
    stockfish_path: str,
    pgn_paths: List[str],
    depth: Optional[int] = None,
    nodes: Optional[int] = None,
    movetime: Optional[float] = None,
    plan_days: int = 7,
    plan_per_day: int = 3,
    plan_minutes_per_day: int = 45,
    plan_include_warmups: bool = True,
    llm_callback: Optional[Callable[[str], str]] = None,
) -> CharlieReport:
    """
    End-to-end function called by the UI:
      - Open engine + build limits.
      - Iterate PGNs and analyze games.
      - Aggregate stats.
      - Build study plan (and optionally attach LLM text).
      - Close engine.
    """
    engine = make_engine(stockfish_path)
    limit = build_limit(depth=depth, nodes=nodes, movetime=movetime)

    games_out: List[GameSummary] = []
    try:
        for game in iter_games_from_pgn_paths(pgn_paths):
            games_out.append(analyze_game(engine, game, limit))
    finally:
        engine.quit()

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
