from __future__ import annotations

from typing import Dict, List

from ..utils.types import AggregatedStats, GameSummary


def tally_openings(games: List[GameSummary]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for g in games:
        name = g.opening.get("opening") or g.opening.get("eco") or "Unknown"
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))


def aggregate_stats(games: List[GameSummary]) -> AggregatedStats:
    total_by_label = {"inaccuracy": 0, "mistake": 0, "blunder": 0}
    total_by_phase = {
        "opening": {"inaccuracy": 0, "mistake": 0, "blunder": 0},
        "middlegame": {"inaccuracy": 0, "mistake": 0, "blunder": 0},
        "endgame": {"inaccuracy": 0, "mistake": 0, "blunder": 0},
    }
    weakness_tags: Dict[str, int] = {}

    for g in games:
        total_by_label["inaccuracy"] += g.inaccuracies.get("white", 0) + g.inaccuracies.get("black", 0)
        total_by_label["mistake"] += g.mistakes.get("white", 0) + g.mistakes.get("black", 0)
        total_by_label["blunder"] += g.blunders.get("white", 0) + g.blunders.get("black", 0)

        for ph in total_by_phase:
            for lab in total_by_phase[ph]:
                total_by_phase[ph][lab] += g.phase_errors.get(ph, {}).get(lab, 0)

        for mr in g.key_positions:
            if mr.label:
                for t in mr.tags:
                    weakness_tags[t] = weakness_tags.get(t, 0) + 1
                phase_tag = f"phase:{mr.phase_label}"
                weakness_tags[phase_tag] = weakness_tags.get(phase_tag, 0) + 1

        op = g.opening.get("opening") or ""
        if "Sicilian" in op:
            weakness_tags["opening:Sicilian"] = weakness_tags.get("opening:Sicilian", 0) + 1
        if "Caro-Kann" in op:
            weakness_tags["opening:Caro-Kann"] = weakness_tags.get("opening:Caro-Kann", 0) + 1
        if "French" in op:
            weakness_tags["opening:French"] = weakness_tags.get("opening:French", 0) + 1

    return AggregatedStats(
        games=len(games),
        total_by_label=total_by_label,
        total_by_phase=total_by_phase,
        openings_count=tally_openings(games),
        weakness_tags_count=dict(sorted(weakness_tags.items(), key=lambda kv: kv[1], reverse=True)),
    )
