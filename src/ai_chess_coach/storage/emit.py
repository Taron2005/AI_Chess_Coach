from __future__ import annotations

import json
from typing import List

from ..utils.serialize import dataclass_to_jsonable
from ..utils.types import CharlieReport


def emit_json(report: CharlieReport, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataclass_to_jsonable(report), f, ensure_ascii=False, indent=2)


def emit_markdown(report: CharlieReport, path: str) -> None:
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
        lines.append(
            f"- Errors: W inacc {g.inaccuracies['white']}, mist {g.mistakes['white']}, bl {g.blunders['white']}; "
            f"B inacc {g.inaccuracies['black']}, mist {g.mistakes['black']}, bl {g.blunders['black']}\n"
        )
        lines.append(f"- Coach note: {g.coach_message}\n")

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
    if report.study_plan.days:
        for day in report.study_plan.days:
            lines.append(f"### Day {day['day']}  —  ~{day.get('total_minutes', '')} min\n")
            for t in day["tasks"]:
                dur = f" ({t.get('duration_min', '')} min)" if t.get("duration_min") else ""
                reason = f" — _{t.get('reason','')}_ " if t.get("reason") else ""
                lines.append(f"- **{t['tag']}** → {t['action']}{dur}{reason}\n")
    else:
        lines.append("_No study tasks detected. You played quite solidly!_\n")

    if report.study_plan.llm_text:
        lines.append("\n### LLM-generated Coaching Plan (optional)\n")
        lines.append(report.study_plan.llm_text.strip() + "\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def emit_key_positions_csv(report: CharlieReport, path: str) -> None:
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["game_index", "white", "black", "date", "ply", "side", "phase",
                    "played", "best", "delta_cp", "label", "tags", "fen"])
        for gi, g in enumerate(report.games, 1):
            for mr in g.key_positions:
                if not mr.label:
                    continue
                w.writerow([
                    gi,
                    g.headers.get("White", "?"),
                    g.headers.get("Black", "?"),
                    g.headers.get("Date", "?"),
                    mr.ply,
                    mr.side_to_move,
                    mr.phase_label,
                    mr.san_played,
                    mr.san_best or "",
                    mr.delta_cp if mr.delta_cp is not None else "",
                    mr.label or "",
                    ",".join(mr.tags),
                    mr.fen_before,
                ])
