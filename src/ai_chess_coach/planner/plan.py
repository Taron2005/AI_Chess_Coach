from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

from ..utils.types import GameSummary, StudyPlan
from .study_actions import STUDY_ACTIONS, maybe_override_actions


def _default_reason(tag: str, counts: Dict[str, int]) -> str:
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
    llm_callback: Optional[Callable[[str], str]] = None,
    games: Optional[List[GameSummary]] = None,
) -> StudyPlan:
    maybe_override_actions()

    sorted_tags = [t for t, _ in sorted(weakness_tags_count.items(), key=lambda kv: kv[1], reverse=True)]

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

    # spaced repetition for top 2
    spaced: List[Dict[str, Any]] = []
    for i, item in enumerate(tasks_pool):
        spaced.append(item)
        if i < 2:
            spaced.append({**item, "action": item["action"] + " (quick review)"})
    tasks_pool = spaced

    if include_warmups:
        tasks_pool.insert(0, {"tag": "warmup:puzzles", "action": STUDY_ACTIONS["warmup:puzzles"], "reason": "Activate pattern recognition."})
        tasks_pool.append({"tag": "review:game", "action": STUDY_ACTIONS["review:game"], "reason": "Consolidate from real play."})

    total_min = max(10, minutes_per_day)
    per_task_min = max(5, round(total_min / max(1, per_day)))

    days_out: List[Dict[str, Any]] = []
    idx = 0
    for d in range(1, days + 1):
        day_tasks = tasks_pool[idx:idx + per_day]
        if not day_tasks:
            break
        for t in day_tasks:
            t.setdefault("duration_min", per_task_min)
        days_out.append({"day": d, "total_minutes": min(total_min, per_task_min * len(day_tasks)), "tasks": day_tasks})
        idx += per_day

    plan = StudyPlan(days=days_out)

    if llm_callback:
        def top_k(dct: Dict[str, int], k: int = 6) -> str:
            return ", ".join([f"{k1}×{v}" for k1, v in list(sorted(dct.items(), key=lambda kv: kv[1], reverse=True))[:k]])

        context = [
            f"Total games: {len(games) if games else '?'}",
            f"Top weaknesses: {top_k(weakness_tags_count, k=8)}",
        ]

        if games:
            spots: List[str] = []
            for gi, g in enumerate(games[:2], 1):
                for mr in g.key_positions:
                    if mr.label and mr.delta_cp and mr.delta_cp >= 300:
                        spots.append(
                            f"G{gi} ply {mr.ply}: played {mr.san_played}, best {mr.san_best or '?'} "
                            f"(Δ{mr.delta_cp}cp), tags={','.join(mr.tags)}"
                        )
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
            plan.llm_source = "groq"
        except Exception:
            plan.llm_text = None
            plan.llm_source = None

    return plan
