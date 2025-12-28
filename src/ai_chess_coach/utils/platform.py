from __future__ import annotations

import asyncio
import os


def set_windows_event_loop_policy() -> None:
    """
    On Windows, certain Python builds need Proactor loop policy for subprocesses.
    Safe no-op elsewhere.
    """
    if os.name == "nt":
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        except Exception:
            pass
