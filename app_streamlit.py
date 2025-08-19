#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app_streamlit.py — Streamlit UI for Charlie (AI Chess Coach)

Run:
  streamlit run app_streamlit.py

This file is the UI/UX layer:
- Collects inputs (Stockfish path, search mode/limits, Groq LLM toggle, plan knobs).
- Accepts PGN uploads.
- Calls the core pipeline in `charlie_core.py`.
- Renders summaries, per‑game tables, SVG boards, and export buttons.
- When Groq is enabled, clearly marks that the coaching plan is fully LLM‑generated.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from typing import List, Optional, Callable

import pandas as pd
import streamlit as st
import chess
import chess.svg
import streamlit.components.v1 as components
# On Windows, Streamlit + subprocess can need Proactor loop policy
if os.name == "nt":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass

# Core functions and dataclasses from the analysis module
from charlie_core import (
    run_pipeline,
    emit_markdown,
    emit_key_positions_csv,
    dataclass_to_jsonable,
    CharlieReport,
    make_engine,
    build_limit,
)


# ---------- Groq Cloud-only LLM helper ----------
def make_groq_llm(model: str, api_key: str, api_base: str = "https://api.groq.cloud") -> Callable[[str], str]:
    """
    Returns a callable (prompt: str) -> str that hits Groq Cloud.

    Notes
    -----
    - The endpoint here uses a generic `/v1/models/{model}/invoke` shape.
    - If your Groq workspace uses OpenAI-compatible endpoints, replace this with
      a `POST /v1/chat/completions` wrapper.
    """
    import requests
    endpoint = f"{api_base.rstrip('/')}/v1/models/{model}/invoke"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def _call(prompt: str) -> str:
        payload = {"input": prompt}  # minimal payload; adapt if your deployment expects different fields
        r = requests.post(endpoint, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()

        # Try common response fields found in various Groq deployments
        if isinstance(data, dict):
            for key in ("output", "response", "result", "results", "text"):
                if key in data:
                    v = data[key]
                    if isinstance(v, str):
                        return v.strip()
                    if isinstance(v, list) and v and isinstance(v[0], str):
                        return v[0].strip()
                    if isinstance(v, dict) and "text" in v:
                        return v["text"].strip()
            # Fallback: return JSON as a string (useful for debugging)
            return json.dumps(data)
        return str(data)

    return _call


# ---------- Small helpers ----------
def _write_uploaded_files(uploaded_files) -> List[str]:
    """
    Save uploaded PGN files into a temp directory and return their paths.
    Streamlit `UploadedFile` provides file-like buffers; we persist to disk so
    the core can open them as normal files.
    """
    out_paths: List[str] = []
    tmpdir = tempfile.mkdtemp(prefix="charlie_pgns_")
    for uf in uploaded_files:
        # Ensure .pgn extension (some browsers strip it)
        path = os.path.join(tmpdir, uf.name if uf.name.endswith(".pgn") else (uf.name + ".pgn"))
        with open(path, "wb") as f:
            f.write(uf.getbuffer())
        out_paths.append(path)
    return out_paths


def _render_fen_svg(fen: str) -> str:
    """Create an SVG board from a FEN using python-chess' SVG helper."""
    board = chess.Board(fen)
    return chess.svg.board(board=board, size=350)


def _download_bytes_button(label: str, data: bytes, file_name: str, mime: str):
    """Thin wrapper for Streamlit's download button."""
    st.download_button(label=label, data=data, file_name=file_name, mime=mime, use_container_width=True)


def _plan_only_markdown(report: CharlieReport) -> str:
    """
    Build a lightweight Markdown string containing only the study plan
    (heuristic + optional LLM text). Useful for quick sharing.
    """
    lines = ["# Charlie — Study Plan\n"]
    if not report.study_plan.days:
        lines.append("_No study tasks detected._\n")
    else:
        for day in report.study_plan.days:
            lines.append(f"## Day {day['day']}  —  ~{day.get('total_minutes','')} min\n")
            for t in day["tasks"]:
                dur = f" ({t.get('duration_min','')} min)" if t.get("duration_min") else ""
                reason = f" — _{t.get('reason','')}_ " if t.get("reason") else ""
                lines.append(f"- **{t['tag']}** → {t['action']}{dur}{reason}\n")
    if report.study_plan.llm_text:
        lines.append("\n## LLM Coaching Plan (optional)\n")
        lines.append(report.study_plan.llm_text.strip() + "\n")
    return "\n".join(lines)


# ---------- UI ----------
st.set_page_config(page_title="Charlie — AI Chess Coach", page_icon="♟️", layout="wide")
st.title("♟️ Charlie — AI Chess Coach")
st.caption("Local, free prototype. Upload PGNs, point to your Stockfish, and get a study plan.")

with st.sidebar:
    # --- Engine config ---
    st.header("Engine")
    stockfish_path = st.text_input(
        "Path to Stockfish binary (.exe on Windows)",
        value=r"C:\Users\User\Documents\Taron\stockfish/stockfish-windows-x86-64-avx2.exe" if os.name == "nt" else "/usr/local/bin/stockfish",
        help="Example (Windows): C:\\tools\\stockfish\\stockfish.exe",
    )

    # User chooses one type of limit; the core builds the matching engine.Limit
    mode = st.radio("Search mode (pick one)", ["Depth", "Nodes", "Move time (sec)"], horizontal=False)
    depth = st.slider("Depth", 6, 22, 12) if mode == "Depth" else None
    nodes = st.number_input("Nodes (e.g., 200000)", min_value=10000, step=10000, value=200000) if mode == "Nodes" else None
    movetime = st.number_input("Move time per position (seconds)", min_value=0.05, step=0.05, value=0.25) if mode == "Move time (sec)" else None

    # --- Study plan knobs ---
    st.header("Study Plan")
    plan_days = st.slider("Days", 1, 14, 7)
    plan_per_day = st.slider("Tasks per day", 1, 6, 3)
    plan_minutes = st.slider("Minutes per day (total)", 15, 120, 45, step=5)
    plan_warmups = st.checkbox("Include warm-ups & review", value=True)

    # --- Groq Cloud only (optional) ---
    st.subheader("Groq Cloud (optional)")
    # This checkbox decides whether to use the LLM; the app still works without it
    use_groq = st.checkbox("Use Groq Cloud to generate coaching plan", value=False)
    llm_model = st.text_input("Groq model", value="llama-3.1-70b-versatile")
    groq_api_key = ""
    groq_api_base = "https://api.groq.cloud"
    if use_groq:
        # If API key is empty, we continue without LLM (pure heuristic plan)
        groq_api_key = st.text_input("Groq API key", type="password")
        groq_api_base = st.text_input("Groq API base URL", value=groq_api_base)
        st.caption("If not provided, the app will run without LLM.")

    # --- File upload & actions ---
    st.header("Files")
    uploaded = st.file_uploader("Upload one or more PGN files", type=["pgn"], accept_multiple_files=True)

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        analyze_btn = st.button("Analyze with Charlie", type="primary", use_container_width=True)
    with col_btn2:
        test_btn = st.button("Test Stockfish", use_container_width=True)

    st.markdown("---")
    st.caption("If Windows blocked the file after download, in PowerShell run:\n"
               "`Unblock-File \"C:\\\\tools\\\\stockfish\\\\stockfish.exe\"`")


# --- Quick engine sanity check (path + subprocess + minimal search) ---
if test_btn:
    try:
        eng = make_engine(stockfish_path)
        board = chess.Board()  # startpos
        eng.analyse(board, build_limit(depth=6))
        eng.quit()
        st.success("Stockfish launched and analyzed the start position successfully ✅")
    except Exception as e:
        st.error(f"Stockfish test failed: {e}")


# --- Main analysis action ---
if analyze_btn:
    # Basic validation: need PGNs + valid engine path
    if not uploaded:
        st.error("Please upload at least one PGN file.")
        st.stop()
    if not os.path.exists(stockfish_path):
        st.error("Stockfish path is invalid. Please set the correct path in the sidebar.")
        st.stop()

    pgn_paths = _write_uploaded_files(uploaded)

    # Optional LLM hook (Groq only). If not set, the pipeline will skip LLM usage.
    llm_cb: Optional[Callable[[str], str]] = None
    if use_groq and groq_api_key:
        try:
            llm_cb = make_groq_llm(llm_model, groq_api_key, api_base=groq_api_base)
            st.info(f"LLM enabled: **Groq Cloud** · model **{llm_model}**")
        except Exception as e:
            st.warning(f"Could not set up Groq Cloud LLM: {e}. Continuing without LLM.")
    else:
        st.caption("LLM is disabled. Heuristic plan will be used.")

    # Run the core pipeline (engine evals + aggregation + plan, and maybe LLM plan)
    st.info(f"Analyzing {len(pgn_paths)} PGN file(s)…")
    with st.spinner("Stockfish is thinking..."):
        report: CharlieReport = run_pipeline(
            stockfish_path=stockfish_path,
            pgn_paths=pgn_paths,
            depth=depth,
            nodes=nodes,
            movetime=movetime,
            plan_days=plan_days,
            plan_per_day=plan_per_day,
            plan_minutes_per_day=plan_minutes,
            plan_include_warmups=plan_warmups,
            llm_callback=llm_cb,
        )

    # --- Output sections ---
    st.success("Analysis complete!")
    st.subheader("Aggregated Summary")

    # Quick, visual summary tiles
    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("Total Games", report.aggregated.games)
    with colB:
        lbl = report.aggregated.total_by_label
        st.metric("Errors (All):", f"Inacc {lbl['inaccuracy']} / Mist {lbl['mistake']} / Bl {lbl['blunder']}")
    with colC:
        # Three most common openings encountered in uploaded PGNs
        top_openings = list(report.aggregated.openings_count.items())[:3]
        if top_openings:
            st.write("**Top Openings**")
            for name, cnt in top_openings:
                st.write(f"- {name}: {cnt}")
        else:
            st.write("No openings detected.")

    # --- Study Plan (heuristic + optional LLM) ---
    st.markdown("---")
    st.subheader("Study Plan")

    # If checked and LLM plan exists, we show *only* the LLM output and label it clearly
    show_llm_only = st.checkbox("Show only the LLM coaching plan when available", value=True)

    if report.study_plan.llm_text and show_llm_only:
        st.success("This coaching plan is **fully generated by the LLM (Groq Cloud)**.")
        st.markdown(report.study_plan.llm_text)
    else:
        # Heuristic engine-driven plan (always available regardless of LLM)
        if report.study_plan.days:
            for day in report.study_plan.days:
                with st.expander(f"Day {day['day']}  —  ~{day.get('total_minutes','')} min"):
                    for t in day["tasks"]:
                        dur = f" ({t.get('duration_min','')} min)" if t.get("duration_min") else ""
                        reason = f" — _{t.get('reason','')}_ " if t.get("reason") else ""
                        st.write(f"• **{t['tag']}** — {t['action']}{dur}{reason}")
        else:
            st.write("_No study tasks detected._")

        # If there is LLM content too, render it with a clear provenance note
        if report.study_plan.llm_text:
            st.subheader("LLM Coaching Plan (Groq Cloud)")
            st.info("The following section is **entirely written by the LLM**.")
            st.markdown(report.study_plan.llm_text)

    # --- Per-game tables & boards ---
    st.markdown("---")
    st.subheader("Per-Game Details")

    for i, g in enumerate(report.games, 1):
        with st.expander(f"Game {i}: {g.headers.get('White','?')} vs {g.headers.get('Black','?')} ({g.headers.get('Date','?')}) — Result: {g.headers.get('Result','?')}"):
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("White ACPL", g.white_acpl)
            with c2: st.metric("Black ACPL", g.black_acpl)
            with c3: st.write(f"**Opening:** {g.opening.get('eco','?')} — {g.opening.get('opening','?')}")
            # Lightweight coach note derived from total errors & dominant phase
            st.caption(g.coach_message)

            # Build a small dataframe with key (labeled) positions
            rows = []
            for mr in g.key_positions:
                if not mr.label:
                    continue  # keep the table focused on actual issues
                rows.append({
                    "ply": mr.ply,
                    "side": mr.side_to_move,
                    "phase": f"{mr.phase_label} ({mr.phase_score:.2f})",
                    "played": mr.san_played,
                    "best": mr.san_best or "",
                    "Δcp": mr.delta_cp if mr.delta_cp is not None else "",
                    "label": mr.label or "",
                    "tags": ", ".join(mr.tags),
                    "fen": mr.fen_before
                })
            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)

                # Render up to 3 top-delta boards for quick visual inspection
                st.write("**Top 3 key positions (boards):**")
                top3 = sorted([r for r in rows if isinstance(r["Δcp"], (int, float))], key=lambda x: x["Δcp"], reverse=True)[:3]
                if top3:
                    cols = st.columns(len(top3))
                    for idx, r in enumerate(top3):
                        svg = _render_fen_svg(r["fen"])
                        with cols[idx]:
                            components.html(svg, height=380, scrolling=False)
                else:
                    st.write("_No large deltas to preview._")
            else:
                st.write("_No notable errors in this game._")

    # --- Export buttons (JSON, Markdown, CSV, Plan-only Markdown) ---
    st.markdown("---")
    st.subheader("Export")

    # JSON (full dataclass dump)
    json_bytes = (json.dumps(dataclass_to_jsonable(report), ensure_ascii=False, indent=2)).encode("utf-8")
    _download_bytes_button("⬇️ Download JSON (full report)", json_bytes, "charlie_report.json", "application/json")

    # Markdown (full)
    md_tmp = os.path.join(tempfile.gettempdir(), "charlie_report.md")
    emit_markdown(report, md_tmp)
    with open(md_tmp, "rb") as f:
        md_bytes = f.read()
    _download_bytes_button("⬇️ Download Markdown (full report)", md_bytes, "charlie_report.md", "text/markdown")

    # CSV (key positions only)
    csv_tmp = os.path.join(tempfile.gettempdir(), "charlie_key_positions.csv")
    emit_key_positions_csv(report, csv_tmp)
    with open(csv_tmp, "rb") as f:
        csv_bytes = f.read()
    _download_bytes_button("⬇️ Download Key Positions CSV", csv_bytes, "charlie_key_positions.csv", "text/csv")

    # Plan-only Markdown (quick share)
    plan_md = _plan_only_markdown(report).encode("utf-8")
    _download_bytes_button("⬇️ Download Study Plan (Markdown only)", plan_md, "charlie_study_plan.md", "text/markdown")

else:
    # Landing hint when nothing has been run yet
    st.info("⬅️ Upload PGN files, set your Stockfish path, choose plan settings (and Groq if you want), then click **Analyze with Charlie**.")
