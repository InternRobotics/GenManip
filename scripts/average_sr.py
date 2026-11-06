#!/usr/bin/env python3
"""Compute average success rate from GenManip result/progress JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _extract_subtask_rates(data: Any) -> list[tuple[str, float, int | None]]:
    """Return (name, sr, episode_count) entries from supported JSON shapes."""
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")

    task = data.get("task")
    if isinstance(task, dict) and isinstance(task.get("sub_task_list"), list):
        rows: list[tuple[str, float, int | None]] = []
        for item in task["sub_task_list"]:
            if not isinstance(item, dict):
                continue
            sr = _as_number(item.get("success_rate", item.get("sr")))
            if sr is None:
                continue
            name = str(item.get("sub_task_name", "<unknown>"))
            episodes = item.get("episode_list")
            episode_count = len(episodes) if isinstance(episodes, list) else None
            rows.append((name, sr, episode_count))
        return rows

    rows = []
    for name, item in data.items():
        if not isinstance(item, dict):
            continue
        sr = _as_number(item.get("sr", item.get("success_rate")))
        if sr is None:
            continue
        episode_count = None
        episodes = item.get("episode_list")
        if isinstance(episodes, list):
            episode_count = len(episodes)
        rows.append((str(name), sr, episode_count))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print average success rate from a result/progress JSON file."
    )
    parser.add_argument("json_path", type=Path, help="Path to result JSON")
    parser.add_argument(
        "--details",
        action="store_true",
        help="Print each parsed subtask success rate before the average.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = _extract_subtask_rates(data)
    if not rows:
        raise RuntimeError(f"No subtask success rates found in {args.json_path}")

    if args.details:
        for name, sr, episode_count in rows:
            suffix = f" episodes={episode_count}" if episode_count is not None else ""
            print(f"{name}\tsr={sr:.10g}{suffix}")

    macro_sr = sum(sr for _, sr, _ in rows) / len(rows)
    weighted_rows = [(sr, n) for _, sr, n in rows if n is not None]

    print(f"subtasks: {len(rows)}")
    print(f"average_sr: {macro_sr:.10f}")
    if weighted_rows:
        total_episodes = sum(n for _, n in weighted_rows)
        if total_episodes > 0:
            weighted_sr = sum(sr * n for sr, n in weighted_rows) / total_episodes
            print(f"episode_weighted_average_sr: {weighted_sr:.10f}")
            print(f"episodes: {total_episodes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
