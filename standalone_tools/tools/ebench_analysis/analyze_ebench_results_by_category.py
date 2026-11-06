#!/usr/bin/env python3
"""Analyze ebench full-test and test-mini results by task categories.

Usage:
    /isaac-sim/python.sh standalone_tools/tools/ebench_analysis/analyze_ebench_results_by_category.py \
        --result-root saved/eval_results/ebench/test_release

Optional:
    --taxonomy-json defaults to the task_category_taxonomy.json next to this script.
    --test-mini-json defaults to the test_mini_selection.json next to this script.
    --output-json defaults to <result-root>/category_analysis.json.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

CATEGORY_KEYS = ["scene", "atomic_skill", "range", "precision", "mobility"]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TAXONOMY_JSON = SCRIPT_DIR / "task_category_taxonomy.json"
DEFAULT_TEST_MINI_JSON = SCRIPT_DIR / "test_mini_selection.json"


def normalize_result_root(path: Path) -> Path:
    path = path.resolve()
    if path.name == "ebench":
        return path
    child = path / "ebench"
    if child.is_dir():
        return child
    return path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_result_infos(task_result_dir: Path) -> dict[str, dict[str, Any]]:
    results = {}
    if not task_result_dir.is_dir():
        return results
    for ep_dir in sorted(
        p for p in task_result_dir.iterdir() if p.is_dir() and p.name.isdigit()
    ):
        info_path = ep_dir / "result_info.json"
        if not info_path.exists():
            continue
        with info_path.open("r", encoding="utf-8") as f:
            results[ep_dir.name] = json.load(f)
    return results


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "num_tasks": 0,
            "num_episodes": 0,
            "task_avg_score": None,
            "task_avg_sr": None,
            "episode_avg_score": None,
            "episode_avg_sr": None,
            "tasks": [],
        }
    task_to_scores = defaultdict(list)
    task_to_srs = defaultdict(list)
    for r in records:
        task_to_scores[r["task_key"]].append(float(r.get("score", 0.0)))
        task_to_srs[r["task_key"]].append(float(r.get("success_rate", 0.0)))
    task_avg_score = sum(sum(v) / len(v) for v in task_to_scores.values()) / len(
        task_to_scores
    )
    task_avg_sr = sum(sum(v) / len(v) for v in task_to_srs.values()) / len(task_to_srs)
    episode_avg_score = sum(float(r.get("score", 0.0)) for r in records) / len(records)
    episode_avg_sr = sum(float(r.get("success_rate", 0.0)) for r in records) / len(
        records
    )
    return {
        "num_tasks": len(task_to_scores),
        "num_episodes": len(records),
        "task_avg_score": task_avg_score,
        "task_avg_sr": task_avg_sr,
        "episode_avg_score": episode_avg_score,
        "episode_avg_sr": episode_avg_sr,
        "tasks": sorted(task_to_scores.keys()),
    }


def build_split_records(
    result_root: Path,
    taxonomy: dict[str, Any],
    test_mini: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    full_records = []
    mini_records = []
    mini_tasks = test_mini.get("tasks", {})
    for task_key in taxonomy["tasks"]:
        bench, task = task_key.split("/", 1)
        task_result_dir = result_root / bench / task
        result_infos = iter_result_infos(task_result_dir)
        if not result_infos:
            continue
        mini_eps = set(mini_tasks.get(task_key, []))
        for ep, info in result_infos.items():
            record = {
                "task_key": task_key,
                "bench": bench,
                "task": task,
                "episode": ep,
                "score": float(info.get("score", 0.0)),
                "success_rate": float(info.get("success_rate", 0.0)),
            }
            full_records.append(record)
            if ep in mini_eps:
                mini_records.append(record)
    return full_records, mini_records


def analyze_by_category(
    records: list[dict[str, Any]], taxonomy: dict[str, Any]
) -> dict[str, Any]:
    per_category = {}
    for category in CATEGORY_KEYS:
        category_records = {}
        for value, task_keys in taxonomy["categories"][category].items():
            task_key_set = set(task_keys)
            subset = [r for r in records if r["task_key"] in task_key_set]
            category_records[value] = summarize_records(subset)
        per_category[category] = category_records
    return per_category


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--taxonomy-json", type=Path, default=DEFAULT_TAXONOMY_JSON)
    parser.add_argument("--test-mini-json", type=Path, default=DEFAULT_TEST_MINI_JSON)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_result_root = args.result_root.resolve()
    result_root = normalize_result_root(raw_result_root)
    taxonomy = load_json(args.taxonomy_json.resolve())
    test_mini = load_json(args.test_mini_json.resolve())
    full_records, mini_records = build_split_records(result_root, taxonomy, test_mini)

    output_json = (
        args.output_json.resolve()
        if args.output_json is not None
        else raw_result_root / "category_analysis.json"
    )
    output = {
        "result_root": str(result_root),
        "taxonomy_json": str(args.taxonomy_json.resolve()),
        "test_mini_json": str(args.test_mini_json.resolve()),
        "splits": {
            "full_test": {
                "overall": summarize_records(full_records),
                **analyze_by_category(full_records, taxonomy),
            },
            "test_mini": {
                "overall": summarize_records(mini_records),
                **analyze_by_category(mini_records, taxonomy),
            },
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "num_full_episodes": len(full_records),
                "num_test_mini_episodes": len(mini_records),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
