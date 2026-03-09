#!/usr/bin/env python3
"""Build readable result tables from results/evaluation_results.json.

Examples:
  python src/jepa/make_result_tables.py --print-summary
  python src/jepa/make_result_tables.py --metric auc --print-leaderboard
  python src/jepa/make_result_tables.py --task "HC vs MDD" --print-task-table
  python src/jepa/make_result_tables.py --export-dir results/tables
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_INPUT = Path("results/evaluation_results.json")
DEFAULT_EXPORT_DIR = Path("results/tables")

DISPLAY_EVAL_TYPES = [
    "frozen_linear",
    "frozen_mlp",
    "finetune_loso",
]


def _is_nan(x: Any) -> bool:
    return isinstance(x, float) and math.isnan(x)


def _safe_float(x: Any) -> float:
    if x is None:
        return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")


def _fmt_pct(v: Any) -> str:
    fv = _safe_float(v)
    if _is_nan(fv):
        return "nan"
    return f"{100.0 * fv:.1f}%"


def _fmt_metric(v: Any) -> str:
    fv = _safe_float(v)
    if _is_nan(fv):
        return "nan"
    return f"{fv:.3f}"


def _fmt_metric_pm(v: Any, std: Any, as_percent: bool = False) -> str:
    fv = _safe_float(v)
    fs = _safe_float(std)
    if _is_nan(fv):
        return "nan"
    if _is_nan(fs):
        return _fmt_pct(fv) if as_percent else _fmt_metric(fv)
    if as_percent:
        return f"{100.0 * fv:.1f}%±{100.0 * fs:.1f}%"
    return f"{fv:.3f}±{fs:.3f}"


def load_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a list of result records in JSON file")
    clean: List[Dict[str, Any]] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        clean.append(row)
    return clean


def filter_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        et = r.get("eval_type", "")
        if et in DISPLAY_EVAL_TYPES:
            out.append(r)
    return out


def unique_tasks(rows: Iterable[Dict[str, Any]]) -> List[str]:
    tasks = sorted({str(r.get("task", "")) for r in rows})
    return [t for t in tasks if t]


def unique_methods(rows: Iterable[Dict[str, Any]]) -> List[str]:
    methods = sorted({str(r.get("method", "")) for r in rows})
    return [m for m in methods if m]


def row_score(row: Dict[str, Any], metric: str) -> Tuple[float, float]:
    primary = _safe_float(row.get(metric))
    tie = _safe_float(row.get("bacc"))
    if _is_nan(primary):
        primary = -1.0
    if _is_nan(tie):
        tie = -1.0
    return primary, tie


def leaderboard(rows: List[Dict[str, Any]], metric: str) -> List[Dict[str, Any]]:
    winners: List[Dict[str, Any]] = []
    tasks = unique_tasks(rows)
    for task in tasks:
        for et in DISPLAY_EVAL_TYPES:
            subset = [r for r in rows if r.get("task") == task and r.get("eval_type") == et]
            if not subset:
                continue
            best = max(subset, key=lambda r: row_score(r, metric))
            winners.append(
                {
                    "task": task,
                    "eval_type": et,
                    "winner": best.get("method", ""),
                    "metric": metric,
                    "metric_value": _safe_float(best.get(metric)),
                    "bacc": _safe_float(best.get("bacc")),
                    "auc": _safe_float(best.get("auc")),
                    "f1_macro": _safe_float(best.get("f1_macro")),
                    "n_valid_sites": best.get("n_valid_sites", ""),
                }
            )
    return winners


def task_table(rows: List[Dict[str, Any]], task: str) -> List[Dict[str, Any]]:
    subset = [r for r in rows if r.get("task") == task]
    subset.sort(key=lambda r: (str(r.get("eval_type", "")), str(r.get("method", ""))))
    return subset


def print_markdown_table(headers: List[str], records: List[List[str]]) -> None:
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for rec in records:
        print("| " + " | ".join(rec) + " |")


def print_summary(rows: List[Dict[str, Any]]) -> None:
    tasks = unique_tasks(rows)
    methods = unique_methods(rows)
    print(f"Rows: {len(rows)}")
    print(f"Tasks: {len(tasks)}")
    print(f"Methods: {len(methods)}")
    print("Eval types:")
    for et in DISPLAY_EVAL_TYPES:
        n = sum(1 for r in rows if r.get("eval_type") == et)
        print(f"  - {et}: {n}")


def print_leaderboard(rows: List[Dict[str, Any]], metric: str) -> None:
    lb = leaderboard(rows, metric)
    headers = ["Task", "Eval", "Winner", metric.upper(), "bAcc", "AUC", "F1", "Sites"]
    recs: List[List[str]] = []
    for r in lb:
        recs.append(
            [
                str(r["task"]),
                str(r["eval_type"]),
                str(r["winner"]),
                _fmt_metric(r["metric_value"]),
                _fmt_pct(r["bacc"]),
                _fmt_metric(r["auc"]),
                _fmt_metric(r["f1_macro"]),
                str(r["n_valid_sites"]),
            ]
        )
    print_markdown_table(headers, recs)


def print_task(rows: List[Dict[str, Any]], task: str) -> None:
    tt = task_table(rows, task)
    headers = ["Method", "Eval", "bAcc", "AUC", "F1", "Sites"]
    recs: List[List[str]] = []
    for r in tt:
        recs.append(
            [
                str(r.get("method", "")),
                str(r.get("eval_type", "")),
                _fmt_metric_pm(r.get("bacc"), r.get("bacc_std"), as_percent=True),
                _fmt_metric_pm(r.get("auc"), r.get("auc_std"), as_percent=False),
                _fmt_metric(r.get("f1_macro")),
                str(r.get("n_valid_sites", "")),
            ]
        )
    print_markdown_table(headers, recs)


def export_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def export_tables(rows: List[Dict[str, Any]], out_dir: Path, metric: str) -> None:
    # Full normalized table
    full_fields = [
        "method",
        "eval_type",
        "task",
        "chance",
        "bacc",
        "bacc_std",
        "auc",
        "auc_std",
        "f1_macro",
        "precision",
        "recall",
        "n_valid_sites",
        "n_hc",
        "n_disease",
    ]
    export_csv(out_dir / "all_results_filtered.csv", rows, full_fields)

    # Leaderboard table
    lb = leaderboard(rows, metric)
    lb_fields = ["task", "eval_type", "winner", "metric", "metric_value", "bacc", "auc", "f1_macro", "n_valid_sites"]
    export_csv(out_dir / f"leaderboard_{metric}.csv", lb, lb_fields)

    # One csv per task for quick inspection
    for t in unique_tasks(rows):
        safe = (
            t.replace("/", "_")
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )
        tt = task_table(rows, t)
        export_csv(out_dir / "tasks" / f"{safe}.csv", tt, full_fields)


def main() -> None:
    p = argparse.ArgumentParser(description="Create readable tables from evaluation_results.json")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to evaluation_results.json")
    p.add_argument("--metric", choices=["auc", "bacc", "f1_macro"], default="auc", help="Primary metric for leaderboard ranking")
    p.add_argument("--task", type=str, default="", help="Print a single task table")
    p.add_argument("--print-summary", action="store_true", help="Print dataset summary")
    p.add_argument("--print-leaderboard", action="store_true", help="Print leaderboard markdown table")
    p.add_argument("--print-task-table", action="store_true", help="Print markdown table for --task")
    p.add_argument("--export-dir", type=Path, default=None, help="Export CSV tables into this directory")
    args = p.parse_args()

    rows = filter_rows(load_rows(args.input))

    if args.print_summary:
        print_summary(rows)

    if args.print_leaderboard:
        print_leaderboard(rows, args.metric)

    if args.print_task_table:
        if not args.task:
            raise SystemExit("--print-task-table requires --task")
        print_task(rows, args.task)

    if args.export_dir is not None:
        export_tables(rows, args.export_dir, args.metric)
        print(f"Exported tables to: {args.export_dir}")

    if not any([args.print_summary, args.print_leaderboard, args.print_task_table, args.export_dir is not None]):
        # Default behavior: give a quick useful view.
        print_summary(rows)
        print()
        print_leaderboard(rows, args.metric)


if __name__ == "__main__":
    main()
