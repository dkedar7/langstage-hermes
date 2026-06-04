"""Summarise a completed Terminal-Bench Harbor job into the numbers the
writeup needs.

Reads ``<job_dir>/result.json`` plus each ``<job_dir>/<task>/result.json``
and produces a short markdown table:

- resolved / total
- mean & median wall time
- mean & median cost (uses per-task ``cost_usd`` if Harbor recorded it;
  otherwise re-computes from token counts via the rates the adapter
  exports as env vars)
- per-task pass/fail with one-line failure-mode classification

Usage::

    python examples/bench_summary.py ~/hermes-bench/jobs/jobs/2026-06-04__11-18-07

Optional flags:
    --tee FILE     also write the markdown to FILE (so the writeup can
                   pick it up directly)
    --csv FILE     emit a per-trial CSV for charting

This script is intentionally read-only — it never touches the agent or
the audit log. Run it as many times as you want against the same job.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def _reward(t: dict) -> float:
    return (t.get("verifier_result") or {}).get("rewards", {}).get("reward", 0.0) or 0.0


def _agent_metadata(t: dict) -> dict:
    return (t.get("agent_result") or {}).get("metadata") or {}


def _classify_failure(t: dict) -> str:
    """Bucket a failed trial — adapter-error vs timeout vs verifier-said-no.

    Harbor records ``exception_info`` for cleanly-raised exceptions
    (e.g. AgentTimeoutError). Our adapter writes ``metadata.error`` for
    crashes it caught inside ``run()``. Everything else is the agent
    ran fine but the verifier returned 0.0.
    """
    exc = t.get("exception_info")
    if exc:
        kind = str(exc.get("kind") or exc.get("type") or exc).lower()
        if "timeout" in kind:
            return "timeout"
        if "rate" in kind:
            return "rate_limit"
        return "exception"
    metadata = _agent_metadata(t)
    if metadata.get("error"):
        err = str(metadata["error"]).lower()
        if "argument list too long" in err:
            return "adapter_argv"
        if "credit balance" in err or "billing" in err:
            return "credit_exhausted"
        if "could not resolve authentication" in err or "401" in err:
            return "model_auth"
        if "timeout" in err:
            return "timeout"
        if "rate limit" in err:
            return "rate_limit"
        return "adapter_error"
    # Agent ran with no recorded error but verifier said 0.0. Distinguish
    # "barely ran" (model returned empty, agent loop exited in seconds) so
    # the writeup can call those out as a separate failure mode rather
    # than counting them as honest attempts.
    n_msgs = metadata.get("n_messages") or 0
    elapsed = metadata.get("elapsed_sec") or 0
    if n_msgs <= 4 and elapsed < 30:
        return "premature_stop"
    return "wrong_answer"


def summarise(job_dir: Path) -> dict:
    """Walk ``job_dir`` and collect per-task results."""
    trial_dirs = [p for p in sorted(job_dir.iterdir()) if p.is_dir()]
    trials: list[dict] = []
    for d in trial_dirs:
        result_path = d / "result.json"
        if not result_path.exists():
            continue
        try:
            with open(result_path) as f:
                trial = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        trial["_task_name"] = d.name
        trials.append(trial)

    resolved = sum(1 for t in trials if _reward(t) >= 1.0)
    walls = [m["elapsed_sec"] for t in trials if (m := _agent_metadata(t)).get("elapsed_sec")]
    in_tokens = [(t.get("agent_result") or {}).get("n_input_tokens") or 0 for t in trials]
    out_tokens = [(t.get("agent_result") or {}).get("n_output_tokens") or 0 for t in trials]
    cache_tokens = [(t.get("agent_result") or {}).get("n_cache_tokens") or 0 for t in trials]

    # Recompute cost properly from raw tokens (the in-trial `cost_usd`
    # may carry the legacy unit bug). Defaults: Sonnet 4.5 list prices.
    ipk = 3.00 / 1_000_000  # fresh input $/tok
    opk = 15.00 / 1_000_000
    cpk = 0.30 / 1_000_000  # cache read
    per_trial_cost = []
    for t in trials:
        ar = t.get("agent_result") or {}
        inp = ar.get("n_input_tokens") or 0
        out = ar.get("n_output_tokens") or 0
        cache = ar.get("n_cache_tokens") or 0
        fresh = max(inp - cache, 0)
        per_trial_cost.append(fresh * ipk + cache * cpk + out * opk)

    failures_by_kind: dict[str, int] = {}
    for t in trials:
        if _reward(t) >= 1.0:
            continue
        kind = _classify_failure(t)
        failures_by_kind[kind] = failures_by_kind.get(kind, 0) + 1

    return {
        "n_trials": len(trials),
        "resolved": resolved,
        "resolved_pct": (resolved / len(trials) * 100) if trials else 0.0,
        "mean_wall": statistics.mean(walls) if walls else 0.0,
        "median_wall": statistics.median(walls) if walls else 0.0,
        "total_cost": sum(per_trial_cost),
        "mean_cost": statistics.mean(per_trial_cost) if per_trial_cost else 0.0,
        "total_input_tokens": sum(in_tokens),
        "total_output_tokens": sum(out_tokens),
        "total_cache_tokens": sum(cache_tokens),
        "failures_by_kind": failures_by_kind,
        "trials": [
            {
                "task": t["_task_name"],
                "resolved": _reward(t) >= 1.0,
                "elapsed_sec": _agent_metadata(t).get("elapsed_sec"),
                "cost_usd": c,
                "n_messages": _agent_metadata(t).get("n_messages"),
                "failure_kind": None if _reward(t) >= 1.0 else _classify_failure(t),
            }
            for t, c in zip(trials, per_trial_cost, strict=True)
        ],
    }


def render_markdown(s: dict) -> str:
    out: list[str] = []
    out.append("## Terminal-Bench 2.0 summary")
    out.append("")
    out.append(f"- **Resolved**: {s['resolved']} / {s['n_trials']} ({s['resolved_pct']:.1f}%)")
    out.append(f"- **Mean wall time / task**: {s['mean_wall']:.1f}s  (median {s['median_wall']:.1f}s)")
    out.append(f"- **Total cost**: ${s['total_cost']:.2f}  (mean ${s['mean_cost']:.3f}/task)")
    cache_pct = (s["total_cache_tokens"] / s["total_input_tokens"] * 100) if s["total_input_tokens"] else 0
    out.append(
        f"- **Total tokens**: {s['total_input_tokens']:,} in / "
        f"{s['total_output_tokens']:,} out / "
        f"{s['total_cache_tokens']:,} cache_read ({cache_pct:.1f}% cache hit)"
    )
    if s["failures_by_kind"]:
        out.append("")
        out.append("### Failure breakdown")
        for kind, n in sorted(s["failures_by_kind"].items(), key=lambda kv: -kv[1]):
            out.append(f"- {kind}: {n}")
    out.append("")
    out.append("### Per-task results")
    out.append("| Task | Resolved | Wall (s) | Cost ($) | Messages | Failure |")
    out.append("|---|---|---|---|---|---|")
    for t in s["trials"]:
        mark = "✅" if t["resolved"] else "❌"
        wall = f"{t['elapsed_sec']:.1f}" if t.get("elapsed_sec") is not None else "—"
        cost = f"{t['cost_usd']:.3f}" if t.get("cost_usd") is not None else "—"
        msgs = str(t["n_messages"]) if t.get("n_messages") is not None else "—"
        fk = t.get("failure_kind") or ""
        out.append(f"| {t['task']} | {mark} | {wall} | {cost} | {msgs} | {fk} |")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_dir", type=Path, help="Path to the Harbor job dir (contains per-task subdirs)")
    parser.add_argument("--tee", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    if not args.job_dir.is_dir():
        print(f"not a directory: {args.job_dir}", file=sys.stderr)
        return 2

    s = summarise(args.job_dir)
    md = render_markdown(s)
    print(md)
    if args.tee:
        args.tee.write_text(md, encoding="utf-8")
    if args.csv:
        import csv

        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["task", "resolved", "elapsed_sec", "cost_usd", "n_messages", "failure_kind"])
            for t in s["trials"]:
                writer.writerow(
                    [
                        t["task"],
                        int(bool(t["resolved"])),
                        t.get("elapsed_sec") or "",
                        t.get("cost_usd") or "",
                        t.get("n_messages") or "",
                        t.get("failure_kind") or "",
                    ]
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
