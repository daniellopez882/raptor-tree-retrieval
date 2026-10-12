"""Markdown tables printed from a results file; the README tables are copied from this."""

from __future__ import annotations

from typing import Any

from raptree.bench.runner import CONTROL, SUBSETS
from raptree.retrieve import METHODS

LABELS = {
    "truncation": "First 2K tokens",
    "bm25": "BM25 top-k",
    "dense": "Dense top-k",
    "raptor-collapsed": "RAPTOR collapsed",
    "raptor-traversal": "RAPTOR traversal",
    CONTROL: "No context (options only)",
}


def _pct(x: float) -> str:
    return f"{100.0 * x:.1f}"


def _p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def accuracy_table(results: dict[str, Any], reader: str) -> str:
    acc = results["accuracy"][reader]
    lines = [
        "| Context | All | 95% CI | Hard | 95% CI | Easy | 95% CI |",
        "|---|---:|---|---:|---|---:|---|",
    ]
    for method in (*METHODS, CONTROL):
        cells = [LABELS[method]]
        for subset in SUBSETS:
            entry = acc[method].get(subset)
            if entry is None:
                cells += ["", ""]
                continue
            lo, hi = entry["ci95"]
            cells += [_pct(entry["accuracy"]), f"[{_pct(lo)}, {_pct(hi)}]"]
        lines.append("| " + " | ".join(cells) + " |")
    n = acc[METHODS[0]]
    counts = ", ".join(f"{s}: n={n[s]['n']}" for s in SUBSETS if s in n)
    lines.append("")
    lines.append(f"Accuracy in %, chance = 25.0. {counts}.")
    return "\n".join(lines)


def paired_table(results: dict[str, Any], reader: str, against: str = "raptor-collapsed") -> str:
    lines = [
        "| A | B | A - B (points) | only A right | only B right | p | p (Holm) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for t in results["paired_tests"][reader]:
        if against not in (t["a"], t["b"]):
            continue
        lines.append(
            f"| {LABELS[t['a']]} | {LABELS[t['b']]} | {100.0 * t['difference']:+.1f} | "
            f"{t['a_only_correct']} | {t['b_only_correct']} | {_p(t['p_value'])} | "
            f"{_p(t['p_holm'])} |"
        )
    return "\n".join(lines)


def retrieval_table(results: dict[str, Any]) -> str:
    lines = [
        "| Context | Mean tokens | Distinct-sentence tokens | Nodes | From summaries | "
        "Latency p50 (ms) | p90 (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, r in results["retrieval"]["methods"].items():
        lat = r["latency_ms"]
        lines.append(
            f"| {LABELS[method]} | {r['mean_tokens']:.0f} | {r['mean_distinct_tokens']:.0f} | "
            f"{r['mean_nodes']:.1f} | {_pct(r['summary_token_share'])}% | {lat['median']:.2f} | "
            f"{lat['p90']:.2f} |"
        )
    return "\n".join(lines)


def tree_table(results: dict[str, Any]) -> str:
    t = results["trees"]
    rows = [
        ("Build seconds per article", t["build_seconds"]),
        ("Nodes per tree", t["nodes"]),
        ("Leaves per tree", t["leaves"]),
        ("Article tokens", t["article_tokens"]),
    ]
    lines = ["| Per article | Mean | Median | p90 | Max |", "|---|---:|---:|---:|---:|"]
    for label, s in rows:
        lines.append(
            f"| {label} | {s['mean']:.1f} | {s['median']:.1f} | {s['p90']:.1f} | {s['max']:.1f} |"
        )
    depth = ", ".join(f"{k} layers: {v}" for k, v in t["depth_histogram"].items())
    stages = ", ".join(f"{k} {v:.0f}s" for k, v in t["stage_seconds_total"].items())
    lines.append("")
    lines.append(f"Tree depth (layers incl. leaves): {depth}. Stage totals: {stages}.")
    return "\n".join(lines)


def reader_table(results: dict[str, Any]) -> str:
    lines = ["| Reader | Work | Seconds | Throughput |", "|---|---|---:|---:|"]
    for r in results["readers"].values():
        if "pairs" in r:
            work = f"{r['pairs']} pairs, {r['unique_pairs_scored']} distinct"
            rate, unit = r["pairs_per_second"], "pairs/s"
        else:
            work = f"{r['prompts']} prompts, {r['unparsed_replies']} unparsed"
            rate, unit = r["prompts_per_second"], "prompts/s"
        speed = f"{rate:.1f} {unit}" if rate is not None else "-"
        lines.append(f"| {r['model']} | {work} | {r['seconds']:.0f} | {speed} |")
    return "\n".join(lines)


def render(results: dict[str, Any]) -> str:
    m = results["machine"]
    parts = [
        f"Measured on {m['cpu']}, {m['physical_cores']} cores, {m['ram_gb']} GB RAM, {m['os']}, "
        f"Python {m['python']}, CPU only; {results['date']}; git {str(results['git_sha'])[:7]}.",
        f"Command: `{results['command']}`",
        f"Wall time {results['wall_seconds'] / 60:.1f} min, of which tree building "
        f"{results['trees']['total_build_seconds'] / 60:.1f} min.",
    ]
    for reader in results["accuracy"]:
        parts += [f"### Accuracy, reader = {reader}", accuracy_table(results, reader)]
        parts += [f"### Paired tests, reader = {reader}", paired_table(results, reader)]
    parts += ["### Retrieved contexts", retrieval_table(results)]
    parts += ["### Trees", tree_table(results)]
    parts += ["### Readers", reader_table(results)]
    not_run = [f"- {x['what']}: not run ({x['why']})" for x in results["not_run"]]
    parts += ["### Not run", "\n".join(not_run) or "- nothing"]
    return "\n\n".join(parts) + "\n"
