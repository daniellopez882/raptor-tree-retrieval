"""``raptree-bench``: run the QuALITY benchmark, print its tables, capture the walkthrough."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Annotated, Any

import typer

from raptree.bench.machine import git_state, local_date, local_timestamp, machine_info
from raptree.bench.report import render
from raptree.bench.runner import ALL_READERS, READERS, BenchConfig, run_benchmark
from raptree.cli import describe, embedder_for
from raptree.llm import LLM_PROVIDERS
from raptree.profiles import PROFILES, SUMMARIZERS, make_builder
from raptree.quality import load_quality
from raptree.sources import QUALITY_DEV, data_dir, quality_path

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)

HEADLINE = Path("bench/results/quality-dev-2k.json")
WALKTHROUGH = Path("bench/results/walkthrough-62139.json")


def _command_line(subcommand: str, params: dict[str, Any]) -> str:
    """The effective command with every option spelled out, so a rerun needs no defaults."""
    parts = ["uv", "run", "raptree-bench", subcommand]
    for name, value in params.items():
        flag = "--" + name.replace("_", "-")
        if value is None:
            continue
        if isinstance(value, bool):
            parts.append(flag if value else "--no-" + name.replace("_", "-"))
        else:
            parts += [flag, value.as_posix() if isinstance(value, Path) else str(value)]
    return shlex.join(parts)


@app.command()
def run(
    ctx: typer.Context,
    out: Annotated[Path, typer.Option(help="Results JSON; predictions go next to it.")] = HEADLINE,
    budget: Annotated[int, typer.Option(help="Context budget in WordPiece tokens.")] = 2000,
    beam: Annotated[int, typer.Option(help="Beam width of tree traversal.")] = 3,
    evidence_tokens: Annotated[int, typer.Option(help="Evidence kept per option.")] = 192,
    readers: Annotated[
        str, typer.Option(help=f"Comma-separated, from {', '.join(ALL_READERS)}.")
    ] = ",".join(READERS),
    profile: Annotated[str, typer.Option(help="onnx (measured) or offline.")] = "onnx",
    summarizer: Annotated[str, typer.Option(help="extractive (measured) or llm.")] = "extractive",
    llm: Annotated[
        str, typer.Option(help="Provider for llm steps: openai (RAPTREE_LLM_*) or fake.")
    ] = "openai",
    limit_articles: Annotated[int | None, typer.Option(help="Smoke runs only.")] = None,
    limit_questions: Annotated[int | None, typer.Option(help="Per article; smoke only.")] = None,
    threads: Annotated[int | None, typer.Option(help="onnxruntime intra-op threads.")] = None,
    save_trees: Annotated[bool, typer.Option(help="Keep trees for `raptree tui`.")] = True,
) -> None:
    """Build all dev trees, retrieve with every method, read, and write the results JSON."""
    if profile not in PROFILES:
        raise typer.BadParameter(f"choose from {', '.join(PROFILES)}", param_hint="--profile")
    if summarizer not in SUMMARIZERS:
        raise typer.BadParameter(f"choose from {', '.join(SUMMARIZERS)}", param_hint="--summarizer")
    if llm not in LLM_PROVIDERS:
        raise typer.BadParameter(f"choose from {', '.join(LLM_PROVIDERS)}", param_hint="--llm")
    chosen = tuple(r.strip() for r in readers.split(",") if r.strip())
    if not chosen or any(r not in ALL_READERS for r in chosen):
        raise typer.BadParameter(f"choose from {', '.join(ALL_READERS)}", param_hint="--readers")
    config = BenchConfig(
        budget=budget,
        beam=beam,
        evidence_tokens=evidence_tokens,
        readers=chosen,
        profile=PROFILES[profile],
        summarizer=SUMMARIZERS[summarizer],
        llm_provider=llm,
        limit_articles=limit_articles,
        limit_questions=limit_questions,
        threads=threads,
    )
    if config.is_subset and not out.name.startswith("smoke-"):
        raise typer.BadParameter(
            "a run limited with --limit-* must write to a file named smoke-*.json",
            param_hint="--out",
        )
    fake = config.uses_llm and config.llm_provider == "fake"
    if out.resolve() == HEADLINE.resolve() and (config.profile != "onnx" or fake):
        raise typer.BadParameter(
            f"{HEADLINE} holds the onnx-profile results; pass another --out", param_hint="--out"
        )
    dataset = _dataset_or_exit()
    trees_dir = data_dir() / "trees" / profile if save_trees else None
    results, predictions = run_benchmark(
        config, dataset, trees_dir=trees_dir, log=lambda m: typer.echo(m, err=True)
    )
    document = {
        "benchmark": "QuALITY v1.0.1 dev, multiple choice under a fixed context budget",
        "date": local_date(),
        "timestamp": local_timestamp(),
        **git_state(Path.cwd()),
        "command": _command_line("run", ctx.params),
        "machine": machine_info(),
        "dataset_file": {"url": QUALITY_DEV.url, "sha256": QUALITY_DEV.sha256},
        **results,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    pred_path = out.with_name(out.stem + ".predictions.jsonl")
    with pred_path.open("w", encoding="utf-8", newline="\n") as fh:
        for record in predictions:
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")
    typer.echo(f"wrote {out} and {pred_path}")


@app.command()
def report(results: Annotated[Path, typer.Argument(help="A results JSON from `run`.")]) -> None:
    """Print the Markdown tables for a results file."""
    typer.echo(render(json.loads(results.read_text(encoding="utf-8"))))


def _dataset_or_exit() -> Path:
    dataset = quality_path()
    if not dataset.exists():
        typer.echo("QuALITY is missing; run `raptree fetch` first.", err=True)
        raise typer.Exit(2)
    return dataset


@app.command()
def walkthrough(
    ctx: typer.Context,
    article: Annotated[str, typer.Option(help="QuALITY dev article id.")] = "62139",
    node: Annotated[int, typer.Option(help="Summary node to open for the provenance view.")] = 84,
    question: Annotated[int, typer.Option(help="1-based question to retrieve for.")] = 7,
    width: Annotated[int, typer.Option(help="Terminal columns of every capture.")] = 100,
    profile: Annotated[str, typer.Option(help="onnx (as quoted) or offline.")] = "onnx",
    out: Annotated[Path, typer.Option(help="Where to write the captures.")] = WALKTHROUGH,
) -> None:
    """Build one article's tree and capture the TUI level by level.

    The README walkthrough quotes these screens verbatim, so they are stored with the same
    provenance fields as the benchmark results.
    """
    if profile not in PROFILES:
        raise typer.BadParameter(f"choose from {', '.join(PROFILES)}", param_hint="--profile")
    found = [a for a in load_quality(_dataset_or_exit()) if a.id == article]
    if not found:
        raise typer.BadParameter(f"no QuALITY dev article {article}", param_hint="--article")
    art = found[0]
    if not 1 <= question <= len(art.questions):
        raise typer.BadParameter(
            f"article has {len(art.questions)} questions", param_hint="--question"
        )
    from raptree.tui import capture_screen

    tree = make_builder(PROFILES[profile]).build(art.id, art.title, art.text)
    if not 0 <= node < len(tree.nodes):
        raise typer.BadParameter(f"tree has {len(tree.nodes)} nodes", param_hint="--node")
    tree.save(data_dir() / "trees" / profile / f"{art.id}.npz")
    embedder = embedder_for(tree)

    def screen(height: int, **steps: Any) -> str:
        return capture_screen(tree, art.questions, embedder, size=(width, height), **steps)

    q = art.questions[question - 1]
    screens = {
        "show": describe(tree) + "\n",
        "top_layer": screen(16),
        "one_level_down": screen(22, depth=1),
        "summary_node": screen(22, node=node),
        "question_collapsed": screen(34, question=question - 1),
        "question_traversal": screen(34, question=question - 1, method="raptor-traversal"),
    }
    document = {
        "what": "README walkthrough: verbatim TUI captures of one QuALITY dev article's tree",
        "date": local_date(),
        "timestamp": local_timestamp(),
        **git_state(Path.cwd()),
        "command": _command_line("walkthrough", ctx.params),
        "machine": machine_info(),
        "article": {"id": art.id, "title": art.title, "source": art.source},
        "question": {"index": question, "id": q.id, "hard": q.hard, "gold": q.gold},
        "tree": {
            "layers": [len(layer) for layer in tree.layers],
            "nodes": len(tree.nodes),
            "sentences": len(tree.segments),
            "article_tokens": tree.meta["article_tokens"],
            "build_config": tree.meta["build_config"],
        },
        "screens": screens,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    typer.echo(f"wrote {out}")
