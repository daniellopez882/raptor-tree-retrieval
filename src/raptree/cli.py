"""``raptree``: fetch data, build a tree, inspect it in the terminal, ask it questions."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Annotated

import typer

from raptree.embed import Embedder, HashingEmbedder, OnnxSentenceEmbedder
from raptree.llm import LLM_PROVIDERS, ProviderError
from raptree.profiles import PROFILES, SUMMARIZERS, Profile, SummarizerKind, make_builder
from raptree.quality import Question, load_quality
from raptree.retrieve import METHODS, Method, Retriever, query_text
from raptree.samples import SAMPLE_ID, SAMPLE_TITLE, sample_questions, sample_text
from raptree.sources import (
    MODELS,
    data_dir,
    fetch_model,
    fetch_quality,
    quality_path,
    require_model,
)
from raptree.tree import RaptorTree

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)


def _profile(name: str) -> Profile:
    if name not in PROFILES:
        raise typer.BadParameter(f"choose from {', '.join(PROFILES)}", param_hint="--profile")
    return PROFILES[name]


def _method(name: str) -> Method:
    for m in METHODS:
        if m == name:
            return m
    raise typer.BadParameter(f"choose from {', '.join(METHODS)}", param_hint="--method")


def embedder_for(tree: RaptorTree) -> Embedder:
    """The embedder a tree was built with, so query vectors live in the same space."""
    name = str(tree.meta.get("embedder", ""))
    if name.startswith("hashing-"):
        return HashingEmbedder(int(name.split("-", 1)[1]))
    paths = require_model("embedder")
    return OnnxSentenceEmbedder(paths.weights, paths.tokenizer)


def questions_for(tree: RaptorTree) -> list[Question]:
    if tree.doc_id == SAMPLE_ID:
        return sample_questions()
    path = quality_path()
    if not path.exists():
        return []
    for article in load_quality(path):
        if article.id == tree.doc_id:
            return article.questions
    return []


def build_tree(
    doc_id: str,
    title: str,
    text: str,
    profile: Profile,
    summarizer: SummarizerKind = "extractive",
    llm: str = "openai",
) -> RaptorTree:
    return make_builder(profile, summarizer=summarizer, llm=llm).build(doc_id, title, text)


def describe(tree: RaptorTree, preview: int = 70) -> str:
    lines = [
        f"{tree.title} [{tree.doc_id}]",
        f"{len(tree.segments)} sentences, {tree.meta['article_tokens']} tokens "
        f"({tree.meta['token_counter']}), summaries by {tree.meta['summarizer']}",
    ]
    for depth, layer in reversed(list(enumerate(tree.layers))):
        lines.append(f"layer {depth}: {len(layer)} nodes")
        for node_id in layer[:6]:
            node = tree.nodes[node_id]
            text = node.text if len(node.text) <= preview else node.text[: preview - 3] + "..."
            kids = f"{len(node.children)} children" if node.children else "leaf"
            lines.append(f"  #{node.id:<4} {node.tokens:>4} tok  {kids:<12} {text}")
        if len(layer) > 6:
            lines.append(f"  ... {len(layer) - 6} more")
    return "\n".join(lines)


@app.command()
def fetch(
    only: Annotated[str | None, typer.Option(help="'quality' or 'models'; default both.")] = None,
) -> None:
    """Download QuALITY dev and the three ONNX models, verifying every sha256."""
    if only not in (None, "quality", "models"):
        raise typer.BadParameter("use quality or models", param_hint="--only")
    if only in (None, "quality"):
        fetch_quality(typer.echo)
    if only in (None, "models"):
        for key in MODELS:
            fetch_model(key, typer.echo)


@app.command()
def build(
    article: Annotated[str | None, typer.Argument(help="QuALITY article id, e.g. 52845.")] = None,
    text_file: Annotated[Path | None, typer.Option(help="Build from a plain-text file.")] = None,
    sample: Annotated[bool, typer.Option(help="Build the bundled sample story.")] = False,
    profile: Annotated[str, typer.Option(help="onnx or offline.")] = "onnx",
    summarizer: Annotated[str, typer.Option(help="extractive or llm.")] = "extractive",
    llm: Annotated[
        str, typer.Option(help="For --summarizer llm: openai (reads RAPTREE_LLM_*) or fake.")
    ] = "openai",
    out: Annotated[Path | None, typer.Option(help="Where to write the .npz tree.")] = None,
) -> None:
    """Build and save one tree."""
    chosen = _profile(profile)
    if summarizer not in SUMMARIZERS:
        raise typer.BadParameter(f"choose from {', '.join(SUMMARIZERS)}", param_hint="--summarizer")
    if llm not in LLM_PROVIDERS:
        raise typer.BadParameter(f"choose from {', '.join(LLM_PROVIDERS)}", param_hint="--llm")
    if sample:
        doc_id, title, text = SAMPLE_ID, SAMPLE_TITLE, sample_text()
    elif text_file is not None:
        doc_id, title, text = text_file.stem, text_file.stem, text_file.read_text("utf-8")
    elif article is not None:
        found = [a for a in load_quality(fetch_quality(lambda _: None)) if a.id == article]
        if not found:
            raise typer.BadParameter(f"no QuALITY dev article {article}")
        doc_id, title, text = found[0].id, found[0].title, found[0].text
    else:
        raise typer.BadParameter("give an article id, --text-file or --sample")
    try:
        tree = build_tree(doc_id, title, text, chosen, SUMMARIZERS[summarizer], llm)
    except ProviderError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    target = out or data_dir() / "trees" / chosen / f"{doc_id}.npz"
    tree.save(target)
    typer.echo(describe(tree))
    typer.echo(f"saved {target} (built in {tree.meta['build_seconds']:.1f}s)")


@app.command()
def show(tree_path: Annotated[Path, typer.Argument(help="A saved .npz tree.")]) -> None:
    """Print each layer of a tree, top layer first."""
    typer.echo(describe(RaptorTree.load(tree_path)))


@app.command()
def ask(
    tree_path: Annotated[Path, typer.Argument(help="A saved .npz tree.")],
    question: Annotated[str, typer.Argument(help="The question (options optional).")],
    method: Annotated[str, typer.Option(help=", ".join(METHODS))] = "raptor-collapsed",
    budget: Annotated[int, typer.Option(help="Context budget in tokens.")] = 2000,
    beam: Annotated[int, typer.Option(help="Traversal beam width.")] = 3,
) -> None:
    """Show what a retrieval method puts into the context for a question."""
    tree = RaptorTree.load(tree_path)
    chosen = _method(method)
    qvec = embedder_for(tree).embed([question])[0]
    found = Retriever(tree, budget, beam).run(chosen, question, qvec)
    typer.echo(f"{chosen}: {len(found.node_ids)} nodes, {found.tokens}/{budget} tokens")
    for rank, node_id in enumerate(found.node_ids, 1):
        node = tree.nodes[node_id]
        typer.echo(f"{rank:>3}. L{node.layer} #{node.id:<4} {node.tokens:>4} tok  {node.text[:90]}")


@app.command()
def tui(
    tree_path: Annotated[Path, typer.Argument(help="A saved .npz tree.")],
    budget: Annotated[int, typer.Option(help="Context budget in tokens.")] = 2000,
) -> None:
    """Browse a tree interactively and highlight what each method retrieves."""
    from raptree.tui import TreeExplorer

    tree = RaptorTree.load(tree_path)
    TreeExplorer(tree, questions_for(tree), embedder_for(tree), budget=budget).run()


@app.command()
def capture(
    tree_path: Annotated[Path, typer.Argument(help="A saved .npz tree.")],
    depth: Annotated[int, typer.Option(help="Levels to expand below the top layer.")] = 0,
    node: Annotated[int | None, typer.Option(help="Put the cursor on this node id.")] = None,
    question: Annotated[int | None, typer.Option(help="1-based question to ask.")] = None,
    method: Annotated[str, typer.Option(help=", ".join(METHODS))] = "raptor-collapsed",
    width: Annotated[int, typer.Option()] = 120,
    height: Annotated[int, typer.Option()] = 32,
    out: Annotated[Path | None, typer.Option(help="Write here instead of stdout.")] = None,
) -> None:
    """Render the TUI headless after scripted steps and print the screen as text."""
    from raptree.tui import TreeExplorer, screen_text

    tree = RaptorTree.load(tree_path)
    explorer = TreeExplorer(tree, questions_for(tree), embedder_for(tree))
    explorer.method = _method(method)

    async def script() -> str:
        async with explorer.run_test(size=(width, height)) as pilot:
            if depth:
                explorer.expand_to_depth(depth)
            if question is not None:
                explorer.ask(question - 1)
            if node is not None:
                await pilot.pause()
                explorer.focus_node(node)
            await pilot.pause()
            await pilot.pause()
            return screen_text(explorer)

    text = asyncio.run(script())
    if out is None:
        # The screen is full of box-drawing characters; bypass the console code page.
        sys.stdout.buffer.write(text.encode("utf-8"))
        sys.stdout.flush()
    else:
        out.write_text(text, encoding="utf-8")


@app.command()
def demo() -> None:
    """Offline end-to-end run on the bundled story: build, check invariants, retrieve."""
    tree = build_tree(SAMPLE_ID, SAMPLE_TITLE, sample_text(), "offline")
    typer.echo(describe(tree))
    covered = sorted({leaf for i in tree.layers[-1] for leaf in tree.nodes[i].leaves})
    if covered != list(tree.layers[0]):
        raise typer.Exit(1)
    q = sample_questions()[0]
    query = query_text(q.question, q.options)
    qvec = embedder_for(tree).embed([query])[0]
    retriever = Retriever(tree, budget=400, beam=2)
    typer.echo(f"\nQ: {q.question}")
    for method in METHODS:
        found = retriever.run(method, query, qvec)
        layers = [tree.nodes[i].layer for i in found.node_ids]
        typer.echo(f"  {method:<17} {found.tokens:>4}/400 tokens, layers {layers}")
    typer.echo("demo ok: every leaf is reachable from the top layer")
