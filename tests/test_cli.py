from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from raptree.cli import app
from raptree.tree import RaptorTree

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


def test_demo_runs_offline() -> None:
    result = runner.invoke(app, ["demo"])
    assert result.exit_code == 0, result.output
    assert "demo ok" in result.output
    for method in ("truncation", "bm25", "dense", "raptor-collapsed", "raptor-traversal"):
        assert method in result.output


def test_build_show_ask_and_capture_the_sample(tmp_path: Path) -> None:
    tree_path = tmp_path / "keeper.npz"
    built = runner.invoke(
        app, ["build", "--sample", "--profile", "offline", "--out", str(tree_path)]
    )
    assert built.exit_code == 0, built.output
    tree = RaptorTree.load(tree_path)
    assert tree.doc_id == "sample-keeper"

    shown = runner.invoke(app, ["show", str(tree_path)])
    assert shown.exit_code == 0
    assert f"layer 0: {len(tree.layers[0])} nodes" in shown.output

    asked = runner.invoke(
        app,
        ["ask", str(tree_path), "Who brought the supplies?", "--method", "bm25", "--budget", "200"],
    )
    assert asked.exit_code == 0, asked.output
    assert asked.output.startswith("bm25:")

    capture = tmp_path / "screen.txt"
    captured = runner.invoke(
        app,
        ["capture", str(tree_path), "--question", "3", "--depth", "1", "--out", str(capture)],
    )
    assert captured.exit_code == 0, captured.output
    screen = capture.read_text(encoding="utf-8")
    assert "How many steps did Marta climb" in screen
    assert "raptor-collapsed:" in screen


def test_unknown_method_is_a_usage_error(tmp_path: Path) -> None:
    tree_path = tmp_path / "keeper.npz"
    runner.invoke(app, ["build", "--sample", "--profile", "offline", "--out", str(tree_path)])
    result = runner.invoke(app, ["ask", str(tree_path), "q", "--method", "magic"])
    assert result.exit_code == 2


def test_build_needs_a_source() -> None:
    assert runner.invoke(app, ["build", "--profile", "offline"]).exit_code == 2


def test_build_with_llm_summaries_from_the_fake(tmp_path: Path) -> None:
    tree_path = tmp_path / "keeper-llm.npz"
    result = runner.invoke(
        app,
        [
            "build",
            "--sample",
            "--profile",
            "offline",
            "--summarizer",
            "llm",
            "--llm",
            "fake",
            "--out",
            str(tree_path),
        ],
    )
    assert result.exit_code == 0, result.output
    tree = RaptorTree.load(tree_path)
    assert tree.meta["summarizer"].startswith("llm(fake")
    assert any(s.origin == "summary" for s in tree.segments)


def test_llm_summaries_without_an_endpoint_fail_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RAPTREE_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("RAPTREE_LLM_MODEL", raising=False)
    result = runner.invoke(
        app,
        ["build", "--sample", "--profile", "offline", "--summarizer", "llm"],
    )
    assert result.exit_code == 1
    assert "RAPTREE_LLM_BASE_URL" in result.output
