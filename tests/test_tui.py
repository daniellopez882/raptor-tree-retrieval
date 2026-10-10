"""Textual pilot tests: drive the explorer headless and compare the screen with text snapshots.

Set UPDATE_SNAPSHOTS=1 to rewrite the files under tests/snapshots after a deliberate change.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING

from textual.widgets import OptionList, Static

from raptree.embed import HashingEmbedder
from raptree.quality import Question
from raptree.tui import TreeExplorer, screen_text

if TYPE_CHECKING:
    from textual.pilot import Pilot

    from raptree.tree import RaptorTree

SNAPSHOTS = Path(__file__).parent / "snapshots"
SIZE = (110, 26)

QUESTIONS = [
    Question(
        "h1",
        "hand",
        "Who captained the ferry?",
        ["Ines Ortega", "The baker", "Nobody", "Ada"],
        0,
        False,
    ),
    Question(
        "h2",
        "hand",
        "Why did the ferry turn back?",
        ["Engine trouble", "A storm rose", "No passengers", "Late flour"],
        1,
        True,
    ),
]


def assert_snapshot(name: str, text: str) -> None:
    path = SNAPSHOTS / f"{name}.txt"
    if os.environ.get("UPDATE_SNAPSHOTS") == "1":
        path.parent.mkdir(exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    assert text == path.read_text(encoding="utf-8")


async def settle(pilot: Pilot[None]) -> None:
    """Let queued messages (footer bindings, tree re-layout) finish before reading the screen."""
    for _ in range(3):
        await pilot.pause()


def explorer(tree: RaptorTree) -> TreeExplorer:
    return TreeExplorer(tree, QUESTIONS, HashingEmbedder(), budget=40, beam=1)


def test_initial_screen_shows_the_top_layer(hand_tree: RaptorTree) -> None:
    async def run() -> str:
        app = explorer(hand_tree)
        async with app.run_test(size=SIZE) as pilot:
            await settle(pilot)
            return screen_text(app)

    assert_snapshot("initial", asyncio.run(run()))


def test_expanding_levels_reaches_the_leaves(hand_tree: RaptorTree) -> None:
    async def run() -> str:
        app = explorer(hand_tree)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("e", "e")
            await settle(pilot)
            return screen_text(app)

    text = asyncio.run(run())
    assert "L0 #3" in text
    assert_snapshot("expanded", text)


def test_selecting_a_question_highlights_retrieved_nodes(hand_tree: RaptorTree) -> None:
    async def run() -> tuple[str, str]:
        app = explorer(hand_tree)
        async with app.run_test(size=SIZE) as pilot:
            app.query_one(OptionList).focus()
            await pilot.press("down", "enter")
            await settle(pilot)
            collapsed = screen_text(app)
            assert app.retrieval is not None
            assert app.retrieval.method == "raptor-collapsed"
            assert app.retrieval.tokens <= 40
            await pilot.press("m")  # collapsed -> traversal
            await settle(pilot)
            assert app.retrieval.method == "raptor-traversal"
            return collapsed, screen_text(app)

    collapsed, traversal = asyncio.run(run())
    assert "●" in collapsed
    assert "Why did the ferry turn back?" in collapsed
    assert_snapshot("question-collapsed", collapsed)
    assert_snapshot("question-traversal", traversal)


def test_cursor_on_a_node_shows_its_provenance(hand_tree: RaptorTree) -> None:
    async def run() -> str:
        app = explorer(hand_tree)
        async with app.run_test(size=SIZE) as pilot:
            app.focus_node(5)
            await settle(pilot)
            return str(app.query_one("#detail", Static).content)

    detail = asyncio.run(run())
    assert "L1 #5" in detail
    assert "summary of 2 children: 2-3" in detail
    assert "covers leaves 2-3 (2 of 4, 50%)" in detail
    assert "parents: #6" in detail
    assert "copied verbatim" in detail


def test_collapse_folds_everything_back(hand_tree: RaptorTree) -> None:
    async def run() -> tuple[str, str]:
        app = explorer(hand_tree)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("e", "e")
            await settle(pilot)
            expanded = screen_text(app)
            await pilot.press("c")
            await settle(pilot)
            return expanded, screen_text(app)

    expanded, folded = asyncio.run(run())
    assert "L0 #0" in expanded
    assert "L0 #0" not in folded
