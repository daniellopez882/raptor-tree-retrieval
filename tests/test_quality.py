from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from raptree.quality import load_quality

if TYPE_CHECKING:
    from pathlib import Path


def test_lines_are_grouped_by_article(quality_file: Path) -> None:
    articles = load_quality(quality_file)
    assert [a.id for a in articles] == ["90001", "90002"]
    keeper = articles[0]
    assert len(keeper.questions) == 4
    assert [q.gold for q in keeper.questions] == [1, 2, 2, 0]  # QuALITY labels are 1-based
    assert [q.hard for q in keeper.questions] == [True, True, False, False]
    assert all(q.article_id == "90001" for q in keeper.questions)


def test_conflicting_article_texts_are_rejected(tmp_path: Path, quality_file: Path) -> None:
    lines = quality_file.read_text(encoding="utf-8").splitlines()
    bad = tmp_path / "bad.jsonl"
    bad.write_text(lines[0] + "\n" + lines[2].replace("Marta", "Maria") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="two different texts"):
        load_quality(bad)
