"""QuALITY v1.0.1 loader (HTML-stripped JSONL, one line per article and question writer)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Question:
    id: str
    article_id: str
    question: str
    options: list[str]
    gold: int  # 0-based index into options
    hard: bool


@dataclass(frozen=True)
class Article:
    id: str
    title: str
    source: str
    licence: str
    text: str
    questions: list[Question]


def load_quality(path: Path) -> list[Article]:
    """Group the JSONL lines by ``article_id``; each article appears once per question writer."""
    articles: dict[str, Article] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            aid = str(row["article_id"])
            if aid not in articles:
                articles[aid] = Article(
                    id=aid,
                    title=row["title"],
                    source=row.get("source", ""),
                    licence=row.get("license", ""),
                    text=row["article"],
                    questions=[],
                )
            elif articles[aid].text != row["article"]:
                raise ValueError(f"article {aid} has two different texts")
            for q in row["questions"]:
                articles[aid].questions.append(
                    Question(
                        id=q["question_unique_id"],
                        article_id=aid,
                        question=q["question"],
                        options=list(q["options"]),
                        gold=int(q["gold_label"]) - 1,
                        hard=bool(q["difficult"]),
                    )
                )
    return list(articles.values())
