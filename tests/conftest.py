from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from raptree.embed import HashingEmbedder
from raptree.profiles import make_builder
from raptree.samples import SAMPLE_ID, SAMPLE_TITLE, sample_questions, sample_text
from raptree.tree import Node, RaptorTree, Segment

if TYPE_CHECKING:
    from pathlib import Path

SECOND_ARTICLE = (
    "The orchard at Hollins Farm had forty apple trees and one pear tree. "
    "Every September the family picked the apples by hand and pressed them in a wooden mill. "
    "The juice went into oak barrels in the cellar, where it slowly turned to cider. "
    "In the spring of the dry year, the well behind the barn ran low. "
    "Grandmother Ada carried water to the youngest trees in two tin buckets every evening. "
    "By August most of the old trees had dropped half their fruit, but the young ones held. "
    "The cider that year was sharp and thin, and the family sold almost none of it. "
    "Ada said the young trees had been worth the water, because they would bear for fifty years."
)


@pytest.fixture(scope="session")
def sample_tree() -> RaptorTree:
    return make_builder("offline").build(SAMPLE_ID, SAMPLE_TITLE, sample_text())


def _quality_row(aid: str, writer: str, title: str, text: str, questions: list[dict]) -> dict:
    return {
        "article_id": aid,
        "set_unique_id": f"{aid}_{writer}",
        "source": "fixture",
        "title": title,
        "license": "MIT (test fixture)",
        "article": text,
        "questions": questions,
    }


@pytest.fixture
def quality_file(tmp_path: Path) -> Path:
    """A three-line QuALITY-format file: the sample story (two writers) and a short article."""
    qs = sample_questions()

    def as_row(i: int) -> dict:
        q = qs[i]
        return {
            "question": q.question,
            "question_unique_id": q.id,
            "options": q.options,
            "gold_label": q.gold + 1,
            "difficult": int(q.hard),
        }

    orchard_q = {
        "question": "Why did Ada water the young trees?",
        "question_unique_id": "orchard-1",
        "options": [
            "They would bear fruit for fifty years",
            "The cider needed more pears",
            "The well was full",
            "The old trees were dead",
        ],
        "gold_label": 1,
        "difficult": 0,
    }
    rows = [
        _quality_row(
            "90001", "w1", "The Last Keeper of Gull Rock", sample_text(), [as_row(0), as_row(1)]
        ),
        _quality_row("90002", "w1", "Hollins Farm", SECOND_ARTICLE, [orchard_q]),
        _quality_row(
            "90001", "w2", "The Last Keeper of Gull Rock", sample_text(), [as_row(2), as_row(3)]
        ),
    ]
    path = tmp_path / "mini.quality.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def hand_tree() -> RaptorTree:
    """A fixed seven-node tree (four leaves, two summaries, a root) for TUI and retrieval tests."""
    sentences = [
        "The ferry left the north pier at dawn.",
        "Its captain was a woman called Ines Ortega.",
        "A storm rose over the strait by noon.",
        "The ferry turned back toward the pier.",
        "In the town the bakery opened late that day.",
        "The baker blamed the storm for the late flour delivery.",
        "By evening the sea was calm again.",
        "Ines Ortega took the ferry out a second time.",
    ]
    segments = []
    pos = 0
    for i, s in enumerate(sentences):
        segments.append(
            Segment(
                id=i, text=s, tokens=len(s.split()), origin="article", start=pos, end=pos + len(s)
            )
        )
        pos += len(s) + 1
    leaf_groups = [[0, 1], [2, 3], [4, 5], [6, 7]]
    nodes = [
        Node(
            id=i,
            layer=0,
            text=" ".join(sentences[s] for s in g),
            tokens=sum(segments[s].tokens for s in g),
            segments=g,
            children=[],
            leaves=[i],
        )
        for i, g in enumerate(leaf_groups)
    ]
    summaries = [([0, 1], [0, 2]), ([2, 3], [4, 7])]
    for children, segs in summaries:
        nodes.append(
            Node(
                id=len(nodes),
                layer=1,
                text=" ".join(sentences[s] for s in segs),
                tokens=sum(segments[s].tokens for s in segs),
                segments=segs,
                children=children,
                leaves=children,
            )
        )
    nodes.append(
        Node(
            id=6,
            layer=2,
            text=" ".join(sentences[s] for s in [0, 7]),
            tokens=segments[0].tokens + segments[7].tokens,
            segments=[0, 7],
            children=[4, 5],
            leaves=[0, 1, 2, 3],
        )
    )
    emb = HashingEmbedder()
    return RaptorTree(
        doc_id="hand",
        title="The Ferry",
        segments=segments,
        nodes=nodes,
        layers=[[0, 1, 2, 3], [4, 5], [6]],
        segment_embeddings=emb.embed(sentences),
        node_embeddings=emb.embed([n.text for n in nodes]),
        meta={
            "embedder": emb.name,
            "reducer": "none",
            "summarizer": "hand-written",
            "token_counter": "words",
            "build_seconds": 0.0,
            "article_tokens": sum(s.tokens for s in segments),
        },
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(7)
