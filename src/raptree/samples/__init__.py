"""A short original story with four questions, used by the offline demo and the tests."""

from __future__ import annotations

import json
from importlib import resources

from raptree.quality import Question

SAMPLE_ID = "sample-keeper"
SAMPLE_TITLE = "The Last Keeper of Gull Rock"


def sample_text() -> str:
    return resources.files(__name__).joinpath("keeper.txt").read_text(encoding="utf-8")


def sample_questions() -> list[Question]:
    raw = json.loads(resources.files(__name__).joinpath("keeper_questions.json").read_text("utf-8"))
    return [
        Question(
            id=f"{SAMPLE_ID}-{i + 1}",
            article_id=SAMPLE_ID,
            question=q["question"],
            options=q["options"],
            gold=q["gold"],
            hard=q["hard"],
        )
        for i, q in enumerate(raw)
    ]
