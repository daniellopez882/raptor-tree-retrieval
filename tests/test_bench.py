from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from raptree.bench.cli import app
from raptree.bench.report import render
from raptree.bench.runner import CONTROL, BenchConfig, run_benchmark
from raptree.retrieve import METHODS

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


@pytest.fixture
def data_dir(tmp_path: Path, quality_file: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "data"
    target = root / "quality" / "QuALITY.v1.0.1.htmlstripped.dev"
    target.parent.mkdir(parents=True)
    shutil.copy(quality_file, target)
    monkeypatch.setenv("RAPTREE_DATA_DIR", str(root))
    return root


def test_benchmark_end_to_end_offline(quality_file: Path, tmp_path: Path) -> None:
    config = BenchConfig(
        profile="offline", readers=("overlap",), budget=300, bootstrap_resamples=200
    )
    results, predictions = run_benchmark(
        config, quality_file, trees_dir=tmp_path / "trees", log=lambda m: None
    )

    assert results["dataset"] == {"articles": 2, "questions": 5, "hard_questions": 2}
    assert sorted(p.name for p in (tmp_path / "trees").iterdir()) == ["90001.npz", "90002.npz"]
    for method, stats in results["retrieval"]["methods"].items():
        assert stats["mean_tokens"] <= 300, method
        assert stats["mean_distinct_tokens"] <= stats["mean_tokens"]
    acc = results["accuracy"]["overlap"]
    assert set(acc) == {*METHODS, CONTROL}
    for method in acc.values():
        assert method["all"]["n"] == 5
        assert method["hard"]["n"] + method["easy"]["n"] == 5
        lo, hi = method["all"]["ci95"]
        assert lo <= method["all"]["accuracy"] <= hi
    tests = results["paired_tests"]["overlap"]
    assert len(tests) == 15  # every pair of the five methods and the no-context control
    assert all(t["p_value"] <= t["p_holm"] <= 1.0 for t in tests)
    assert results["trees"]["depth_histogram"]
    assert {r["what"] for r in results["not_run"]} >= {
        "LLM reader over the full 2,000-token context"
    }

    assert len(predictions) == 5
    assert set(predictions[0]["methods"]) == {*METHODS, CONTROL}
    assert predictions[0]["methods"][CONTROL]["tokens"] == 0


def test_cli_writes_results_with_provenance(data_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "smoke-test.json"
    result = runner.invoke(
        app,
        [
            "run",
            "--profile",
            "offline",
            "--readers",
            "overlap",
            "--limit-articles",
            "1",
            "--limit-questions",
            "2",
            "--no-save-trees",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    doc = json.loads(out.read_text(encoding="utf-8"))
    for key in ("date", "git_sha", "command", "machine", "config", "accuracy", "trees"):
        assert key in doc
    assert doc["machine"]["cpu"]
    assert doc["machine"]["packages"]["numpy"]
    assert doc["dataset"]["questions"] == 2
    assert "raptree-bench run" in doc["command"]
    predictions = out.with_name("smoke-test.predictions.jsonl").read_text().splitlines()
    assert len(predictions) == 2

    rendered = render(doc)
    assert "| RAPTOR collapsed |" in rendered
    assert "Measured on" in rendered
    report = runner.invoke(app, ["report", str(out)])
    assert report.exit_code == 0
    assert "### Accuracy, reader = overlap" in report.output


def test_limited_runs_cannot_overwrite_headline_results(data_dir: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--profile",
            "offline",
            "--limit-articles",
            "1",
            "--out",
            str(tmp_path / "quality.json"),
        ],
    )
    assert result.exit_code == 2
    assert not (tmp_path / "quality.json").exists()


def test_unknown_reader_is_rejected(quality_file: Path) -> None:
    with pytest.raises(ValueError, match="unknown reader"):
        run_benchmark(
            BenchConfig(profile="offline", readers=("oracle",), limit_articles=1),
            quality_file,
            log=lambda m: None,
        )


def test_llm_steps_run_through_the_fake_provider(quality_file: Path) -> None:
    config = BenchConfig(
        profile="offline",
        readers=("overlap", "llm"),
        summarizer="llm",
        llm_provider="fake",
        budget=300,
        bootstrap_resamples=100,
    )
    results, predictions = run_benchmark(config, quality_file, log=lambda m: None)

    assert results["config"]["summarizer"] == "llm"
    assert results["config"]["llm_provider"] == "fake"
    llm = results["readers"]["llm"]
    assert llm["model"] == "llm(fake)"
    assert llm["prompts"] == 5 * (len(METHODS) + 1)
    assert 0 <= llm["unparsed_replies"] <= llm["prompts"]
    assert set(results["accuracy"]) == {"overlap", "llm"}
    # Only the fake stood in, so the run must say that no real LLM was measured.
    assert [x["what"] for x in results["not_run"]] == ["any real LLM"]
    assert all("llm" in m["predictions"] for p in predictions for m in p["methods"].values())


def test_extractive_runs_list_both_llm_configurations_as_not_run(quality_file: Path) -> None:
    config = BenchConfig(profile="offline", readers=("overlap",), bootstrap_resamples=50)
    assert config.as_dict()["llm_provider"] is None
    assert [x["what"] for x in config.not_run()] == [
        "RAPTOR with LLM-written summaries",
        "LLM reader over the full 2,000-token context",
    ]


def test_headline_file_only_takes_the_onnx_profile(
    data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["run", "--profile", "offline", "--readers", "overlap"])
    assert result.exit_code == 2
    assert not (tmp_path / "bench" / "results").exists()


def test_cli_rejects_unknown_readers_and_providers(data_dir: Path, tmp_path: Path) -> None:
    out = str(tmp_path / "smoke-x.json")
    bad_reader = runner.invoke(app, ["run", "--readers", "oracle", "--out", out])
    assert bad_reader.exit_code == 2
    bad_llm = runner.invoke(app, ["run", "--llm", "mystery", "--out", out])
    assert bad_llm.exit_code == 2
