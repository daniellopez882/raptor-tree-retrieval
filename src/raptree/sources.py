"""Pinned downloads: QuALITY and the three ONNX models, each checked against a sha256.

The dataset lives under ``$RAPTREE_DATA_DIR`` (default ``~/.cache/raptor-tree-retrieval``);
the models go to the Hugging Face cache (``$HF_HOME``) so other tools can share them.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

DATA_ENV = "RAPTREE_DATA_DIR"

Log = Callable[[str], None]


def data_dir() -> Path:
    configured = os.environ.get(DATA_ENV)
    base = Path(configured) if configured else Path.home() / ".cache" / "raptor-tree-retrieval"
    return base.expanduser()


class ChecksumError(RuntimeError):
    pass


@dataclass(frozen=True)
class PinnedFile:
    relpath: str
    url: str
    size: int
    sha256: str


QUALITY_COMMIT = "05e85750d4c5444d2a0a4ad299f6df5f4df06068"
QUALITY_DEV = PinnedFile(
    relpath="quality/QuALITY.v1.0.1.htmlstripped.dev",
    url=(
        f"https://raw.githubusercontent.com/nyu-mll/quality/{QUALITY_COMMIT}"
        "/data/v1.0.1/QuALITY.v1.0.1.htmlstripped.dev"
    ),
    size=8_804_985,
    sha256="99852d874994078e4b4112b71ceca4dd35aa3a24ff6d3a35c051be25295b4fef",
)


@dataclass(frozen=True)
class ModelFile:
    filename: str
    size: int
    sha256: str


@dataclass(frozen=True)
class PinnedModel:
    key: str
    repo: str
    revision: str
    licence: str
    weights: ModelFile
    tokenizer: ModelFile


MODELS: dict[str, PinnedModel] = {
    m.key: m
    for m in (
        PinnedModel(
            key="embedder",
            repo="sentence-transformers/all-MiniLM-L6-v2",
            revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
            licence="Apache-2.0",
            weights=ModelFile(
                "onnx/model_quint8_avx2.onnx",
                23_046_789,
                "b941bf19f1f1283680f449fa6a7336bb5600bdcd5f84d10ddc5cd72218a0fd21",
            ),
            tokenizer=ModelFile(
                "tokenizer.json",
                466_247,
                "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
            ),
        ),
        PinnedModel(
            key="nli",
            repo="MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33",
            revision="c07f66d9cbf781191bee66edfe8ad7856f045781",
            licence="MIT",
            weights=ModelFile(
                "onnx/model.onnx",
                51_140_063,
                "89ca1f96830106de21fcea15b55685060dd6faafd7864a790594910f2e661e3f",
            ),
            tokenizer=ModelFile(
                "tokenizer.json",
                711_396,
                "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
            ),
        ),
        PinnedModel(
            key="cross-encoder",
            repo="cross-encoder/ms-marco-MiniLM-L6-v2",
            revision="233902d25c440f23af6f7d6e94d2946bac0bee0a",
            licence="Apache-2.0",
            weights=ModelFile(
                "onnx/model_quint8_avx2.onnx",
                23_200_716,
                "c80a8b34256ea453093d612e3ac48d3d965a0c0a48c7906709af8b8e28461bf9",
            ),
            tokenizer=ModelFile(
                "tokenizer.json",
                711_396,
                "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
            ),
        ),
    )
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(path: Path, size: int, sha256: str) -> None:
    actual_size = path.stat().st_size
    if actual_size != size:
        raise ChecksumError(f"{path}: {actual_size} bytes, pinned {size}")
    actual = sha256_file(path)
    if actual != sha256:
        raise ChecksumError(f"{path}: sha256 {actual}, pinned {sha256}")


def quality_path() -> Path:
    return data_dir() / QUALITY_DEV.relpath


def fetch_quality(log: Log = print) -> Path:
    dest = quality_path()
    if dest.exists():
        _verify(dest, QUALITY_DEV.size, QUALITY_DEV.sha256)
        log(f"verified   {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"download   {QUALITY_DEV.url}")
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(QUALITY_DEV.url, timeout=120) as r:
            shutil.copyfileobj(r, out, length=1 << 20)
        _verify(tmp, QUALITY_DEV.size, QUALITY_DEV.sha256)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    log(f"verified   {dest}")
    return dest


@dataclass(frozen=True)
class ModelPaths:
    weights: Path
    tokenizer: Path


def _cached(model: PinnedModel, item: ModelFile) -> Path | None:
    from huggingface_hub import try_to_load_from_cache

    hit = try_to_load_from_cache(model.repo, item.filename, revision=model.revision)
    return Path(hit) if isinstance(hit, str) else None


def locate_model(key: str) -> ModelPaths | None:
    """Cached, verified paths of a pinned model, or None when it has not been fetched."""
    model = MODELS[key]
    weights, tokenizer = _cached(model, model.weights), _cached(model, model.tokenizer)
    if weights is None or tokenizer is None:
        return None
    _verify(weights, model.weights.size, model.weights.sha256)
    _verify(tokenizer, model.tokenizer.size, model.tokenizer.sha256)
    return ModelPaths(weights, tokenizer)


def fetch_model(key: str, log: Log = print) -> ModelPaths:
    from huggingface_hub import hf_hub_download

    model = MODELS[key]
    paths = []
    for item in (model.weights, model.tokenizer):
        local = Path(hf_hub_download(model.repo, item.filename, revision=model.revision))
        _verify(local, item.size, item.sha256)
        log(f"verified   {model.repo}@{model.revision[:8]}/{item.filename}")
        paths.append(local)
    return ModelPaths(paths[0], paths[1])


def require_model(key: str) -> ModelPaths:
    found = locate_model(key)
    if found is None:
        raise FileNotFoundError(
            f"model '{key}' is not in the Hugging Face cache; run `raptree fetch`"
        )
    return found
