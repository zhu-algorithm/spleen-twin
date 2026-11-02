"""Reproducibility and scientific-claim metadata for simulation runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class EvidenceClass(str, Enum):
    MEASURED = "measured"
    LITERATURE = "literature"
    INFERRED = "inferred"
    CALIBRATED = "calibrated"
    SURROGATE = "surrogate"
    MODEL_KNOB = "model_knob"


@dataclass(frozen=True)
class ParameterRecord:
    name: str
    value: Any
    unit: str
    evidence: EvidenceClass
    source: str
    consumer: str


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class RunManifest:
    seed: int
    config: dict[str, Any]
    parameters: list[ParameterRecord]
    inputs: dict[str, str]
    created_at: str = ""
    package_version: str = ""
    python_version: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if not self.package_version:
            try:
                self.package_version = importlib.metadata.version("spleen-twin")
            except importlib.metadata.PackageNotFoundError:
                self.package_version = "source"
        if not self.python_version:
            self.python_version = platform.python_version()

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for p in out["parameters"]:
            if isinstance(p["evidence"], EvidenceClass):
                p["evidence"] = p["evidence"].value
        return out

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")


__all__ = ["EvidenceClass", "ParameterRecord", "RunManifest", "sha256_file"]
