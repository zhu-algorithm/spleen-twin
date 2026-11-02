"""Crash-safe, replayable campaign registry and acquisition audit log."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class CampaignRound:
    round_index: int
    input_checksum: str
    n_curated: int
    n_model: int
    n_baseline: int
    metrics: dict
    failures: list[dict] = field(default_factory=list)
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")


class CampaignRegistry:
    schema_version = "campaign-registry-1"

    def __init__(self, campaign_id: str):
        self.campaign_id = campaign_id
        self.rounds: list[CampaignRound] = []

    def add(self, round_record: CampaignRound) -> None:
        if self.rounds and round_record.round_index <= self.rounds[-1].round_index:
            raise ValueError("round indices must be strictly increasing")
        self.rounds.append(round_record)

    def to_dict(self) -> dict:
        return {"schema_version": self.schema_version, "campaign_id": self.campaign_id,
                "rounds": [asdict(r) for r in self.rounds]}

    def write(self, path: str | Path) -> None:
        target = Path(path)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        temporary.replace(target)

    @classmethod
    def read(cls, path: str | Path) -> "CampaignRegistry":
        data = json.loads(Path(path).read_text())
        if data.get("schema_version") != cls.schema_version:
            raise ValueError("unsupported campaign registry schema")
        obj = cls(data["campaign_id"])
        obj.rounds = [CampaignRound(**r) for r in data["rounds"]]
        return obj


__all__ = ["CampaignRound", "CampaignRegistry"]
