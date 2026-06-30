"""Run manifest: tracks pipeline stage progress for a single experiment output directory."""

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

STAGE_ORDER = ["prepare", "segment-2d", "segment-3d", "track", "mcherry", "extract"]
TERMINAL_STATUSES = {"completed", "failed", "skipped"}
MANIFEST_FILENAME = "manifest.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _hash_config(config: Dict[str, Any]) -> str:
    serialised = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode()).hexdigest()


@dataclass
class StageRecord:
    status: str = "not_started"
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    output_dir: Optional[str] = None
    config_snapshot: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StageRecord":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class RunManifest:
    def __init__(
        self,
        output_dir: str,
        input_dir: str,
        experiment_id: str,
        created_at: str,
        updated_at: str,
        config_hash: str,
        stages: Dict[str, StageRecord],
    ):
        self.output_dir = output_dir
        self.input_dir = input_dir
        self.experiment_id = experiment_id
        self.created_at = created_at
        self.updated_at = updated_at
        self.config_hash = config_hash
        self.stages = stages

    @classmethod
    def create(
        cls,
        output_dir: str,
        input_dir: str,
        config: Dict[str, Any],
        experiment_id: Optional[str] = None,
    ) -> "RunManifest":
        if experiment_id is None:
            experiment_id = Path(output_dir).name
        now = _now_iso()
        stages = {s: StageRecord() for s in STAGE_ORDER}
        manifest = cls(
            output_dir=str(output_dir),
            input_dir=str(input_dir),
            experiment_id=experiment_id,
            created_at=now,
            updated_at=now,
            config_hash=_hash_config(config),
            stages=stages,
        )
        manifest.save()
        return manifest

    @classmethod
    def load(cls, path: str) -> "RunManifest":
        with open(path) as f:
            data = json.load(f)
        stages = {
            name: StageRecord.from_dict(record)
            for name, record in data.get("stages", {}).items()
        }
        # Ensure all canonical stages exist
        for s in STAGE_ORDER:
            if s not in stages:
                stages[s] = StageRecord()
        return cls(
            output_dir=data["output_dir"],
            input_dir=data["input_dir"],
            experiment_id=data["experiment_id"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            config_hash=data.get("config_hash", ""),
            stages=stages,
        )

    @classmethod
    def load_from_output_dir(cls, output_dir: str) -> "RunManifest":
        return cls.load(Path(output_dir) / MANIFEST_FILENAME)

    def _get_stage(self, stage: str) -> StageRecord:
        if stage not in self.stages:
            self.stages[stage] = StageRecord()
        return self.stages[stage]

    def start_stage(self, stage: str, config: Optional[Dict[str, Any]] = None) -> None:
        record = self._get_stage(stage)
        record.status = "running"
        record.started_at = _now_iso()
        record.completed_at = None
        record.error = None
        if config is not None:
            record.config_snapshot = config
        self.save()

    def complete_stage(
        self,
        stage: str,
        output_dir: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        record = self._get_stage(stage)
        record.status = "completed"
        record.completed_at = _now_iso()
        if output_dir is not None:
            record.output_dir = str(output_dir)
        if metadata is not None:
            record.metadata = metadata
        self.save()

    def fail_stage(self, stage: str, error: Optional[str] = None) -> None:
        record = self._get_stage(stage)
        record.status = "failed"
        record.completed_at = _now_iso()
        if error is not None:
            record.error = error
        self.save()

    def skip_stage(self, stage: str, reason: Optional[str] = None) -> None:
        record = self._get_stage(stage)
        record.status = "skipped"
        record.completed_at = _now_iso()
        if reason is not None:
            record.metadata = {"reason": reason}
        self.save()

    def next_steps(self) -> List[str]:
        return [
            s for s in STAGE_ORDER
            if self.stages.get(s, StageRecord()).status not in TERMINAL_STATUSES
        ]

    def overall_status(self) -> str:
        statuses = [self.stages.get(s, StageRecord()).status for s in STAGE_ORDER]
        if any(s == "failed" for s in statuses):
            failed = [STAGE_ORDER[i] for i, s in enumerate(statuses) if s == "failed"]
            return f"FAILED ({', '.join(failed)})"
        if any(s == "running" for s in statuses):
            running = [STAGE_ORDER[i] for i, s in enumerate(statuses) if s == "running"]
            return f"running: {', '.join(running)}"
        if all(s in TERMINAL_STATUSES for s in statuses):
            return "complete"
        return "in_progress"

    def summary(self) -> str:
        lines = [f"Run: {self.experiment_id}  ({self.overall_status()})"]
        for stage in STAGE_ORDER:
            record = self.stages.get(stage, StageRecord())
            lines.append(f"  {stage:14s} {record.status}")
        remaining = self.next_steps()
        if remaining:
            lines.append(f"Next steps: {' → '.join(remaining)}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "config_hash": self.config_hash,
            "stages": {name: record.to_dict() for name, record in self.stages.items()},
            "next_steps": self.next_steps(),
        }

    def save(self) -> None:
        self.updated_at = _now_iso()
        path = Path(self.output_dir) / MANIFEST_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        try:
            with open(tmp, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
            os.replace(tmp, path)
        except FileNotFoundError:
            # Another process won the race and already renamed the tmp file.
            tmp.unlink(missing_ok=True)
            if not path.exists():
                raise
        except Exception:
            tmp.unlink(missing_ok=True)
            raise


def create_or_load_manifest(
    output_dir: str,
    input_dir: str,
    config: Dict[str, Any],
) -> RunManifest:
    """Load an existing manifest from output_dir, or create a new one."""
    manifest_path = os.path.join(output_dir, MANIFEST_FILENAME)
    if os.path.exists(manifest_path):
        return RunManifest.load(manifest_path)
    return RunManifest.create(output_dir, input_dir, config)


def find_manifests(results_dir: str) -> List[RunManifest]:
    """Recursively find all manifest.json files under results_dir."""
    manifests = []
    for manifest_path in Path(results_dir).rglob(MANIFEST_FILENAME):
        try:
            manifests.append(RunManifest.load(str(manifest_path)))
        except (json.JSONDecodeError, KeyError):
            pass
    return manifests
