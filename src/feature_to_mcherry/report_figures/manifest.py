"""Per-figure status tracking and the run manifest (manifest.json/manifest.md)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List


@dataclass
class FigureStatus:
    """Outcome of attempting to generate one figure.

    ``status``: ``"generated"`` (fully built), ``"partial"`` (built with some
    experiments/inputs missing), or ``"blocked"`` (not built at all).
    """

    figure_id: str
    title: str
    status: str
    output_paths: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    note: str = ""

    def __post_init__(self) -> None:
        if self.status not in ("generated", "partial", "blocked"):
            raise ValueError(
                "status must be 'generated', 'partial', or 'blocked', got "
                f"{self.status!r}"
            )


def _to_markdown_table(rows: List[FigureStatus]) -> str:
    header = "| figure_id | title | status | missing | note |"
    separator = "| --- | --- | --- | --- | --- |"
    body = [
        f"| {r.figure_id} | {r.title} | {r.status} | {', '.join(r.missing)} "
        f"| {r.note} |"
        for r in rows
    ]
    return "\n".join([header, separator, *body])


class ManifestWriter:
    """Collects :class:`FigureStatus` entries across a report-figures run."""

    def __init__(self) -> None:
        self._entries: List[FigureStatus] = []

    def add(self, status: FigureStatus) -> None:
        self._entries.append(status)

    @property
    def entries(self) -> List[FigureStatus]:
        return list(self._entries)

    def write(self, output_dir: Path) -> None:
        """Write ``manifest.json`` and ``manifest.md`` into ``output_dir``."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / "manifest.json"
        json_path.write_text(
            json.dumps([asdict(e) for e in self._entries], indent=2, default=str)
        )

        md_path = output_dir / "manifest.md"
        lines = [
            "# Report figures — generation manifest",
            "",
            _to_markdown_table(self._entries),
            "",
        ]
        md_path.write_text("\n".join(lines))
