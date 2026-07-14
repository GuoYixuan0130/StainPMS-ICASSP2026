"""Small, dependency-light helpers shared by the ResiMix stage driver.

Nothing in this module opens a dataset or a checkpoint unless a caller passes
an explicit path.  This is intentional: split isolation and checkpoint
identity are preconditions of a ResiMix run, not optional conveniences.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA256 for one explicitly named file."""
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(path: str | Path, expected: str, label: str) -> str:
    """Fail closed when an explicitly frozen input is absent or changed."""
    candidate = Path(path)
    if not candidate.is_file():
        raise FileNotFoundError(f"{label} is missing: {candidate}")
    actual = sha256_file(candidate)
    if actual.lower() != expected.lower():
        raise ValueError(
            f"{label} SHA256 mismatch: expected {expected.lower()}, got {actual.lower()}"
        )
    return actual


def parse_epoch_schedule(value: str | Sequence[int], total_epochs: int) -> tuple[int, ...]:
    """Parse and validate pre-registered evaluation checkpoints.

    Epoch zero is the frozen initialisation before any optimizer step; a
    positive number denotes the state after that many completed epochs.
    """
    if isinstance(value, str):
        pieces = [item.strip() for item in value.split(",") if item.strip()]
        values = [int(item) for item in pieces]
    else:
        values = [int(item) for item in value]
    schedule = tuple(sorted(set(values)))
    if not schedule:
        return ()
    if schedule[0] < 0 or schedule[-1] > int(total_epochs):
        raise ValueError(
            f"evaluation epochs {schedule} must be within [0, {total_epochs}]"
        )
    return schedule


@dataclass(frozen=True)
class EvaluationSchedule:
    total_epochs: int
    evaluation_epochs: tuple[int, ...]

    @classmethod
    def from_cli(cls, total_epochs: int, raw: str) -> "EvaluationSchedule":
        return cls(int(total_epochs), parse_epoch_schedule(raw, int(total_epochs)))

    def should_evaluate(self, completed_epochs: int) -> bool:
        return int(completed_epochs) in self.evaluation_epochs


def write_json(path: str | Path, payload: Mapping) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def append_csv(path: str | Path, row: Mapping[str, object], fieldnames: Iterable[str]) -> None:
    """Append a stable-schema CSV row, rejecting accidental schema drift."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    names = list(fieldnames)
    exists = destination.is_file()
    if exists:
        with destination.open("r", newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle), [])
        if header != names:
            raise ValueError(f"CSV schema mismatch for {destination}: {header} != {names}")
    with destination.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="raise")
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in names})


def make_checkpoint_manifest_entry(
    *, label: str, path: str | Path, expected_sha256: str, actual_sha256: str
) -> dict[str, str]:
    return {
        "label": str(label),
        "path": str(Path(path)),
        "expected_sha256": str(expected_sha256).lower(),
        "actual_sha256": str(actual_sha256).lower(),
    }
