"""Explicit action schema with hard separation of feature/label namespaces."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping


class ActionType(str, Enum):
    ADD = "ADD"
    SPLIT = "SPLIT"


@dataclass(frozen=True, order=True)
class Point:
    x: int
    y: int

    def as_dict(self) -> dict[str, int]:
        return {"x": int(self.x), "y": int(self.y)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Point":
        return cls(x=int(payload["x"]), y=int(payload["y"]))


FORBIDDEN_PREDECODE_TOKENS = (
    "gt",
    "groundtruth",
    "ground_truth",
    "target",
    "delta",
    "utility",
    "positive",
    "harm",
    "matched_iou",
    "tp",
    "fp",
    "fn",
    "aji",
)


def _walk_keys(value: Any, prefix: str = "") -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            yield name
            yield from _walk_keys(child, name)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _walk_keys(child, f"{prefix}[{index}]")


def assert_no_gt_leakage(features: Mapping[str, Any]) -> None:
    """Reject forbidden label-derived fields from pre-decode feature payloads."""

    offenders = []
    for key in _walk_keys(features):
        normalized = key.lower().replace("-", "_")
        if any(token in normalized for token in FORBIDDEN_PREDECODE_TOKENS):
            offenders.append(key)
    if offenders:
        raise ValueError(f"Forbidden GT/utility-derived pre-decode fields: {sorted(offenders)}")


def _box_or_none(value: tuple[int, int, int, int] | None) -> list[int] | None:
    return list(value) if value is not None else None


@dataclass(frozen=True)
class ActionCandidate:
    action_id: str
    image_id: str
    action_type: ActionType
    affected_instance_ids: tuple[int, ...]
    positive_points: tuple[Point, ...]
    negative_points: tuple[Point, ...]
    action_cost: int
    generation_features: Mapping[str, Any] = field(default_factory=dict)
    decoded_features: Mapping[str, Any] = field(default_factory=dict)
    utility_fields: Mapping[str, Any] = field(default_factory=dict)
    conflict_ids: tuple[str, ...] = ()
    generator_version: str = "stainroute-v1"
    config_hash: str = ""
    tile_box: tuple[int, int, int, int] | None = None
    support_box: tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        if not self.action_id:
            raise ValueError("action_id is required")
        if not self.image_id:
            raise ValueError("image_id is required")
        if self.action_cost <= 0:
            raise ValueError("action_cost must be positive")
        if self.action_type is ActionType.ADD:
            if len(self.positive_points) != 1 or self.negative_points:
                raise ValueError("ADD requires exactly one positive point and no negative points")
            if self.action_cost != 1:
                raise ValueError("ADD action_cost must be 1")
        if self.action_type is ActionType.SPLIT:
            if len(self.positive_points) != 2 or len(self.negative_points) != 2:
                raise ValueError("SPLIT requires two positive and two mutual-negative points")
            if len(self.affected_instance_ids) != 1:
                raise ValueError("SPLIT requires exactly one parent predicted instance")
            if self.action_cost != 2:
                raise ValueError("SPLIT action_cost must be 2")
        assert_no_gt_leakage(self.generation_features)
        for name, box in (("tile_box", self.tile_box), ("support_box", self.support_box)):
            if box is not None and (len(box) != 4 or box[2] < box[0] or box[3] < box[1]):
                raise ValueError(f"Invalid {name}: {box}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "image_id": self.image_id,
            "action_type": self.action_type.value,
            "affected_instance_ids": list(self.affected_instance_ids),
            "positive_points": [point.as_dict() for point in self.positive_points],
            "negative_points": [point.as_dict() for point in self.negative_points],
            "action_cost": self.action_cost,
            "generation_features": dict(self.generation_features),
            "decoded_features": dict(self.decoded_features),
            "utility_fields": dict(self.utility_fields),
            "conflict_ids": list(self.conflict_ids),
            "generator_version": self.generator_version,
            "config_hash": self.config_hash,
            "tile_box": _box_or_none(self.tile_box),
            "support_box": _box_or_none(self.support_box),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActionCandidate":
        def box(name: str) -> tuple[int, int, int, int] | None:
            value = payload.get(name)
            return tuple(int(item) for item in value) if value is not None else None

        return cls(
            action_id=str(payload["action_id"]),
            image_id=str(payload["image_id"]),
            action_type=ActionType(str(payload["action_type"])),
            affected_instance_ids=tuple(int(item) for item in payload.get("affected_instance_ids", [])),
            positive_points=tuple(Point.from_dict(item) for item in payload.get("positive_points", [])),
            negative_points=tuple(Point.from_dict(item) for item in payload.get("negative_points", [])),
            action_cost=int(payload["action_cost"]),
            generation_features=dict(payload.get("generation_features", {})),
            decoded_features=dict(payload.get("decoded_features", {})),
            utility_fields=dict(payload.get("utility_fields", {})),
            conflict_ids=tuple(str(item) for item in payload.get("conflict_ids", [])),
            generator_version=str(payload.get("generator_version", "stainroute-v1")),
            config_hash=str(payload.get("config_hash", "")),
            tile_box=box("tile_box"),
            support_box=box("support_box"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str) -> "ActionCandidate":
        return cls.from_dict(json.loads(payload))
