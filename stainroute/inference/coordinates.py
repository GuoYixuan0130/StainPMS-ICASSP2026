"""Coordinate transforms shared by candidate/decode code paths."""

from __future__ import annotations

from stainroute.actions.schema import Point


def crop_box_around_point(point: Point, image_width: int, image_height: int, crop_size: int) -> tuple[int, int, int, int]:
    if image_width <= 0 or image_height <= 0 or crop_size <= 0:
        raise ValueError("image dimensions and crop_size must be positive")
    if not 0 <= point.x < image_width or not 0 <= point.y < image_height:
        raise ValueError(f"Point {point} is outside image {image_width}x{image_height}")
    crop_width = min(crop_size, image_width)
    crop_height = min(crop_size, image_height)
    x1 = max(0, min(image_width - crop_width, int(round(point.x - crop_width / 2))))
    y1 = max(0, min(image_height - crop_height, int(round(point.y - crop_height / 2))))
    return x1, y1, x1 + crop_width, y1 + crop_height


def global_to_crop(point: Point, crop_box: tuple[int, int, int, int]) -> Point:
    x1, y1, x2, y2 = crop_box
    if not x1 <= point.x < x2 or not y1 <= point.y < y2:
        raise ValueError(f"Point {point} is outside crop {crop_box}")
    return Point(point.x - x1, point.y - y1)


def crop_to_global(point: Point, crop_box: tuple[int, int, int, int]) -> Point:
    x1, y1, _, _ = crop_box
    return Point(point.x + x1, point.y + y1)
