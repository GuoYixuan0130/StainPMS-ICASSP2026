"""Evaluation primitives shared by StainRoute oracle code."""

from .pq import PQEvaluation, evaluate_pq, matched_iou_sum, pq_factorized

__all__ = ["PQEvaluation", "evaluate_pq", "matched_iou_sum", "pq_factorized"]
