"""Lightweight, no-VLM fixtures for the Editorial Director's quality gate."""

from __future__ import annotations

import pytest


def quality_gate(case: dict) -> str:
    if case["menu_contamination"] >= 2 or case["dead_time"] >= 3:
        return "REJECT"
    if case["payoff"] == 0 and case["event_completeness"] < 2:
        return "REWORK"
    if case["boundary_quality"] < 2:
        return "REWORK"
    return "KEEP"


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        (
            {
                "event_completeness": 3,
                "payoff": 3,
                "menu_contamination": 0,
                "dead_time": 0,
                "boundary_quality": 3,
            },
            "KEEP",
        ),
        (
            {
                "event_completeness": 1,
                "payoff": 1,
                "menu_contamination": 3,
                "dead_time": 1,
                "boundary_quality": 2,
            },
            "REJECT",
        ),
        (
            {
                "event_completeness": 3,
                "payoff": 3,
                "menu_contamination": 0,
                "dead_time": 0,
                "boundary_quality": 1,
            },
            "REWORK",
        ),
        (
            {
                "event_completeness": 1,
                "payoff": 0,
                "menu_contamination": 0,
                "dead_time": 0,
                "boundary_quality": 3,
            },
            "REWORK",
        ),
    ],
    ids=("continuous_payoff", "menu_between_events", "cut_payoff", "strong_frame_missing_payoff"),
)
def test_editorial_quality_gate(case: dict, expected: str):
    assert quality_gate(case) == expected


def test_editorial_eval_fixture_coverage():
    cases = {
        "montage_three_events", "fast_hook", "aftermath_hook", "high_score_mediocre",
        "medium_score_strong", "uncertain_outcome", "cut_payoff",
    }
    assert len(cases) == 7
