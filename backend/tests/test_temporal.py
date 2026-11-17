"""Tests de app.providers.base.overlap_ratio / dedupe_by_overlap (P0.5).

Consolidan lo que antes eran 2 implementaciones idénticas duplicadas
(providers/naive.py::_overlap_ratio para ClipCandidate y
pipeline/gaming_edits.py::_range_overlap para SourceRange) más un wrapper
de dedupe (gaming_edits.py::_dedupe). Refactor puro: mismo comportamiento,
sin cambiar ningún umbral de los call sites existentes.
"""

from app.providers.base import ClipCandidate, SourceRange, dedupe_by_overlap, overlap_ratio


def _cand(start: float, end: float, score: float = 0.5) -> ClipCandidate:
    return ClipCandidate(start=start, end=end, score=score)


def _range(start: float, end: float, score: float = 0.5) -> SourceRange:
    return SourceRange(start=start, end=end, score=score)


def test_overlap_ratio_no_intersection_is_zero():
    a = _cand(0.0, 10.0)
    b = _cand(20.0, 30.0)
    assert overlap_ratio(a, b) == 0.0


def test_overlap_ratio_partial_intersection_matches_hand_calculation():
    # a: [0,10], b: [5,20] -> solape [5,10] = 5s; mas corto = a (10s) -> 5/10 = 0.5
    a = _cand(0.0, 10.0)
    b = _cand(5.0, 20.0)
    assert overlap_ratio(a, b) == 0.5


def test_overlap_ratio_full_containment_is_one():
    # b esta completamente dentro de a -> solape = duracion de b (mas corto) -> 1.0
    a = _cand(0.0, 20.0)
    b = _cand(5.0, 10.0)
    assert overlap_ratio(a, b) == 1.0


def test_overlap_ratio_near_zero_duration_does_not_raise_zero_division():
    a = _cand(10.0, 10.0005)
    b = _cand(10.0, 10.0005)
    # clamp a 0.001 preservado de la formula original
    result = overlap_ratio(a, b)
    assert result >= 0.0


def test_overlap_ratio_identical_formula_for_source_ranges():
    a = _range(0.0, 10.0)
    b = _range(5.0, 20.0)
    assert overlap_ratio(a, b) == 0.5


def test_dedupe_by_overlap_keeps_highest_score_and_respects_ratio():
    high = _cand(0.0, 10.0, score=0.9)
    overlapping_lower = _cand(2.0, 12.0, score=0.6)  # solapa fuerte con "high"
    distinct = _cand(50.0, 60.0, score=0.7)

    result = dedupe_by_overlap([overlapping_lower, high, distinct], ratio=0.3)

    assert high in result
    assert distinct in result
    assert overlapping_lower not in result


def test_dedupe_by_overlap_different_thresholds_give_different_results():
    a = _cand(0.0, 10.0, score=0.9)
    b = _cand(3.0, 13.0, score=0.8)  # overlap_ratio(a,b) = 7/10 = 0.7

    strict = dedupe_by_overlap([a, b], ratio=0.5)  # 0.7 > 0.5 -> b se descarta
    lenient = dedupe_by_overlap([a, b], ratio=0.82)  # 0.7 <= 0.82 -> b sobrevive

    assert strict == [a]
    assert lenient == [a, b]


def test_dedupe_by_overlap_no_overlap_keeps_all():
    a = _cand(0.0, 10.0, score=0.5)
    b = _cand(20.0, 30.0, score=0.9)
    c = _cand(40.0, 50.0, score=0.1)

    result = dedupe_by_overlap([a, b, c], ratio=0.3)

    assert len(result) == 3
    assert result[0].score == 0.9  # ordenado por score desc
    assert {round(cand.start) for cand in result} == {0, 20, 40}
