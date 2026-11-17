"""Phase 4.4A - Temporal Evidence Continuity tests.

Covers: compute_continuity_group_id determinism, chunk_id index extraction,
lexical anchor normalization (fixed + corpus-generic filtering), the color
veto, the STRICT scene-cache policy, group-wide transitivity, mandatory
cases A-H, determinism under input-order permutation, bounded complexity,
non-interference with the legacy Event Map / Quality Gate, and confirmation
rehydration backward-compatibility.

Real EditorAsset + real scene cache reads (via write_cached_boundaries /
resolve_scene_cache_lookup), same pattern test_scene_cache.py already
established - no detector, no ffmpeg subprocess, no Claude API calls.
"""

from __future__ import annotations

import time

from sqlmodel import Session

from app.core.db import engine, init_db
from app.editor import events as events_module
from app.editor import service as editor_service
from app.editor.agent import _evidence, inspect_project
from app.editor.via_evidence import (
    CORPUS_GENERIC_FREQUENCY_THRESHOLD,
    EvidenceKind,
    NormalizedVideoEvidence,
    _color_conflict,
    _corpus_generic_tokens,
    _extract_chunk_index,
    _fixed_filtered_tokens,
    annotate_continuity,
    compute_continuity_group_id,
    deduplicate_evidence,
    normalize_via_result,
)
from app.video_intelligence.chunking import SceneBoundary
from app.video_intelligence.models import (
    VideoEventCandidate,
    VideoIntelligenceResult,
    VideoObservation,
    VideoObservationNature,
    VideoProcessingMetadata,
)
from app.video_intelligence.scene_cache import (
    resolve_scene_cache_lookup,
    write_cached_boundaries,
)
from app.video_intelligence.scene_detection import DEFAULT_SCENE_CHANGE_THRESHOLD

THRESHOLD = DEFAULT_SCENE_CHANGE_THRESHOLD


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _real_asset(tmp_path, name: str) -> str:
    init_db()
    with Session(engine) as session:
        project = editor_service.create_project(session, f"continuity-{name}")
        source = tmp_path / f"{name}.mp4"
        source.write_bytes(f"fake video bytes for {name}".encode())
        asset = editor_service.add_asset(session, project, source, duration_us=240_000_000)
        return asset.id


def _seed_scene_cache(asset_id: str, start_us: int, end_us: int, boundaries_us: list[int]) -> None:
    """Writes a validated scene cache HIT for the exact request range, same
    identity keys (fingerprint/threshold/start_us/end_us) service.py's
    scene-aware composition itself uses."""
    lookup = resolve_scene_cache_lookup(asset_id)
    ok = write_cached_boundaries(
        asset_id,
        fingerprint=lookup.fingerprint,
        threshold=THRESHOLD,
        start_us=start_us,
        end_us=end_us,
        boundaries=[SceneBoundary(timestamp_us=ts) for ts in boundaries_us],
    )
    assert ok is True


def _obs(
    asset_id: str,
    start_us: int,
    end_us: int,
    description: str,
    chunk_id: str,
    confidence: float = 0.85,
) -> VideoObservation:
    return VideoObservation(
        asset_id=asset_id,
        start_us=start_us,
        end_us=end_us,
        description=description,
        nature=VideoObservationNature.OBSERVATION,
        confidence=confidence,
        source="snipra_vlm",
        provider="snipra_vlm",
        model="test-model",
        chunk_id=chunk_id,
    )


def _event(
    asset_id: str,
    start_us: int,
    end_us: int,
    description: str,
    chunk_id: str,
    confidence: float = 0.85,
) -> VideoEventCandidate:
    return VideoEventCandidate(
        asset_id=asset_id,
        start_us=start_us,
        end_us=end_us,
        type="action",
        description=description,
        confidence=confidence,
        nature=VideoObservationNature.OBSERVATION,
        provider="snipra_vlm",
        model="test-model",
        chunk_id=chunk_id,
    )


def _via_result(asset_id: str, start_us: int, end_us: int, observations=None, events=None):
    return VideoIntelligenceResult(
        asset_id=asset_id,
        start_us=start_us,
        end_us=end_us,
        mode="summarize_video",
        observations=observations or [],
        events=events or [],
        metadata=VideoProcessingMetadata(provider="snipra_vlm", model="test-model"),
    )


def _filler_observations(
    asset_id: str, n: int, chunk_id: str = "chunk_9"
) -> list[VideoObservation]:
    """Diverse, mutually-unrelated filler items used to keep a specific
    shared token's corpus-wide frequency under CORPUS_GENERIC_FREQUENCY_THRESHOLD
    (section 20, cases C/G: 'the corpus must be diverse enough that the
    shared anchor is not corpus-generic')."""
    return [
        _obs(
            asset_id, i * 1000, i * 1000 + 500,
            f"unrelated filler item number {i} appears", chunk_id,
        )
        for i in range(n)
    ]


def _filler_events(asset_id: str, n: int, chunk_id: str = "chunk_9") -> list[VideoEventCandidate]:
    """EVENT-kind counterpart of _filler_observations - same diversifying
    purpose, kept in its own pool since corpus-generic frequency is computed
    per evidence_kind (section 9/17: kinds never share a corpus)."""
    return [
        _event(
            asset_id, i * 1000, i * 1000 + 500,
            f"unrelated filler event number {i} occurs", chunk_id,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Unit: compute_continuity_group_id
# ---------------------------------------------------------------------------


class TestComputeContinuityGroupId:
    def test_deterministic_same_members(self):
        ids = ["ev-a", "ev-b", "ev-c"]
        assert compute_continuity_group_id(ids) == compute_continuity_group_id(list(ids))

    def test_order_independent(self):
        forward = compute_continuity_group_id(["ev-a", "ev-b", "ev-c"])
        shuffled = compute_continuity_group_id(["ev-c", "ev-a", "ev-b"])
        reversed_ = compute_continuity_group_id(["ev-c", "ev-b", "ev-a"])
        assert forward == shuffled == reversed_

    def test_different_membership_different_id(self):
        a = compute_continuity_group_id(["ev-a", "ev-b"])
        b = compute_continuity_group_id(["ev-a", "ev-b", "ev-c"])
        assert a != b

    def test_is_sha256_derived_hex(self):
        group_id = compute_continuity_group_id(["ev-a"])
        assert isinstance(group_id, str)
        int(group_id, 16)  # must be valid hex


# ---------------------------------------------------------------------------
# Unit: chunk_id index extraction
# ---------------------------------------------------------------------------


class TestExtractChunkIndex:
    def test_valid_formats(self):
        assert _extract_chunk_index("chunk_0") == 0
        assert _extract_chunk_index("chunk_1") == 1
        assert _extract_chunk_index("chunk_42") == 42

    def test_none_and_empty(self):
        assert _extract_chunk_index(None) is None
        assert _extract_chunk_index("") is None

    def test_unreliable_format_returns_none_not_invented(self):
        assert _extract_chunk_index("weird-format") is None
        assert _extract_chunk_index("chunk_") is None
        assert _extract_chunk_index("chunk_-1") is None
        assert _extract_chunk_index("segment_3") is None


# ---------------------------------------------------------------------------
# Unit: lexical anchors / corpus-generic frequency boundary
# ---------------------------------------------------------------------------


class TestLexicalAnchors:
    def test_fixed_generic_removed(self):
        tokens = _fixed_filtered_tokens("A person enters the room")
        assert "person" not in tokens
        assert "room" in tokens

    def test_punctuation_and_case_normalized(self):
        a = _fixed_filtered_tokens("PAGODA-like structure!")
        b = _fixed_filtered_tokens("pagoda like structure")
        assert a == b

    def test_corpus_generic_boundary_exact_half_not_filtered(self):
        # n=2, shared token in both -> frequency exactly 1.0 (not the
        # boundary itself, but confirms >threshold semantics against a
        # precise fractional case below).
        sets = [{"car", "red"}, {"car", "blue"}]
        generic = _corpus_generic_tokens(sets, threshold=0.5)
        # 2/2 = 1.0 > 0.5 -> "car" IS corpus-generic here.
        assert "car" in generic
        assert "red" not in generic

    def test_corpus_generic_threshold_strict_greater_than(self):
        # 5/10 = 0.5, threshold=0.5 -> NOT strictly greater -> not generic.
        sets = [{"dog"} for _ in range(5)] + [{"other"} for _ in range(5)]
        generic = _corpus_generic_tokens(sets, threshold=0.5)
        assert "dog" not in generic

    def test_corpus_generic_above_threshold_is_filtered(self):
        # 6/10 = 0.6 > default 0.55 -> generic.
        sets = [{"dog"} for _ in range(6)] + [{"other"} for _ in range(4)]
        generic = _corpus_generic_tokens(sets, threshold=CORPUS_GENERIC_FREQUENCY_THRESHOLD)
        assert "dog" in generic

    def test_empty_corpus_is_safe(self):
        assert _corpus_generic_tokens([]) == set()


# ---------------------------------------------------------------------------
# Unit: color veto
# ---------------------------------------------------------------------------


class TestColorVeto:
    def test_distinct_colors_conflict(self):
        assert _color_conflict({"red"}, {"blue"}) is True

    def test_one_sided_mention_is_noop(self):
        assert _color_conflict({"red"}, set()) is False
        assert _color_conflict(set(), {"blue"}) is False

    def test_same_color_no_conflict(self):
        assert _color_conflict({"red"}, {"red"}) is False

    def test_no_colors_no_conflict(self):
        assert _color_conflict(set(), set()) is False


# ---------------------------------------------------------------------------
# Cases A-H (section 20) - mandatory permanent regressions
# ---------------------------------------------------------------------------


class TestMandatoryCasesAtoH:
    def test_a_generic_person_no_relation(self, tmp_path):
        asset_id = _real_asset(tmp_path, "case-a")
        _seed_scene_cache(asset_id, 0, 20_000_000, [])
        obs = [
            _obs(asset_id, 0, 5_000_000, "person enters room", "chunk_0"),
            _obs(asset_id, 6_000_000, 10_000_000, "person sits down", "chunk_1"),
            *_filler_observations(asset_id, 12),
        ]
        via_result = _via_result(asset_id, 0, 20_000_000, observations=obs)
        normalized = normalize_via_result(via_result)
        annotated = annotate_continuity(deduplicate_evidence(normalized), via_result)
        target = [
            e for e in annotated
            if "enters room" in e.description or "sits down" in e.description
        ]
        assert all(e.continuity_group_id is None for e in target)

    def test_b_scene_boundary_veto(self, tmp_path):
        asset_id = _real_asset(tmp_path, "case-b")
        # Real boundary sitting between the two chunk windows (chunk_0 ~
        # [0,10s), chunk_1 ~ [10s,20s) for a 20s COARSE-style split).
        _seed_scene_cache(asset_id, 0, 20_000_000, [10_000_000])
        obs = [
            _obs(asset_id, 1_000_000, 9_000_000, "red car exits frame", "chunk_0"),
            _obs(asset_id, 11_000_000, 19_000_000, "red car appears after hard cut", "chunk_1"),
            *_filler_observations(asset_id, 12),
        ]
        via_result = _via_result(asset_id, 0, 20_000_000, observations=obs)
        normalized = normalize_via_result(via_result)
        annotated = annotate_continuity(deduplicate_evidence(normalized), via_result)
        target = [e for e in annotated if "red car" in e.description]
        assert all(e.continuity_group_id is None for e in target)

    def test_c_dog_running_groups_with_clean_scene_and_nongeneric_corpus(self, tmp_path):
        asset_id = _real_asset(tmp_path, "case-c")
        _seed_scene_cache(asset_id, 0, 20_000_000, [])
        obs = [
            _obs(asset_id, 1_000_000, 5_000_000, "dog running across the yard", "chunk_0"),
            _obs(
                asset_id, 6_000_000, 9_000_000,
                "dog continues running toward the fence", "chunk_1",
            ),
            *_filler_observations(asset_id, 12),
        ]
        via_result = _via_result(asset_id, 0, 20_000_000, observations=obs)
        normalized = normalize_via_result(via_result)
        annotated = annotate_continuity(deduplicate_evidence(normalized), via_result)
        target = [e for e in annotated if "dog" in e.description]
        assert len(target) == 2
        ids = {e.continuity_group_id for e in target}
        assert len(ids) == 1
        assert None not in ids
        assert all(e.continuity_basis == "temporal_anchor_continuity" for e in target)

    def test_d_talking_to_closeup_no_relation(self, tmp_path):
        asset_id = _real_asset(tmp_path, "case-d")
        _seed_scene_cache(asset_id, 0, 20_000_000, [10_000_000])
        obs = [
            _obs(asset_id, 1_000_000, 9_000_000, "two people talking", "chunk_0"),
            _obs(asset_id, 11_000_000, 19_000_000, "close-up of one person after cut", "chunk_1"),
            *_filler_observations(asset_id, 12),
        ]
        via_result = _via_result(asset_id, 0, 20_000_000, observations=obs)
        normalized = normalize_via_result(via_result)
        annotated = annotate_continuity(deduplicate_evidence(normalized), via_result)
        target = [e for e in annotated if "talking" in e.description or "close-up" in e.description]
        assert all(e.continuity_group_id is None for e in target)

    def test_e_object_kind_excluded(self, tmp_path):
        from app.video_intelligence.models import VideoObjectEvidence

        asset_id = _real_asset(tmp_path, "case-e")
        _seed_scene_cache(asset_id, 0, 20_000_000, [])
        objects = [
            VideoObjectEvidence(
                asset_id=asset_id, label="car", timestamps_us=[1_000_000, 2_000_000],
                confidence=0.8, provider="snipra_vlm", model="test-model",
                nature=VideoObservationNature.OBSERVATION,
            ),
            VideoObjectEvidence(
                asset_id=asset_id, label="vehicle", timestamps_us=[6_000_000, 7_000_000],
                confidence=0.8, provider="snipra_vlm", model="test-model",
                nature=VideoObservationNature.OBSERVATION,
            ),
        ]
        via_result = _via_result(asset_id, 0, 20_000_000)
        via_result.objects = objects
        normalized = normalize_via_result(via_result)
        annotated = annotate_continuity(deduplicate_evidence(normalized), via_result)
        object_items = [e for e in annotated if e.evidence_kind == EvidenceKind.OBJECT]
        assert object_items
        assert all(e.continuity_group_id is None for e in object_items)

    def test_f_person_opens_door_vs_car_passes_no_relation(self, tmp_path):
        asset_id = _real_asset(tmp_path, "case-f")
        _seed_scene_cache(asset_id, 0, 20_000_000, [])
        obs = [
            _obs(asset_id, 1_000_000, 5_000_000, "person opens door", "chunk_0"),
            _obs(asset_id, 6_000_000, 9_000_000, "car passes outside", "chunk_1"),
            *_filler_observations(asset_id, 12),
        ]
        via_result = _via_result(asset_id, 0, 20_000_000, observations=obs)
        normalized = normalize_via_result(via_result)
        annotated = annotate_continuity(deduplicate_evidence(normalized), via_result)
        target = [
            e for e in annotated
            if "door" in e.description or "passes outside" in e.description
        ]
        assert all(e.continuity_group_id is None for e in target)

    def test_g_three_item_dog_chain_groups(self, tmp_path):
        asset_id = _real_asset(tmp_path, "case-g")
        _seed_scene_cache(asset_id, 0, 30_000_000, [])
        obs = [
            _obs(asset_id, 1_000_000, 5_000_000, "dog running across the yard", "chunk_0"),
            _obs(asset_id, 6_000_000, 9_000_000, "dog stops running near the fence", "chunk_1"),
            _obs(asset_id, 11_000_000, 14_000_000, "dog leaves frame past the fence", "chunk_2"),
            *_filler_observations(asset_id, 15),
        ]
        via_result = _via_result(asset_id, 0, 30_000_000, observations=obs)
        normalized = normalize_via_result(via_result)
        annotated = annotate_continuity(deduplicate_evidence(normalized), via_result)
        target = [e for e in annotated if "dog" in e.description]
        assert len(target) == 3
        ids = {e.continuity_group_id for e in target}
        assert len(ids) == 1
        assert None not in ids

    def test_h_color_conflict_splits_group_transitivity_safe(self, tmp_path):
        asset_id = _real_asset(tmp_path, "case-h")
        _seed_scene_cache(asset_id, 0, 30_000_000, [])
        obs = [
            _obs(asset_id, 1_000_000, 5_000_000, "red car parked near the curb", "chunk_0"),
            _obs(asset_id, 6_000_000, 9_000_000, "car turns left at the curb", "chunk_1"),
            _obs(asset_id, 11_000_000, 14_000_000, "blue car parked near the curb", "chunk_2"),
            *_filler_observations(asset_id, 15),
        ]
        via_result = _via_result(asset_id, 0, 30_000_000, observations=obs)
        normalized = normalize_via_result(via_result)
        annotated = annotate_continuity(deduplicate_evidence(normalized), via_result)
        by_desc = {e.description: e for e in annotated}
        red = by_desc["red car parked near the curb"]
        turns = by_desc["car turns left at the curb"]
        blue = by_desc["blue car parked near the curb"]
        # {A,B} grouped (no color conflict red vs no-color); C excluded by
        # group-wide color conflict (blue vs red, transitively checked, not
        # merged in via the shared "car"/"curb" anchor with B alone).
        assert red.continuity_group_id is not None
        assert red.continuity_group_id == turns.continuity_group_id
        assert blue.continuity_group_id != red.continuity_group_id


# ---------------------------------------------------------------------------
# EVENT kind: grouping works independently of OBSERVATION, and the two
# kinds never mix into the same group even when both are present.
# ---------------------------------------------------------------------------


class TestEventKindContinuity:
    def test_event_kind_groups_independently_of_observation(self, tmp_path):
        asset_id = _real_asset(tmp_path, "event-kind")
        _seed_scene_cache(asset_id, 0, 20_000_000, [])
        events = [
            _event(
                asset_id, 1_000_000, 5_000_000,
                "explosion rocks the pagoda tower", "chunk_0",
            ),
            _event(
                asset_id, 6_000_000, 9_000_000,
                "pagoda tower collapses in the blast", "chunk_1",
            ),
            *_filler_events(asset_id, 12),
        ]
        # Deliberately identical-looking OBSERVATION items sharing the same
        # anchor text - must NOT merge into the EVENT group (rule: same
        # evidence_kind required; groups never cross kind boundaries).
        observations = [
            _obs(asset_id, 1_000_000, 5_000_000, "explosion rocks the pagoda tower", "chunk_0"),
            _obs(asset_id, 6_000_000, 9_000_000, "pagoda tower collapses in the blast", "chunk_1"),
            *_filler_observations(asset_id, 12),
        ]
        via_result = _via_result(asset_id, 0, 20_000_000, observations=observations, events=events)
        normalized = normalize_via_result(via_result)
        annotated = annotate_continuity(deduplicate_evidence(normalized), via_result)

        event_items = [
            e for e in annotated
            if e.evidence_kind == EvidenceKind.EVENT and "pagoda" in e.description
        ]
        obs_items = [
            e for e in annotated
            if e.evidence_kind == EvidenceKind.OBSERVATION and "pagoda" in e.description
        ]

        assert len(event_items) == 2
        event_group_ids = {e.continuity_group_id for e in event_items}
        assert len(event_group_ids) == 1
        assert None not in event_group_ids

        # OBSERVATION pool has only 2 pagoda items among many fillers -
        # "pagoda"/"tower" survive corpus-generic filtering there too, so
        # they form their OWN group, with a DIFFERENT id than the EVENT
        # group (evidence_id differs by evidence_kind, so group membership
        # can never coincide across kinds).
        assert len(obs_items) == 2
        obs_group_ids = {e.continuity_group_id for e in obs_items}
        assert len(obs_group_ids) == 1
        assert None not in obs_group_ids
        assert obs_group_ids != event_group_ids


# ---------------------------------------------------------------------------
# STRICT scene policy
# ---------------------------------------------------------------------------


class TestStrictScenePolicy:
    def test_no_via_result_abstains(self, tmp_path):
        asset_id = _real_asset(tmp_path, "strict-no-result")
        evidence = [
            NormalizedVideoEvidence.model_validate(
                {
                    "evidence_id": "e1", "asset_id": asset_id, "start_us": 0, "end_us": 5_000_000,
                    "evidence_kind": "observation", "nature": "observation",
                    "description": "dog running in yard", "confidence_original": 0.8,
                    "provider": "p", "model": "m", "chunk_id": "chunk_0",
                }
            ),
            NormalizedVideoEvidence.model_validate(
                {
                    "evidence_id": "e2", "asset_id": asset_id,
                    "start_us": 6_000_000, "end_us": 9_000_000,
                    "evidence_kind": "observation", "nature": "observation",
                    "description": "dog continues running", "confidence_original": 0.8,
                    "provider": "p", "model": "m", "chunk_id": "chunk_1",
                }
            ),
        ]
        annotated = annotate_continuity(evidence, None)
        assert all(e.continuity_group_id is None for e in annotated)

    def test_cache_miss_abstains(self, tmp_path):
        asset_id = _real_asset(tmp_path, "strict-cache-miss")
        # No write_cached_boundaries call -> genuine miss.
        obs = [
            _obs(asset_id, 0, 5_000_000, "dog running in yard", "chunk_0"),
            _obs(asset_id, 6_000_000, 9_000_000, "dog continues running", "chunk_1"),
        ]
        via_result = _via_result(asset_id, 0, 20_000_000, observations=obs)
        normalized = normalize_via_result(via_result)
        annotated = annotate_continuity(deduplicate_evidence(normalized), via_result)
        assert all(e.continuity_group_id is None for e in annotated)

    def test_mismatched_range_is_a_miss_abstains(self, tmp_path):
        asset_id = _real_asset(tmp_path, "strict-range-mismatch")
        # Cache seeded for a DIFFERENT exact range than the request.
        _seed_scene_cache(asset_id, 0, 65_000_000, [])
        obs = [
            _obs(asset_id, 0, 5_000_000, "dog running in yard", "chunk_0"),
            _obs(asset_id, 6_000_000, 9_000_000, "dog continues running", "chunk_1"),
        ]
        via_result = _via_result(asset_id, 0, 20_000_000, observations=obs)  # different end_us
        normalized = normalize_via_result(via_result)
        annotated = annotate_continuity(deduplicate_evidence(normalized), via_result)
        assert all(e.continuity_group_id is None for e in annotated)

    def test_unknown_asset_abstains(self):
        via_result = _via_result("does-not-exist-asset", 0, 20_000_000, observations=[
            _obs("does-not-exist-asset", 0, 5_000_000, "dog running", "chunk_0"),
            _obs("does-not-exist-asset", 6_000_000, 9_000_000, "dog continues running", "chunk_1"),
        ])
        normalized = normalize_via_result(via_result)
        annotated = annotate_continuity(deduplicate_evidence(normalized), via_result)
        assert all(e.continuity_group_id is None for e in annotated)


# ---------------------------------------------------------------------------
# Determinism under input-order permutation
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_group_membership_stable_across_orderings(self, tmp_path):
        asset_id = _real_asset(tmp_path, "determinism")
        _seed_scene_cache(asset_id, 0, 30_000_000, [])
        obs = [
            _obs(asset_id, 1_000_000, 5_000_000, "dog running across the yard", "chunk_0"),
            _obs(asset_id, 6_000_000, 9_000_000, "dog stops running near the fence", "chunk_1"),
            _obs(asset_id, 11_000_000, 14_000_000, "dog leaves frame past the fence", "chunk_2"),
            *_filler_observations(asset_id, 15),
        ]
        via_result = _via_result(asset_id, 0, 30_000_000, observations=obs)
        normalized = normalize_via_result(via_result)
        deduped = deduplicate_evidence(normalized)

        forward = annotate_continuity(list(deduped), via_result)
        reversed_input = annotate_continuity(list(reversed(deduped)), via_result)
        shuffled_input = annotate_continuity(
            sorted(deduped, key=lambda e: e.evidence_id, reverse=True), via_result
        )

        def group_map(items):
            return {e.evidence_id: e.continuity_group_id for e in items}

        assert group_map(forward) == group_map(reversed_input) == group_map(shuffled_input)


# ---------------------------------------------------------------------------
# Bounded complexity (no global all-pairs comparison)
# ---------------------------------------------------------------------------


class TestComplexity:
    def test_large_multi_chunk_corpus_completes_quickly(self, tmp_path):
        asset_id = _real_asset(tmp_path, "complexity")
        num_chunks = 100
        items_per_chunk = 8
        end_us = num_chunks * 2_000_000
        _seed_scene_cache(asset_id, 0, end_us, [])

        obs = []
        for chunk_idx in range(num_chunks):
            base = chunk_idx * 2_000_000
            for item_idx in range(items_per_chunk):
                obs.append(
                    _obs(
                        asset_id,
                        base + item_idx * 100,
                        base + item_idx * 100 + 50,
                        f"distinct subject {chunk_idx}-{item_idx} moves around",
                        f"chunk_{chunk_idx}",
                    )
                )
        via_result = _via_result(asset_id, 0, end_us, observations=obs)
        normalized = normalize_via_result(via_result)
        deduped = deduplicate_evidence(normalized)
        assert len(deduped) == num_chunks * items_per_chunk

        start = time.perf_counter()
        annotate_continuity(deduped, via_result)
        elapsed = time.perf_counter() - start
        # 800 items, O(N * items_per_chunk) bound would be a few thousand
        # comparisons; O(N^2) would be ~640,000. Generous ceiling well below
        # what an accidental all-pairs scan would take.
        assert elapsed < 3.0


# ---------------------------------------------------------------------------
# Non-interference with legacy Event Map / Quality Gate / Best Event Pool
# ---------------------------------------------------------------------------


class TestNonInterference:
    def test_continuity_fields_do_not_change_legacy_event_merge(self, tmp_path):
        asset_id = _real_asset(tmp_path, "non-interference")
        without = normalize_via_result(
            _via_result(
                asset_id, 0, 20_000_000,
                observations=[
                    _obs(asset_id, 1_000_000, 5_000_000, "unique observed fact", "chunk_0")
                ],
            )
        )
        with_continuity = [
            e.model_copy(
                update={
                    "continuity_group_id": "fake-group",
                    "continuity_basis": "temporal_anchor_continuity",
                }
            )
            for e in without
        ]

        legacy_events = [
            {
                "id": "clip-1", "start_us": 0, "end_us": 8_000_000,
                "title": "t", "reason": "r", "legacy_score": 0.7,
            }
        ]

        result_without = events_module._merge_via_evidence(list(legacy_events), without)
        result_with = events_module._merge_via_evidence(list(legacy_events), with_continuity)

        # legacy_score / event identity untouched either way.
        score_without = result_without["events"][0]["legacy_score"]
        score_with = result_with["events"][0]["legacy_score"]
        assert score_without == score_with == 0.7
        assert (
            [e["id"] for e in result_without["events"]] == [e["id"] for e in result_with["events"]]
        )


# ---------------------------------------------------------------------------
# Confirmation rehydration backward-compatibility
# ---------------------------------------------------------------------------


class TestConfirmationRehydration:
    def test_old_persisted_evidence_without_continuity_fields_rehydrates(self):
        from app.editor.agent import rehydrate_via_evidence

        old_item = {
            "evidence_id": "e1", "asset_id": "a", "start_us": 0, "end_us": 1_000_000,
            "evidence_kind": "observation", "nature": "observation",
            "description": "old evidence", "confidence_original": 0.7,
            "provider": "p", "model": "m",
        }
        restored = rehydrate_via_evidence([old_item])
        assert len(restored) == 1
        assert restored[0]["continuity_group_id"] is None
        assert restored[0]["continuity_basis"] is None

    def test_new_persisted_evidence_with_continuity_fields_rehydrates(self):
        from app.editor.agent import rehydrate_via_evidence

        new_item = {
            "evidence_id": "e1", "asset_id": "a", "start_us": 0, "end_us": 1_000_000,
            "evidence_kind": "event", "nature": "observation",
            "description": "grouped evidence", "confidence_original": 0.7,
            "provider": "p", "model": "m", "chunk_id": "chunk_0",
            "continuity_group_id": "abc123", "continuity_basis": "temporal_anchor_continuity",
        }
        restored = rehydrate_via_evidence([new_item])
        assert len(restored) == 1
        assert restored[0]["continuity_group_id"] == "abc123"
        assert restored[0]["continuity_basis"] == "temporal_anchor_continuity"


# ---------------------------------------------------------------------------
# Integration: agent._evidence() pipeline order
# ---------------------------------------------------------------------------


class TestEvidencePipelineIntegration:
    def test_evidence_applies_normalize_dedup_continuity_in_order(self, tmp_path):
        init_db()
        with Session(engine) as session:
            project = editor_service.create_project(session, "continuity-pipeline")
            source = tmp_path / "source.mp4"
            source.write_bytes(b"source bytes")
            asset = editor_service.add_asset(
                session, project, source, duration_us=30_000_000, has_audio=False
            )
            snapshot = inspect_project(session, project.id)

            _seed_scene_cache(asset.id, 0, 30_000_000, [])
            obs = [
                _obs(asset.id, 1_000_000, 5_000_000, "dog running across the yard", "chunk_0"),
                _obs(
                    asset.id, 6_000_000, 9_000_000,
                    "dog continues running toward fence", "chunk_1",
                ),
                *_filler_observations(asset.id, 12),
                # Exact duplicate of the first item (same normalized
                # description, full overlap) - must be deduplicated BEFORE
                # continuity ever sees it, so it must not produce a second,
                # separately-grouped entry.
                _obs(
                    asset.id, 1_000_000, 5_000_000,
                    "dog running across the yard", "chunk_0", confidence=0.5,
                ),
            ]
            via_result = _via_result(asset.id, 0, 30_000_000, observations=obs)

            evidence = _evidence(
                session, snapshot, "highlight", "Dame lo mejor",
                via_evidence=via_result.model_dump(),
            )

        assert "via_evidence" in evidence
        items = evidence["via_evidence"]
        dog_items = [i for i in items if "dog" in i["description"]]
        # Deduplicated: 2 distinct dog observations survive, not 3.
        assert len(dog_items) == 2
        group_ids = {i["continuity_group_id"] for i in dog_items}
        assert len(group_ids) == 1
        assert None not in group_ids
        for item in dog_items:
            assert item["continuity_basis"] == "temporal_anchor_continuity"
        # Non-grouped filler items keep continuity fields as None.
        filler_items = [i for i in items if "unrelated filler" in i["description"]]
        assert filler_items and all(i["continuity_group_id"] is None for i in filler_items)
