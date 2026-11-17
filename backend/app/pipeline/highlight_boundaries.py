"""Deterministic boundary refinement for highlights (SNIPRA v1.5D).

Consumes scored candidates + evidence and produces temporal refinement recommendations.
Pure intelligence -- no mutations, no editorial decisions, no media operations.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from app.editor.via_evidence import NormalizedVideoEvidence
from app.providers.base import ClipCandidate

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

@dataclass(frozen=True)
class RefinementWeights:
    """Weights for peak selection (normalized, sum to 1.0)."""
    via_event_confidence: float = 0.40
    via_payoff_present: float = 0.25
    via_reaction_present: float = 0.10
    signal_peak_loudness: float = 0.15
    transcript_payoff_density: float = 0.10

    def __post_init__(self) -> None:
        total = (
            self.via_event_confidence + self.via_payoff_present +
            self.via_reaction_present + self.signal_peak_loudness +
            self.transcript_payoff_density
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"RefinementWeights must sum to 1.0, got {total}")

DEFAULT_REFINEMENT_WEIGHTS = RefinementWeights()

# Tolerance for temporal matching/snapping (NOT editorial duration)
SNAP_TOLERANCE_START = 1.5  # seconds for transcript/scene snapping
SNAP_TOLERANCE_END = 1.0    # seconds for transcript/scene snapping

# Technical tolerances (NOT editorial rules - no fixed durations)
SILENCE_START_TOLERANCE = 0.5  # seconds after silence start to use as boundary
BOUNDARY_TOLERANCE = 0.5  # seconds for content boundary matching
# No PAYOFF_BUFFER / REACTION_COHERENCE_WINDOW -- use evidence directly


# ------------------------------------------------------------------------------
# Output structures
# ------------------------------------------------------------------------------

class RefinementStatus(StrEnum):
    """Status of the boundary refinement."""
    REFINED = "refined"           # Boundaries adjusted based on sufficient evidence
    UNCHANGED = "unchanged"       # No reliable evidence to change boundaries
    UNCERTAIN = "uncertain"       # Evidence exists but conflicting/insufficient


class AnchorPrecision(StrEnum):
    """Precision of a temporal anchor."""
    POINT = "point"               # Single timestamp (e.g., loudness peak)
    WINDOW = "window"             # Time window (e.g., EVENT start_us..end_us)


@dataclass(frozen=True)
class TemporalAnchor:
    """A single temporal anchor point/window with its evidence source."""
    timestamp: float              # representative timestamp (midpoint for windows)
    source: str                   # e.g., "via_event", "transcript_payoff", "signal_loudness"
    evidence_ref: str             # evidence_id or descriptive reference
    confidence: float             # 0.0-1.0 weight of this anchor
    kind: Literal["peak", "setup", "action", "payoff", "reaction"]
    precision: AnchorPrecision = AnchorPrecision.POINT
    window_start: float | None = None
    window_end: float | None = None


@dataclass(frozen=True)
class RefinedBoundaries:
    """Refinement recommendation for a single candidate."""
    candidate_id: str             # stable identifier
    original_start: float
    original_end: float
    refined_start: float
    refined_end: float
    peak_at: float | None
    setup_at: float | None
    action_at: float | None
    payoff_at: float | None            # Only for POINT precision payoff
    payoff_window_start: float | None = None
    payoff_window_end: float | None = None
    reaction_at: float | None = None   # Only for POINT precision reaction
    reaction_window_start: float | None = None
    reaction_window_end: float | None = None
    status: RefinementStatus = "unchanged"
    confidence: float = 0.0               # 0.0-1.0 confidence in the refinement
    anchors: tuple[TemporalAnchor, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def duration_change(self) -> float:
        """Positive = expanded, negative = reduced."""
        return (self.refined_end - self.refined_start) - (self.original_end - self.original_start)

    @property
    def was_narrowed(self) -> bool:
        return self.refined_start > self.original_start + 0.1 or self.refined_end < self.original_end - 0.1

    @property
    def was_expanded(self) -> bool:
        return self.refined_start < self.original_start - 0.1 or self.refined_end > self.original_end + 0.1


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _stable_candidate_id(candidate: ClipCandidate) -> str:
    """Deterministic identity for candidate."""
    parts = [
        str(candidate.start),
        str(candidate.end),
        candidate.title or "",
        candidate.hook or "",
        candidate.reason or "",
    ]
    canonical = "|".join(parts)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _normalize_text(text: str) -> str:
    """Lowercase, remove accents, remove punctuation, collapse whitespace."""
    import re
    import unicodedata
    folded = unicodedata.normalize("NFKD", text or "")
    ascii_text = "".join(c for c in folded if not unicodedata.combining(c))
    ascii_text = re.sub(r"[^\w\s]", "", ascii_text)
    return " ".join(ascii_text.lower().split())


def _compute_coverage_factor(evidence: NormalizedVideoEvidence) -> float:
    """Compute coverage factor from evidence metadata (0.0-1.0)."""
    if evidence.coverage_ratio is not None:
        return _clamp01(evidence.coverage_ratio)
    if evidence.partial:
        return 0.5
    if evidence.failed_chunks:
        return 0.7
    return 1.0


def _is_structurally_coherent_reaction(
    reaction: TemporalAnchor,
    payoff_windows: list[TemporalAnchor],
    all_anchors: list[TemporalAnchor],
) -> bool:
    """Check if reaction is structurally coherent with payoff.

    Structural coherence is based on evidence, NOT fixed time thresholds:
    A. Same evidence_ref when payoff + reaction come from same NormalizedVideoEvidence
    B. Temporal overlap between payoff and reaction windows
    C. Compatible continuity_group_id when available (future extensibility)
    """
    # A. Same evidence source (same VIA evidence item)
    for pw in payoff_windows:
        if pw.evidence_ref == reaction.evidence_ref:
            return True

    # B. Temporal overlap between windows
    r_start = reaction.window_start or reaction.timestamp
    r_end = reaction.window_end or reaction.timestamp
    for pw in payoff_windows:
        p_start = pw.window_start or pw.timestamp
        p_end = pw.window_end or pw.timestamp
        if r_start <= p_end and r_end >= p_start:
            return True

    return False


def _compute_status(
    *,
    anchors: list,
    boundaries_changed: bool,
    attached_via: list,
    avg_coverage: float,
    avg_quality: float
) -> RefinementStatus:
    """Determine refinement status based on evidence quality and boundary changes."""
    if not anchors:
        return RefinementStatus.UNCHANGED

    # No boundary changes → UNCHANGED (regardless of VIA)
    if not boundaries_changed:
        return RefinementStatus.UNCHANGED

    # If no VIA evidence, can only be UNCERTAIN at best
    if not attached_via:
        return RefinementStatus.UNCERTAIN

    # Low coverage/quality → UNCERTAIN even with VIA
    if avg_coverage < 0.5 or avg_quality < 0.4:
        return RefinementStatus.UNCERTAIN

    return RefinementStatus.REFINED


# ------------------------------------------------------------------------------
# Evidence extraction
# ------------------------------------------------------------------------------

def _extract_via_anchors(
    candidate: ClipCandidate,
    attached_evidence: list[NormalizedVideoEvidence],
) -> list[TemporalAnchor]:
    """Extract temporal anchors from VIA evidence attached to candidate."""
    anchors: list[TemporalAnchor] = []

    for ev in attached_evidence:
        start_s = ev.start_us / 1_000_000
        end_s = ev.end_us / 1_000_000
        (start_s + end_s) / 2
        coverage = _compute_coverage_factor(ev)
        effective_conf = _clamp01(ev.confidence_original * coverage)

        if ev.evidence_kind.value == "event":
            # EVENT START -- real temporal evidence for setup/action onset
            anchors.append(TemporalAnchor(
                timestamp=start_s,
                source="via_event_start",
                evidence_ref=ev.evidence_id,
                confidence=effective_conf,
                kind="setup",
                precision=AnchorPrecision.POINT,
            ))

            # EVENT MIDPOINT -- peak anchor (real climax evidence)
            anchors.append(TemporalAnchor(
                timestamp=(start_s + end_s) / 2,
                source="via_event_mid",
                evidence_ref=ev.evidence_id,
                confidence=effective_conf,
                kind="peak",
                precision=AnchorPrecision.POINT,
            ))

            # EVENT END -- natural boundary
            anchors.append(TemporalAnchor(
                timestamp=end_s,
                source="via_event_end",
                evidence_ref=ev.evidence_id,
                confidence=effective_conf,
                kind="action",
                precision=AnchorPrecision.POINT,
            ))

            # Payoff WINDOW if present
            if ev.payoff_present:
                anchors.append(TemporalAnchor(
                    timestamp=(start_s + end_s) / 2,
                    source="via_payoff_window",
                    evidence_ref=ev.evidence_id,
                    confidence=effective_conf * 0.9,
                    kind="payoff",
                    precision=AnchorPrecision.WINDOW,
                    window_start=start_s,
                    window_end=end_s,
                ))

            # Reaction WINDOW if present
            if ev.reaction_present:
                anchors.append(TemporalAnchor(
                    timestamp=(start_s + end_s) / 2,
                    source="via_reaction_window",
                    evidence_ref=ev.evidence_id,
                    confidence=effective_conf * 0.8,
                    kind="reaction",
                    precision=AnchorPrecision.WINDOW,
                    window_start=start_s,
                    window_end=end_s,
                ))

        elif ev.evidence_kind.value == "scene":
            anchors.append(TemporalAnchor(
                timestamp=start_s,
                source="via_scene_start",
                evidence_ref=ev.evidence_id,
                confidence=ev.confidence_original * 0.5,
                kind="setup",
                precision=AnchorPrecision.POINT,
            ))
            anchors.append(TemporalAnchor(
                timestamp=end_s,
                source="via_scene_end",
                evidence_ref=ev.evidence_id,
                confidence=ev.confidence_original * 0.5,
                kind="setup",
                precision=AnchorPrecision.POINT,
            ))

    return anchors


def _extract_signal_anchors(
    candidate: ClipCandidate,
    signals_doc: dict | None,
) -> list[TemporalAnchor]:
    """Extract anchors from audio/video signals."""
    if not signals_doc:
        return []

    anchors: list[TemporalAnchor] = []

    # Loudness peak -- relative normalization within candidate window
    loudness = signals_doc.get("loudness", [])
    if loudness:
        samples = []
        for item in loudness:
            try:
                ts, level = float(item[0]), float(item[1])
            except (IndexError, TypeError, ValueError):
                continue
            if candidate.start + 0.5 <= ts <= candidate.end - 0.5:
                samples.append((ts, level))

        if samples:
            levels = [level for _, level in samples]
            min_level = min(levels)
            max_level = max(levels)
            level_range = max_level - min_level

            for ts, level in samples:
                norm_conf = (
                    _clamp01((level - min_level) / level_range) if level_range > 0 else 0.0
                )
                anchors.append(TemporalAnchor(
                    timestamp=ts,
                    source="signal_loudness",
                    evidence_ref=f"loudness_{ts:.2f}",
                    confidence=norm_conf,
                    kind="peak",
                    precision=AnchorPrecision.POINT,
                ))

    # Scene boundaries as setup anchors
    scene_times = [float(t) for t in signals_doc.get("scene_times", [])]
    for st in scene_times:
        if candidate.start <= st <= candidate.end:
            anchors.append(TemporalAnchor(
                timestamp=st,
                source="signal_scene",
                evidence_ref=f"scene_{st:.2f}",
                confidence=0.6,
                kind="setup",
                precision=AnchorPrecision.POINT,
            ))

    # Silence intervals as potential boundaries
    silence_intervals = signals_doc.get("silence_intervals", [])
    for s_start, s_end in silence_intervals:
        s_start_f = float(s_start)
        s_end_f = float(s_end)
        if candidate.start <= s_start_f <= candidate.end:
            anchors.append(TemporalAnchor(
                timestamp=s_start_f,
                source="signal_silence_start",
                evidence_ref=f"silence_{s_start_f:.2f}",
                confidence=0.5,
                kind="setup",
                precision=AnchorPrecision.POINT,
            ))
        if candidate.start <= s_end_f <= candidate.end:
            anchors.append(TemporalAnchor(
                timestamp=s_end_f,
                source="signal_silence_end",
                evidence_ref=f"silence_{s_end_f:.2f}",
                confidence=0.5,
                kind="action",
                precision=AnchorPrecision.POINT,
            ))

    return anchors


def _extract_transcript_anchors(
    candidate: ClipCandidate,
    transcript_segments: list[dict],
) -> list[TemporalAnchor]:
    """Extract anchors from transcript payoff terms (NOT reaction)."""
    from app.pipeline.highlight_scoring import REWARD_TERMS, TRANSCRIPT_PAYOFF_TERMS

    anchors: list[TemporalAnchor] = []

    for seg in transcript_segments:
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", seg_start))
        if seg_end <= candidate.start or seg_start >= candidate.end:
            continue

        text = str(seg.get("text", "")).strip()
        if not text:
            continue

        normalized = _normalize_text(text)

        # Check for payoff terms -- supports payoff
        payoff_hits = [t for t in TRANSCRIPT_PAYOFF_TERMS if t in normalized]
        if payoff_hits:
            anchors.append(TemporalAnchor(
                timestamp=seg_end,
                source="transcript_payoff",
                evidence_ref=f"transcript_payoff_{','.join(payoff_hits[:2])}",
                confidence=min(0.9, len(payoff_hits) * 0.3),
                kind="payoff",
                precision=AnchorPrecision.POINT,
            ))

        # Check for reward terms -- supports REWARD/REWARD, NOT reaction
        reward_hits = [t for t in REWARD_TERMS if t in normalized]
        if reward_hits:
            anchors.append(TemporalAnchor(
                timestamp=seg_end,
                source="transcript_reward",
                evidence_ref=f"transcript_reward_{','.join(reward_hits[:2])}",
                confidence=min(0.8, len(reward_hits) * 0.25),
                kind="payoff",  # reward supports payoff, not reaction
                precision=AnchorPrecision.POINT,
            ))

    return anchors


# ------------------------------------------------------------------------------
# Peak selection
# ------------------------------------------------------------------------------

def _select_peak(
    anchors: list[TemporalAnchor],
    candidate: ClipCandidate,
    weights: RefinementWeights,
) -> float | None:
    """Select the peak timestamp from anchors using weighted scoring.

    Only real climax evidence can be peak: via_event_mid, signal_loudness with confidence > 0.
    setup/action/scene cannot compete as peak.
    """
    if not anchors:
        return None

    # Group anchors by timestamp (within 0.5s tolerance)
    grouped: dict[float, list[TemporalAnchor]] = defaultdict(list)
    for a in anchors:
        key = round(a.timestamp * 2) / 2  # 0.5s bins
        grouped[key].append(a)

    best_ts = None
    best_score = 0.0  # Only accept positive scores

    for ts, group in grouped.items():
        if ts < candidate.start or ts > candidate.end:
            continue

        score = 0.0
        for a in group:
            w = 0.0
            if a.kind == "peak":
                if a.source in ("via_event_mid",):
                    w = weights.via_event_confidence
                elif a.source == "signal_loudness":
                    w = weights.signal_peak_loudness
            elif a.kind == "payoff":
                if a.source == "via_payoff_window":
                    w = weights.via_payoff_present
                elif a.source in ("transcript_payoff", "transcript_reward"):
                    w = weights.transcript_payoff_density
            elif a.kind == "reaction":
                w = 0.0
            elif a.kind in ("setup", "action"):
                w = 0.0  # setup/action cannot be peak

            score += a.confidence * w

        # Deterministic tiebreak: prefer earlier timestamp
        if score > best_score or (abs(score - best_score) < 1e-9 and (best_ts is None or ts < best_ts)):
            best_score = score
            best_ts = ts

    # Only return peak if score is meaningfully positive
    if best_score <= 0:
        return None
    return best_ts


# ------------------------------------------------------------------------------
# Boundary refinement
# ------------------------------------------------------------------------------

def _refine_start(
    candidate: ClipCandidate,
    peak_ts: float | None,
    anchors: list[TemporalAnchor],
    transcript_segments: list[dict],
    signals_doc: dict | None,
    video_duration: float,
) -> tuple[float, list[str]]:
    """Refine the start boundary. Returns (refined_start, notes)."""
    notes: list[str] = []

    refined_start = candidate.start

    # 1. Use EVENT START as primary evidence for reducing lead-in
    event_start_anchors = [a for a in anchors if a.source == "via_event_start"]
    if event_start_anchors:
        # Use the latest EVENT start before peak/reference as the start boundary
        reference_ts = peak_ts if peak_ts is not None else (candidate.start + candidate.end) / 2
        valid_starts = [a for a in event_start_anchors if a.timestamp < reference_ts]
        if valid_starts:
            latest_event_start = max(valid_starts, key=lambda a: a.timestamp)
            # Use EVENT start directly as evidence (no arbitrary setup window limit)
            if latest_event_start.timestamp > refined_start + SILENCE_START_TOLERANCE:
                refined_start = latest_event_start.timestamp
                notes.append(f"start_refined_to_event_start:{refined_start:.2f}")

    # 2. Transcript segment boundary near start
    if transcript_segments:
        seg_starts = []
        for seg in transcript_segments:
            seg_start = float(seg.get("start", 0.0))
            if candidate.start - 1.5 <= seg_start <= candidate.start + 1.5:
                seg_starts.append(seg_start)
        if seg_starts:
            transcript_start = min(seg_starts, key=lambda s: abs(s - candidate.start))
            if transcript_start > refined_start and transcript_start < candidate.start + 1.5:
                refined_start = transcript_start
                notes.append(f"start_snapped_to_transcript:{refined_start:.2f}")

    # 3. Scene boundary near start
    if signals_doc:
        scene_times = sorted([float(t) for t in signals_doc.get("scene_times", [])])
        for st in scene_times:
            if candidate.start - 1.5 <= st <= candidate.start + 1.5:
                if st > refined_start:
                    refined_start = st
                    notes.append(f"start_snapped_to_scene:{refined_start:.2f}")
                break

    # 4. Silence start after candidate start can trim lead-in
    if signals_doc:
        silence_intervals = signals_doc.get("silence_intervals", [])
        for s_start, _s_end in silence_intervals:
            s_start_f = float(s_start)
            if candidate.start < s_start_f < candidate.start + 2.0:
                if s_start_f > refined_start:
                    refined_start = s_start_f
                    notes.append(f"start_snapped_to_silence:{refined_start:.2f}")
                break

    # Clamp
    # Use epsilon tolerance instead of fixed 3s editorial minimum
    refined_start = max(0.0, min(refined_start, candidate.end - 0.1))
    return refined_start, notes


def _refine_end(
    candidate: ClipCandidate,
    peak_ts: float | None,
    anchors: list[TemporalAnchor],
    transcript_segments: list[dict],
    signals_doc: dict | None,
    video_duration: float,
) -> tuple[float, list[str]]:
    """Refine the end boundary. Returns (refined_end, notes)."""
    notes: list[str] = []

    refined_end = candidate.end
    original_end = candidate.end

    # 1. Find the latest meaningful content boundary
    content_boundaries: list[float] = []

    # Payoff windows -- strongest signal for meaningful end
    payoff_windows = [a for a in anchors if a.kind == "payoff" and a.precision == AnchorPrecision.WINDOW]
    payoff_points = [a for a in anchors if a.kind == "payoff" and a.precision == AnchorPrecision.POINT]
    if payoff_windows:
        latest_payoff = max(payoff_windows, key=lambda a: a.window_end or a.timestamp)
        content_boundaries.append(latest_payoff.window_end or latest_payoff.timestamp)
        if latest_payoff.window_end:
            notes.append(f"payoff_window_end:{latest_payoff.window_end:.2f}")
        else:
            notes.append(f"payoff_at:{latest_payoff.timestamp:.2f}")

    # Reaction windows -- only if structurally coherent with payoff
    reaction_windows = [a for a in anchors if a.kind == "reaction" and a.precision == AnchorPrecision.WINDOW]
    if reaction_windows and payoff_windows:
        coherent_reactions = [
            r for r in reaction_windows
            if _is_structurally_coherent_reaction(r, payoff_windows, anchors)
        ]
        if coherent_reactions:
            latest_reaction = max(coherent_reactions, key=lambda a: a.window_end or a.timestamp)
            content_boundaries.append(latest_reaction.window_end or latest_reaction.timestamp)
            notes.append(f"reaction_window_end:{(latest_reaction.window_end or latest_reaction.timestamp):.2f}")

    # Event ends -- natural action boundaries
    event_ends = [a for a in anchors if a.source in ("via_event_end", "via_event_mid")]
    if event_ends:
        latest_event = max(event_ends, key=lambda a: a.timestamp)
        content_boundaries.append(latest_event.timestamp)
        notes.append(f"event_end:{latest_event.timestamp:.2f}")

    # Transcript sentence ends near end
    if transcript_segments:
        seg_ends = []
        for seg in transcript_segments:
            seg_end = float(seg.get("end", seg.get("start", 0.0)))
            if candidate.end - 1.0 <= seg_end <= candidate.end + 1.0:
                seg_ends.append(seg_end)
        if seg_ends:
            transcript_end = min(seg_ends)
            content_boundaries.append(transcript_end)
            notes.append(f"transcript_end:{transcript_end:.2f}")

    # Scene boundaries
    if signals_doc:
        scene_times = sorted([float(t) for t in signals_doc.get("scene_times", [])])
        for st in reversed(scene_times):
            if candidate.end - 1.0 <= st <= candidate.end + 1.0:
                content_boundaries.append(st)
                notes.append(f"scene_boundary:{st:.2f}")
                break

    # Silence intervals -- use silence START after meaningful content as potential end boundary
    # silence_end near candidate.end should NOT keep entire silent tail
    if signals_doc:
        silence_intervals = signals_doc.get("silence_intervals", [])
        # Sort silence intervals deterministically by start time
        silence_intervals = sorted(silence_intervals, key=lambda x: x[0])
        # This will be evaluated AFTER we determine the latest meaningful content
        # We'll handle silence starts AFTER determining latest_content

    # Determine refined end: use the LATEST content boundary that is before original end
    # This is the key logic for reducing dead aftermath
    valid_boundaries = [b for b in content_boundaries if b <= original_end + BOUNDARY_TOLERANCE]
    if valid_boundaries:
        latest_content = max(valid_boundaries)
        if latest_content < original_end - BOUNDARY_TOLERANCE:  # Only reduce if meaningful reduction
            refined_end = min(latest_content + BOUNDARY_TOLERANCE, original_end)
            notes.append(f"end_refined_to_content:{refined_end:.2f}")
        else:
            refined_end = candidate.end
    else:
        refined_end = candidate.end

    # Payoff preservation: use payoff window end directly as boundary
    # (no arbitrary buffer -- the payoff window itself IS the boundary)
    payoff_windows = [a for a in anchors if a.kind == "payoff" and a.precision == AnchorPrecision.WINDOW]
    payoff_points = [a for a in anchors if a.kind == "payoff" and a.precision == AnchorPrecision.POINT]

    latest_payoff_end = None
    if payoff_windows:
        latest_payoff_end = max(p.window_end or p.timestamp for p in payoff_windows)
    elif payoff_points:
        latest_payoff_end = max(p.timestamp for p in payoff_points)

    if latest_payoff_end is not None and refined_end < latest_payoff_end:
        # Only extend if payoff would be CUT (refined_end < payoff_end)
        # Use payoff window end directly as boundary (no arbitrary buffer)
        refined_end = min(latest_payoff_end, video_duration)
        notes.append(f"end_at_payoff:{refined_end:.2f}")

    # Coherent reaction preservation (structural, not time-threshold)
    reaction_windows = [a for a in anchors if a.kind == "reaction" and a.precision == AnchorPrecision.WINDOW]

    if reaction_windows and payoff_windows:
        coherent_reactions = [
            r for r in reaction_windows
            if _is_structurally_coherent_reaction(r, payoff_windows, anchors)
        ]
        if coherent_reactions:
            latest_reaction_end = max((r.window_end or r.timestamp) for r in coherent_reactions)
            # Use reaction window end directly (no arbitrary buffer)
            if refined_end < latest_reaction_end:
                refined_end = min(latest_reaction_end, video_duration)
                notes.append(f"end_preserved_for_reaction:{refined_end:.2f}")

    # Silence start AFTER meaningful content can serve as end boundary
    # This avoids keeping entire silent tail - use silence START after content as boundary
    if signals_doc:
        silence_intervals = signals_doc.get("silence_intervals", [])
        # Sort silence intervals deterministically by start time
        silence_intervals = sorted(silence_intervals, key=lambda x: x[0])
        # Find the latest meaningful content end (payoff, event, transcript, scene)
        meaningful_end = max(
            [a.window_end or a.timestamp for a in anchors if a.kind == "payoff" and a.precision == AnchorPrecision.WINDOW]
            + [a.timestamp for a in anchors if a.kind == "payoff" and a.precision == AnchorPrecision.POINT]
            + [a.timestamp for a in anchors if a.source in ("via_event_end", "via_event_mid")]
            + [float(seg.get("end", seg.get("start", 0.0))) for seg in transcript_segments if candidate.end - 1.0 <= float(seg.get("end", 0.0)) <= candidate.end + 1.0]
            + [float(t) for t in signals_doc.get("scene_times", []) if candidate.end - 1.0 <= float(t) <= candidate.end + 1.0] if signals_doc else []
            + [candidate.end], default=candidate.end
        )
        for s_start, _s_end in sorted(silence_intervals, key=lambda x: x[0]):
            s_start_f = float(s_start)
            if meaningful_end <= s_start_f <= candidate.end:
                refined_end = min(s_start_f + 0.5, candidate.end, video_duration)
                notes.append(f"end_at_silence_start:{refined_end:.2f}")
                break

    # Clamp to video duration
    refined_end = min(video_duration, refined_end)
    return refined_end, notes
# ------------------------------------------------------------------------------

def refine_candidate(
    candidate: ClipCandidate,
    via_evidence: list[NormalizedVideoEvidence] | None = None,
    transcript_segments: list[dict] | None = None,
    signals_doc: dict | None = None,
    video_duration: float = 0.0,
    weights: RefinementWeights | None = None,
) -> RefinedBoundaries:
    """Refine candidate boundaries based on available evidence."""
    via_evidence = via_evidence or []
    transcript_segments = transcript_segments or []
    signals_doc = signals_doc or {}
    w = weights or DEFAULT_REFINEMENT_WEIGHTS

    # video_duration is authoritative if provided (> 0)
    effective_video_duration = video_duration if video_duration > 0 else candidate.end

    cand_id = _stable_candidate_id(candidate)

    # Attach VIA evidence to this candidate (by temporal overlap)
    attached_via: list[NormalizedVideoEvidence] = []
    cand_start_us = int(round(candidate.start * 1_000_000))
    cand_end_us = int(round(candidate.end * 1_000_000))
    for ev in via_evidence:
        overlap_us = max(0, min(cand_end_us, ev.end_us) - max(cand_start_us, ev.start_us))
        if overlap_us == 0:
            continue
        cand_dur_us = cand_end_us - cand_start_us
        ev_dur_us = ev.end_us - ev.start_us
        min_dur_us = min(cand_dur_us, ev_dur_us)
        if min_dur_us > 0 and (overlap_us / min_dur_us) >= 0.3:
            attached_via.append(ev)

    # Extract all anchors
    anchors: list[TemporalAnchor] = []
    anchors.extend(_extract_via_anchors(candidate, attached_via))
    anchors.extend(_extract_signal_anchors(candidate, signals_doc))
    anchors.extend(_extract_transcript_anchors(candidate, transcript_segments))

    # Sort anchors by timestamp for deterministic processing
    anchors.sort(key=lambda a: (a.timestamp, a.source, a.evidence_ref))

    # Select peak
    peak_ts = _select_peak(anchors, candidate, w)

    # Refine boundaries
    refined_start, start_notes = _refine_start(
        candidate, peak_ts, anchors, transcript_segments, signals_doc, effective_video_duration
    )
    refined_end, end_notes = _refine_end(
        candidate, peak_ts, anchors, transcript_segments, signals_doc, effective_video_duration
    )

    # Ensure valid range
    # Use epsilon tolerance instead of fixed 1s editorial minimum
    if refined_end <= refined_start + 0.1:
        # Fallback: keep original if refinement would break
        refined_start = candidate.start
        refined_end = min(candidate.end, effective_video_duration)
        start_notes = ["fallback_original"]
        end_notes = []

    # Determine if boundaries actually changed
    start_changed = abs(refined_start - candidate.start) > 0.5
    end_changed = abs(refined_end - candidate.end) > 0.5

    # Compute average quality and coverage
    avg_quality = 0.0
    avg_coverage = 0.0
    if attached_via:
        avg_quality = sum(e.confidence_original for e in attached_via) / len(attached_via)
        avg_coverage = sum(_compute_coverage_factor(e) for e in attached_via) / len(attached_via)

    # Determine status
    status = _compute_status(
        anchors=anchors,
        boundaries_changed=start_changed or end_changed,
        attached_via=attached_via,
        avg_coverage=avg_coverage,
        avg_quality=avg_quality,
    )

    # Adjust confidence
    if attached_via:
        avg_quality = sum(e.confidence_original for e in attached_via) / len(attached_via)
        avg_coverage = sum(_compute_coverage_factor(e) for e in attached_via) / len(attached_via)
        confidence = _clamp01(0.75 * avg_quality * avg_coverage)
    else:
        confidence = 0.4

    # Build payoff/reaction outputs -- ONLY from anchors actually used in refinement
    payoff_at = None
    payoff_window_start = None
    payoff_window_end = None
    reaction_at = None
    reaction_window_start = None
    reaction_window_end = None

    # Find the payoff anchor that actually influenced the end boundary
    # Check WINDOW anchors first (VIA payoff), then POINT anchors (transcript payoff)
    payoff_windows = [a for a in anchors if a.kind == "payoff" and a.precision == AnchorPrecision.WINDOW]
    payoff_points = [a for a in anchors if a.kind == "payoff" and a.precision == AnchorPrecision.POINT]

    payoff_at = None
    payoff_window_start = None
    payoff_window_end = None

    if payoff_windows:
        latest_payoff = max(payoff_windows, key=lambda a: a.window_end or a.timestamp)
        payoff_window_start = latest_payoff.window_start
        payoff_window_end = latest_payoff.window_end
        payoff_at = None  # WINDOW anchors don't produce payoff_at
    else:
        payoff_points = [a for a in anchors if a.kind == "payoff" and a.precision == AnchorPrecision.POINT]
        if payoff_points:
            latest_payoff = max(payoff_points, key=lambda a: a.timestamp)
            payoff_at = latest_payoff.timestamp

    # Find reaction that was actually used (structurally coherent)
    reaction_windows = [a for a in anchors if a.kind == "reaction" and a.precision == AnchorPrecision.WINDOW]
    reaction_at = None
    reaction_window_start = None
    reaction_window_end = None

    if reaction_windows and payoff_windows:
        coherent_reactions = [
            r for r in reaction_windows
            if _is_structurally_coherent_reaction(r, payoff_windows, anchors)
        ]
        if coherent_reactions:
            latest_reaction_end = max([r.window_end or r.timestamp for r in coherent_reactions])
            # Use the earliest window_start from coherent reactions
            earliest_reaction_start = min([r.window_start or r.timestamp for r in coherent_reactions])
            reaction_window_start = earliest_reaction_start
            reaction_window_end = latest_reaction_end
            # Use reaction window end directly (no arbitrary buffer)
            if refined_end < latest_reaction_end:
                refined_end = min(latest_reaction_end, video_duration)
                end_notes.append(f"end_preserved_for_reaction:{refined_end:.2f}")

    # Collect evidence refs
    evidence_refs = tuple(sorted(set(a.evidence_ref for a in anchors)))

    # Build notes
    all_notes = tuple(start_notes + end_notes) if (start_notes or end_notes) else ("no_change",)

    return RefinedBoundaries(
        candidate_id=cand_id,
        original_start=candidate.start,
        original_end=candidate.end,
        refined_start=refined_start,
        refined_end=refined_end,
        peak_at=peak_ts,
        setup_at=None,
        action_at=None,
        payoff_at=payoff_at,
        payoff_window_start=payoff_window_start,
        payoff_window_end=payoff_window_end,
        reaction_at=reaction_at,
        reaction_window_start=reaction_window_start,
        reaction_window_end=reaction_window_end,
        status=status,
        confidence=_clamp01(confidence),
        anchors=tuple(anchors),
        evidence_refs=evidence_refs,
        notes=all_notes,
    )


def refine_candidates(
    candidates: list[ClipCandidate],
    via_evidence: list[NormalizedVideoEvidence] | None = None,
    transcript_segments: list[dict] | None = None,
    signals_doc: dict | None = None,
    video_duration: float = 0.0,
    weights: RefinementWeights | None = None,
) -> list[RefinedBoundaries]:
    """Refine boundaries for multiple candidates deterministically."""
    results = []
    for c in candidates:
        results.append(refine_candidate(
            c, via_evidence, transcript_segments, signals_doc, video_duration, weights
        ))

    # Stable order by candidate_id
    results.sort(key=lambda r: r.candidate_id)
    return results