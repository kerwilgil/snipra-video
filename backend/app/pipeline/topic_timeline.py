"""Deterministic Topic Timeline for video source context (SNIPRA v1.5E).

Builds a temporal sequence of topic spans from transcript segments + VIA evidence.
Pure read-only intelligence — no mutations, no editorial decisions, no media operations.

Answers: "What topic/context is being discussed in each video segment and where does it change?"
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from app.editor.via_evidence import NormalizedVideoEvidence

# ──────────────────────────────────────────────────────────────────────────────
# Configuration — named thresholds (heuristics, not editorial padding)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TopicTimelineConfig:
    """Configuration for topic boundary detection."""
    # Context window: number of adjacent segments on each side for comparison
    context_window: int = 2

    # Topic boundary threshold: weighted Jaccard distance above this = boundary
    # Range 0.0–1.0. Lower = more splits. Higher = fewer splits.
    boundary_threshold: float = 0.85

    # Minimum segments per topic span (prevents over-fragmentation)
    min_segments_per_span: int = 1

    # Gap penalty: temporal gap (seconds) that adds to boundary evidence
    # Only secondary signal; never sole reason for split
    gap_seconds_penalty: float = 5.0

    # Minimum transcript chars in window to consider it informative
    min_chars_per_window: int = 5

    # VIA evidence influence weight (0.0–1.0, secondary only)
    via_influence_weight: float = 0.3

    # Continuity group influence: same group reduces split confidence
    continuity_same_group_reduction: float = 0.15

    # Keywords: top-N per span
    max_keywords_per_span: int = 8

    # Adjacent span merge threshold: keyword TF Jaccard similarity >= this = merge
    # Range 0.0–1.0. Lower = more merging. Higher = fewer merges.
    adjacent_merge_threshold: float = 0.2

    # Single-phrase excursion smoothing threshold
    # Range 0.0–1.0. Lower = more smoothing. Higher = less smoothing.
    single_phrase_excursion_threshold: float = 0.5

    # Large gap threshold (seconds) - gaps >= this create explicit boundaries
    large_gap_threshold: float = 15.0

    # Minimum transcript coverage ratio for a span to be considered well-covered
    # Range 0.0–1.0. Spans below this get confidence penalty.
    min_transcript_coverage: float = 0.5


DEFAULT_TOPIC_CONFIG = TopicTimelineConfig()


# Minimal generic stopwords ES/EN (grammatical function words only)
# Not gaming-specific. Colors/pronouns/articles/prepositions/copulas.
_STOPWORDS: frozenset[str] = frozenset({
    # English
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "must", "can", "shall", "am", "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them", "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those", "here", "there", "where", "what", "who", "how",
    "in", "on", "at", "by", "for", "with", "from", "to", "of", "about", "into",
    "over", "under", "up", "down", "out", "off", "again", "once", "more",
    "very", "so", "too", "also", "just", "only", "even", "than", "as", "like",
    # Spanish
    "un", "una", "unos", "unas", "el", "la", "los", "las", "y", "o", "u", "e",
    "de", "del", "en", "con", "por", "para", "sin", "sobre", "entre", "hasta",
    "desde", "hacia", "durante", "tras", "bajo", "ante", "cabe", "según",
    "es", "son", "era", "eran", "fue", "fueron", "ser", "estar", "ha", "han",
    "había", "habían", "hube", "hubo", "tengo", "tiene", "tenemos",
    "te", "se", "nos", "os", "le", "les", "lo", "mi", "tu", "su",
    "mis", "tus", "sus", "mío", "tuyo", "suyo",
    "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
    "aquel", "aquella", "aquellos", "aquellas", "que", "quien", "quienes",
    "cual", "cuales", "cuando", "donde", "como", "porque", "pues", "asi",
    "si", "no", "sí", "también", "tampoco", "más", "menos", "muy", "mucho",
    "poco", "todo", "todos", "toda", "todas", "nada", "nadie", "nunca",
    "siempre", "ahora", "luego", "antes", "después", "hoy", "ayer", "mañana",
})

# Token regex: alphanumeric + Spanish accents
_TOKEN_RE = re.compile(r"[a-z0-9áéíóúüñ]+")


def _normalize_text(text: str) -> str:
    """Local normalization for topic timeline — identical semantics to 1.5C/1.5D."""
    folded = unicodedata.normalize("NFKD", text or "")
    ascii_text = "".join(c for c in folded if not unicodedata.combining(c))
    ascii_text = re.sub(r"[^\w\s]", "", ascii_text)
    return " ".join(ascii_text.lower().split())


def _tokenize(text: str) -> list[str]:
    """Extract tokens from normalized text, excluding stopwords."""
    normalized = _normalize_text(text)
    tokens = _TOKEN_RE.findall(normalized)
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def _compute_tf(tokens: list[str]) -> Counter[str]:
    """Term frequency counter for multiset Jaccard."""
    return Counter(tokens)


def _weighted_jaccard_tf(tf_left: Counter[str], tf_right: Counter[str]) -> float:
    """Weighted Jaccard using term frequencies (multiset).

    sum(min(freq_left[t], freq_right[t])) / sum(max(freq_left[t], freq_right[t]))
    """
    if not tf_left and not tf_right:
        return 1.0
    if not tf_left or not tf_right:
        return 0.0

    all_tokens = set(tf_left.keys()) | set(tf_right.keys())
    numerator = sum(min(tf_left.get(t, 0), tf_right.get(t, 0)) for t in all_tokens)
    denominator = sum(max(tf_left.get(t, 0), tf_right.get(t, 0)) for t in all_tokens)

    if denominator == 0:
        return 1.0
    return numerator / denominator


def _extract_segment_text(segment: dict) -> str:
    """Get normalized text from a transcript segment."""
    return segment.get("text", "").strip()


def _segment_keywords(segment: dict, max_keywords: int = 8) -> list[str]:
    """Extract top keywords from a single segment using TF."""
    text = _extract_segment_text(segment)
    tokens = _tokenize(text)
    if not tokens:
        return []

    tf = _compute_tf(tokens)
    # Sort by freq DESC, then token ASC for deterministic tie-break
    sorted_tokens = sorted(tf.items(), key=lambda x: (-x[1], x[0]))
    return [token for token, _ in sorted_tokens[:max_keywords]]


def _window_tokens(segments: list[dict], start_idx: int, end_idx: int) -> Counter[str]:
    """Aggregate token frequencies for a window of segments."""
    combined = Counter()
    for i in range(start_idx, min(end_idx, len(segments))):
        text = _extract_segment_text(segments[i])
        tokens = _tokenize(text)
        combined.update(tokens)
    return combined


def _window_chars(segments: list[dict], start_idx: int, end_idx: int) -> int:
    """Total character count in window (excluding whitespace)."""
    total = 0
    for i in range(start_idx, min(end_idx, len(segments))):
        total += len(_extract_segment_text(segments[i]))
    return total


def _segment_has_via_evidence(
    segment: dict,
    via_evidence: list[NormalizedVideoEvidence],
    asset_id: str | None = None,
) -> list[NormalizedVideoEvidence]:
    """Find VIA evidence overlapping with segment time window."""
    seg_start = segment.get("start", 0.0)
    seg_end = segment.get("end", seg_start)
    seg_start_us = int(round(seg_start * 1_000_000))
    seg_end_us = int(round(seg_end * 1_000_000))

    overlapping = []
    for via in via_evidence:
        if asset_id is not None and via.asset_id != asset_id:
            continue
        overlap_us = max(0, min(seg_end_us, via.end_us) - max(seg_start_us, via.start_us))
        if overlap_us > 0:
            overlapping.append(via)
    return overlapping


def _via_keywords(via_list: list[NormalizedVideoEvidence], max_keywords: int = 8) -> list[str]:
    """Extract keywords from VIA evidence descriptions/subjects/objects."""
    all_tokens = Counter()
    for via in via_list:
        # Primary: description
        tokens = _tokenize(via.description)
        all_tokens.update(tokens)

        # Secondary: subjects, objects, observed_action
        for field in (
            via.subjects,
            via.objects,
            [via.observed_action] if via.observed_action else [],
        ):
            for item in field:
                tokens = _tokenize(str(item))
                all_tokens.update(tokens)

    sorted_tokens = sorted(all_tokens.items(), key=lambda x: (-x[1], x[0]))
    return [token for token, _ in sorted_tokens[:max_keywords]]


def _continuity_group_ids(via_list: list[NormalizedVideoEvidence]) -> set[str]:
    """Extract continuity_group_id from VIA evidence."""
    return {via.continuity_group_id for via in via_list if via.continuity_group_id}


def _compute_boundary_evidence(
    segments: list[dict],
    idx: int,
    via_evidence: list[NormalizedVideoEvidence],
    config: TopicTimelineConfig,
) -> tuple[float, list[str]]:
    """
    Compute boundary evidence at position idx (between idx-1 and idx).
    Returns (boundary_score, reasons).
    boundary_score: 0.0–1.0, higher = stronger boundary evidence
    """
    reasons: list[str] = []

    # Use equal window sizes on both sides (symmetric)
    max_left = min(config.context_window, idx)
    max_right = min(config.context_window, len(segments) - idx)
    window_size = min(max_left, max_right)

    if window_size == 0:
        return 0.0, ["insufficient_window"]

    left_start = idx - window_size
    left_end = idx
    right_start = idx
    right_end = idx + window_size

    left_tokens = _window_tokens(segments, left_start, left_end)
    right_tokens = _window_tokens(segments, right_start, right_end)

    left_chars = _window_chars(segments, left_start, left_end)
    right_chars = _window_chars(segments, right_start, right_end)

    # Check minimum content
    if left_chars < config.min_chars_per_window or right_chars < config.min_chars_per_window:
        return 0.0, ["insufficient_content"]

    # Lexical similarity (weighted Jaccard TF)
    similarity = _weighted_jaccard_tf(left_tokens, right_tokens)
    distance = 1.0 - similarity  # 0.0 = identical, 1.0 = completely different

    # Lexical boundary evidence - only if distance exceeds threshold
    if distance >= config.boundary_threshold:
        reasons.append(f"lexical_distance:{distance:.2f}>={config.boundary_threshold:.2f}")
        lexical_evidence = distance
    else:
        lexical_evidence = 0.0

    # Gap penalty (secondary) - only if there's already lexical evidence
    gap_evidence = 0.0
    if idx > 0 and idx < len(segments):
        prev_end = segments[idx - 1].get("end", 0.0)
        curr_start = segments[idx].get("start", 0.0)
        gap = curr_start - prev_end
        if gap > config.gap_seconds_penalty and lexical_evidence > 0:
            gap_evidence = min(0.2, gap / 30.0)  # Cap at 0.2
            reasons.append(f"gap:{gap:.1f}s>{config.gap_seconds_penalty:.1f}s")

    # Compute left/right window times from transcript segments (not global midpoint)
    left_window_start = segments[left_start].get("start", 0.0)
    left_window_end = segments[left_end - 1].get("end", 0.0)
    right_window_start = segments[right_start].get("start", 0.0)
    right_window_end = segments[right_end - 1].get("end", 0.0)

    # VIA evidence support (secondary) - only if lexical evidence exists
    via_evidence_score = 0.0
    if via_evidence and lexical_evidence > 0:
        left_via = []
        right_via = []
        for via in via_evidence:
            via_start = via.start_us / 1_000_000
            via_end = via.end_us / 1_000_000

            # Check temporal overlap with left and right transcript windows
            left_overlap = max(
            0.0, min(via_end, left_window_end) - max(via_start, left_window_start)
        )
            right_overlap = max(
            0.0, min(via_end, right_window_end) - max(via_start, right_window_start)
        )

            # VIA participates in left window if it overlaps left transcript window
            if left_overlap > 0:
                left_via.append(via)
            # VIA participates in right window if it overlaps right transcript window
            if right_overlap > 0:
                right_via.append(via)

        left_via_kw = set(_via_keywords(left_via, max_keywords=20))
        right_via_kw = set(_via_keywords(right_via, max_keywords=20))

        if left_via_kw or right_via_kw:
            via_similarity = _weighted_jaccard_tf(
                Counter({k: 1 for k in left_via_kw}),
                Counter({k: 1 for k in right_via_kw})
            )
            via_distance = 1.0 - via_similarity
            if via_distance >= config.boundary_threshold:
                via_evidence_score = via_distance * config.via_influence_weight
                reasons.append(f"via_distance:{via_distance:.2f}")

    # Continuity group influence (secondary) - reduce split confidence if same group
    continuity_evidence = 0.0
    if via_evidence and lexical_evidence > 0:
        left_via = []
        right_via = []
        for via in via_evidence:
            via_start = via.start_us / 1_000_000
            via_end = via.end_us / 1_000_000

            # Check temporal overlap with left/right transcript windows
            left_overlap = max(
                0.0,
                min(via_end, left_window_end) - max(via.start_us / 1_000_000, left_window_start),
            )
            right_overlap = max(
                0.0,
                min(via_end, right_window_end) - max(via.start_us / 1_000_000, right_window_start),
            )

            if left_overlap > 0:
                left_via.append(via)
            if right_overlap > 0:
                right_via.append(via)

        left_groups = _continuity_group_ids(left_via)
        right_groups = _continuity_group_ids(right_via)

        if left_groups and right_groups:
            # Same group on both sides -> reduce boundary evidence
            common_groups = left_groups & right_groups
            if common_groups:
                continuity_evidence = -config.continuity_same_group_reduction * len(common_groups)
                reasons.append(f"continuity_same_group_reduction:{len(common_groups)}")

    # Combine evidence (lexical primary, others secondary)
    # Only split if COMBINED evidence exceeds boundary_threshold
    # continuity_evidence is negative (reduction), so don't cap it at 0
    total_evidence = max(
        0.0,
        min(1.0, lexical_evidence + gap_evidence + via_evidence_score + continuity_evidence),
    )

    return total_evidence, reasons


@dataclass(frozen=True)
class TopicSpan:
    """A single topic span in the timeline."""
    topic_id: str
    start: float
    end: float
    keywords: tuple[str, ...]
    representative_text: str
    confidence: float
    basis: tuple[str, ...]
    transcript_refs: tuple[str, ...]
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class TopicTimeline:
    """Complete topic timeline for a video asset."""
    asset_id: str | None
    spans: tuple[TopicSpan, ...]
    total_duration: float
    config: TopicTimelineConfig


def _stable_topic_id(
    asset_id: str | None,
    start: float,
    end: float,
    keywords: list[str],
) -> str:
    """Deterministic topic ID from canonical representation."""
    sorted_keywords = sorted(keywords)
    parts = [
        asset_id or "unknown",
        f"{start:.3f}",
        f"{end:.3f}",
        "|".join(sorted_keywords),
    ]
    canonical = "|".join(parts)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _spans_lexical_similarity(span1: TopicSpan, span2: TopicSpan) -> float:
    """Compute lexical similarity between two spans using their full token frequencies."""
    # Use the span keywords as proxy for full token frequencies
    # (Keywords are already frequency-ranked, so this is a good proxy)
    kw1 = Counter(span1.keywords)
    kw2 = Counter(span2.keywords)
    if not kw1 and not kw2:
        return 1.0
    if not kw1 or not kw2:
        return 0.0
    all_tokens = set(kw1.keys()) | set(kw2.keys())
    numerator = sum(min(kw1.get(t, 0), kw2.get(t, 0)) for t in all_tokens)
    denominator = sum(max(kw1.get(t, 0), kw2.get(t, 0)) for t in all_tokens)
    if denominator == 0:
        return 1.0
    return numerator / denominator


def _merge_similar_spans(spans: list[TopicSpan], merge_threshold: float = 0.6) -> list[TopicSpan]:
    """
    Merge adjacent spans with high lexical similarity.

    Two spans are merged if their keyword Jaccard similarity >= merge_threshold.
    This post-processing step prevents over-fragmentation of the same topic.
    """
    if len(spans) <= 1:
        return spans

    merged: list[TopicSpan] = []
    current = spans[0]

    for next_span in spans[1:]:
        similarity = _spans_lexical_similarity(current, next_span)
        if similarity >= merge_threshold:
            # Merge current and next_span - create new span with recalculated fields
            merged_start = current.start
            merged_end = next_span.end
            merged_keywords = list(current.keywords) + list(next_span.keywords)
            # Re-deduplicate and re-rank keywords
            kw_counter = Counter(merged_keywords)
            sorted_kw = sorted(kw_counter.items(), key=lambda x: (-x[1], x[0]))
            new_keywords = tuple(k for k, _ in sorted_kw[:8])

            # Combine other fields
            new_rep_text = (
            current.representative_text
            if len(current.representative_text) >= len(next_span.representative_text)
            else next_span.representative_text
        )
            new_confidence = max(current.confidence, next_span.confidence)
            new_basis = tuple(sorted(set(current.basis + next_span.basis)))
            new_transcript_refs = current.transcript_refs + next_span.transcript_refs
            new_evidence_refs = tuple(sorted(set(current.evidence_refs + next_span.evidence_refs)))

            current = TopicSpan(
                topic_id="",  # Will be recalculated after all merges
                start=merged_start,
                end=merged_end,
                keywords=new_keywords,
                representative_text=new_rep_text,
                confidence=new_confidence,
                basis=new_basis,
                transcript_refs=new_transcript_refs,
                evidence_refs=new_evidence_refs,
            )
        else:
            merged.append(current)
            current = next_span

    merged.append(current)
    return merged


def _smooth_single_phrase_excursions(
    spans: list[TopicSpan],
    segments: list[dict],
    config: TopicTimelineConfig,
) -> list[TopicSpan]:
    """
    P1-2: Smooth single-phrase excursions.

    If a span consists of exactly 1 segment and is surrounded by spans
    with high lexical similarity (both left and right neighbors are similar),
    merge it into the surrounding topic to avoid over-fragmentation.
    """
    if len(spans) <= 2:
        return spans

    # For simplicity, we'll use a heuristic: if a span has exactly 1 transcript_ref,
    # it likely corresponds to a single segment
    single_span_indices = []
    for i, span in enumerate(spans):
        if len(span.transcript_refs) == 1:
            single_span_indices.append(i)

    if not single_span_indices:
        return spans

    merged_spans = []
    i = 0
    while i < len(spans):
        if i in single_span_indices and 0 < i < len(spans) - 1:
            # Check if this single-phrase span is an excursion
            # But only if the LEFT span hasn't already been added to merged_spans
            # (i.e., we haven't already processed it as part of a previous excursion)
            left_idx = i - 1
            right_idx = i + 1

            # Check if left span is already in merged_spans (as the last element)
            left_already_merged = (
                len(merged_spans) > 0
                and merged_spans[-1].start == spans[left_idx].start
            )

            left_span = spans[left_idx]
            right_span = spans[right_idx]

            # Check if left and right spans are lexically similar (excursion pattern)
            left_right_similarity = _spans_lexical_similarity(left_span, right_span)
            excursion_threshold = 0.5  # High similarity between left and right

            if left_right_similarity >= excursion_threshold:
                # This is an excursion - merge all three spans

                merged_keywords = (
                    list(left_span.keywords)
                    + list(spans[i].keywords)
                    + list(right_span.keywords)
                )
                kw_counter = Counter(merged_keywords)
                sorted_kw = sorted(kw_counter.items(), key=lambda x: (-x[1], x[0]))
                new_keywords = tuple(k for k, _ in sorted_kw[:8])

                new_rep_text = (
                    left_span.representative_text
                    if len(left_span.representative_text) >= len(right_span.representative_text)
                    else right_span.representative_text
                )
                new_confidence = max(left_span.confidence, spans[i].confidence, right_span.confidence)
                new_basis = tuple(
                    sorted(
                        set(
                            left_span.basis
                            + spans[i].basis
                            + right_span.basis
                            + ("single_phrase_excursion",)
                        )
                    )
                )
                new_transcript_refs = (
                    left_span.transcript_refs
                    + spans[i].transcript_refs
                    + right_span.transcript_refs
                )
                new_evidence_refs = tuple(
                    sorted(
                        set(
                            left_span.evidence_refs
                            + spans[i].evidence_refs
                            + right_span.evidence_refs
                        )
                    )
                )

                # If left span was already added to merged_spans, remove it
                if left_already_merged:
                    merged_spans.pop()

                merged_span = TopicSpan(
                    topic_id="",
                    start=left_span.start,
                    end=right_span.end,
                    keywords=new_keywords,
                    representative_text=new_rep_text,
                    confidence=new_confidence,
                    basis=new_basis,
                    transcript_refs=new_transcript_refs,
                    evidence_refs=new_evidence_refs,
                )
                merged_spans.append(merged_span)
                i += 3  # Skip the three merged spans
                continue

            merged_spans.append(spans[i])

        else:
            merged_spans.append(spans[i])

        i += 1

    return merged_spans


def _handle_large_gaps(
    spans: list[TopicSpan],
    segments: list[dict],
    via_evidence: list[NormalizedVideoEvidence],
    asset_id: str | None,
    config: TopicTimelineConfig,
) -> list[TopicSpan]:
    """
    P1-4: Handle large gaps in transcript coverage by splitting spans.

    Large gaps should create explicit boundaries. If there's a large gap between
    segments, split the span at the gap point. No span should cross a large gap.
    """
    if not spans or not segments:
        return spans


    # Find actual segment boundaries
    segment_ends = [seg.get("end", 0.0) for seg in segments]
    segment_starts = [seg.get("start", 0.0) for seg in segments]

    # Find large gaps between consecutive segments
    large_gaps = []
    for i in range(len(segments) - 1):
        gap = segment_starts[i + 1] - segment_ends[i]
        if gap >= config.large_gap_threshold:
            # Gap boundary is between last_segment.end and next_segment.start
            large_gaps.append({
                'gap_start': segment_ends[i],
                'gap_end': segment_starts[i + 1],
                'gap_duration': segment_starts[i + 1] - segment_ends[i],
            })

    if not large_gaps:
        return spans

    # Split spans at large gap boundaries - NO span may cross a large gap
    new_spans = []
    for span in spans:
        # Find gaps that fall strictly within this span
        span_gaps = [
            g for g in large_gaps if span.start < g['gap_start'] and g['gap_end'] < span.end
        ]
        if not span_gaps:
            new_spans.append(span)
            continue

        # Split span at each gap boundary
        current_start = span.start
        for gap in span_gaps:
            gap['gap_start']
            gap['gap_end']

            # Create sub-span BEFORE the gap (ends at gap_start)
            if gap['gap_start'] > current_start:
                # Find segments that fall in [current_start, gap_start]
                sub_start_idx = next(
                    (
                        j
                        for j, seg in enumerate(segments)
                        if seg.get("end", 0.0) > current_start + 1e-9
                    ),
                    0,
                )
                sub_end_idx = next(
                    (
                        j + 1
                        for j, seg in enumerate(segments)
                        if seg.get("end", 0.0) >= gap['gap_start'] - 1e-9
                    ),
                    len(segments),
                )

                if sub_start_idx < sub_end_idx:
                    new_spans.append(_materialize_subspan(
                        segments=segments,
                        via_evidence=via_evidence,
                        asset_id=asset_id,
                        config=config,
                        start_idx=sub_start_idx,
                        end_idx=sub_end_idx,
                        span_start=current_start,
                        span_end=gap['gap_start'],
                        additional_basis=("transcript_gap",),
                    ))

            # Move to after the gap
            current_start = gap['gap_end']

        # Final sub-span after last gap - limit to span.end
        if current_start < span.end - 1e-9:
            sub_start_idx = next(
                (
                    j
                    for j, seg in enumerate(segments)
                    if seg.get("start", 0.0) >= current_start - 1e-9
                ),
                len(segments) - 1,
            )
            # Limit to segments that end before or at span.end
            sub_end_idx = next(
                (
                    j + 1
                    for j, seg in enumerate(segments)
                    if seg.get("end", 0.0) >= span.end - 1e-9
                ),
                len(segments),
            )

            if sub_start_idx < sub_end_idx:
                # Materialize final sub-span using shared function
                new_spans.append(_materialize_subspan(
                    segments=segments,
                    via_evidence=via_evidence,
                    asset_id=asset_id,
                    config=config,
                    start_idx=sub_start_idx,
                    end_idx=sub_end_idx,
                    span_start=current_start,
                    span_end=span.end,
                    additional_basis=("transcript_gap",),
                ))
    return new_spans


def _materialize_subspan(
    segments: list[dict],
    via_evidence: list[NormalizedVideoEvidence],
    asset_id: str | None,
    config: TopicTimelineConfig,
    start_idx: int,
    end_idx: int,
    span_start: float,
    span_end: float,
    additional_basis: tuple[str, ...] = (),
) -> TopicSpan:
    """
    Materialize a TopicSpan from a contiguous range of transcript segments.

    Single source of truth for TopicSpan construction — used by both initial
    span building and gap splitting to ensure consistent computation.
    """
    if start_idx >= end_idx:
        # Should not happen, but guard anyway
        return TopicSpan(
            topic_id=_stable_topic_id(None, span_start, span_end, []),
            start=span_start,
            end=span_end,
            keywords=(),
            representative_text="",
            confidence=0.0,
            basis=("empty_subspan",),
            transcript_refs=(),
            evidence_refs=(),
        )

    # Aggregate keywords from segments
    span_tokens = Counter()
    for j in range(start_idx, min(end_idx, len(segments))):
        text = _extract_segment_text(segments[j])
        tokens = _tokenize(text)
        span_tokens.update(tokens)

    # Top keywords: freq DESC, token ASC
    sorted_keywords = sorted(span_tokens.items(), key=lambda x: (-x[1], x[0]))
    keywords = [token for token, _ in sorted_keywords[:8]]

    # Representative text: segment with highest keyword density
    set(keywords)
    _find_representative_segment(
        segments, start_idx, end_idx, set(keywords)
    )

    # Transcript refs
    transcript_refs = tuple(
        f"{segments[j]['start']:.3f}-{segments[j]['end']:.3f}"
        for j in range(start_idx, min(end_idx, len(segments)))
    )

    # Evidence refs
    evidence_refs = []
    span_via_evidence = []
    for j in range(start_idx, min(end_idx, len(segments))):
        seg_via = _segment_has_via_evidence(segments[j], via_evidence, asset_id)
        span_via_evidence.extend(seg_via)
        evidence_refs.extend([v.evidence_id for v in seg_via])

    # Confidence: lexical cohesion + VIA reliability + transcript coverage
    span_duration = span_end - span_start

    # Lexical cohesion: average adjacent segment similarity
    cohesion = 1.0
    if end_idx - start_idx > 1:
        similarities = []
        for j in range(start_idx, min(end_idx - 1, len(segments))):
            text1 = _extract_segment_text(segments[j])
            text2 = _extract_segment_text(segments[j + 1])
            if text1 and text2:
                tf1 = _compute_tf(_tokenize(text1))
                tf2 = _compute_tf(_tokenize(text2))
                similarities.append(_weighted_jaccard_tf(tf1, tf2))
        cohesion = sum(similarities) / len(similarities) if similarities else 0.0
    else:
        cohesion = 1.0

    # VIA support
    via_reliability = 0.0
    span_via_evidence = []
    for j in range(start_idx, min(end_idx, len(segments))):
        seg_via = _segment_has_via_evidence(segments[j], via_evidence, asset_id)
        span_via_evidence.extend(seg_via)

    via_reliability = 0.0
    if span_via_evidence:
        reliabilities = []
        for via in span_via_evidence:
            coverage_factor = 1.0
            if via.coverage_ratio is not None:
                coverage_factor = via.coverage_ratio
            elif via.partial:
                coverage_factor = 0.5
            elif via.failed_chunks:
                coverage_factor = 0.7
            reliabilities.append(via.confidence_original * coverage_factor)
        via_reliability = sum(reliabilities) / len(reliabilities)

    min(1.0, len(span_via_evidence) / 5.0) * via_reliability if span_via_evidence else 0.0

    # Transcript coverage
    span_duration = span_end - span_start
    if span_duration > 0:
        transcript_covered = sum(
            segments[j].get("end", 0.0) - segments[j].get("start", 0.0)
            for j in range(start_idx, min(end_idx, len(segments)))
        )
        transcript_coverage = min(1.0, transcript_covered / span_duration)
    else:
        transcript_coverage = 0.0

    # Lexical cohesion
    cohesion = 1.0
    if end_idx - start_idx > 1:
        similarities = []
        for j in range(start_idx, min(end_idx - 1, len(segments))):
            text1 = _extract_segment_text(segments[j])
            text2 = _extract_segment_text(segments[j + 1])
            if text1 and text2:
                tf1 = _compute_tf(_tokenize(text1))
                tf2 = _compute_tf(_tokenize(text2))
                similarities.append(_weighted_jaccard_tf(tf1, tf2))
        cohesion = sum(similarities) / len(similarities) if similarities else 0.0

    # Confidence: 0.4*cohesion + 0.3*via_reliability + 0.3*transcript_coverage
    transcript_covered = sum(
        segments[j].get("end", 0.0) - segments[j].get("start", 0.0)
        for j in range(start_idx, min(end_idx, len(segments)))
    )
    transcript_coverage = min(
        1.0, transcript_covered / (span_end - span_start)
    ) if (span_end - span_start) > 0 else 0.0
    via_reliability = sum(
        v.confidence_original
        * (v.coverage_ratio or (0.5 if v.partial else 0.7 if v.failed_chunks else 1.0))
        for v in span_via_evidence
    ) / len(span_via_evidence) if span_via_evidence else 0.0
    confidence = _clamp01(
        0.4 * cohesion
        + 0.3 * (via_reliability if span_via_evidence else 0.0)
        + 0.3 * transcript_coverage
    )

    # Keywords
    span_tokens = Counter()
    for j in range(start_idx, min(end_idx, len(segments))):
        text = _extract_segment_text(segments[j])
        tokens = _tokenize(text)
        span_tokens.update(tokens)
    sorted_keywords = sorted(span_tokens.items(), key=lambda x: (-x[1], x[0]))
    keywords = [token for token, _ in sorted_keywords[:8]]

    # Representative text
    _find_representative_segment(segments, start_idx, end_idx, set(keywords))

    # Transcript refs
    transcript_refs = tuple(
        f"{segments[j]['start']:.3f}-{segments[j]['end']:.3f}"
        for j in range(start_idx, min(end_idx, len(segments)))
    )

    # Evidence refs
    evidence_refs = []
    for j in range(start_idx, min(end_idx, len(segments))):
        seg_via = _segment_has_via_evidence(segments[j], via_evidence, asset_id)
        evidence_refs.extend([v.evidence_id for v in seg_via])

    # Basis
    basis = tuple(sorted(set(additional_basis)))

    # Topic ID
    topic_id = _stable_topic_id(asset_id, span_start, span_end, keywords)

    return TopicSpan(
        topic_id=topic_id,
        start=span_start,
        end=span_end,
        keywords=tuple(keywords),
        representative_text=_find_representative_segment(
            segments, start_idx, end_idx, set(keywords)
        ),
        confidence=confidence,
        basis=basis,
        transcript_refs=transcript_refs,
        evidence_refs=tuple(sorted(set(evidence_refs))),
    )


def _find_representative_segment(
    segments: list[dict],
    span_start_idx: int,
    span_end_idx: int,
    span_keywords: set[str],
) -> str:
    """Find the segment with highest keyword density in the span."""
    best_segment = None
    best_density = -1.0
    best_start = float("inf")

    for i in range(span_start_idx, min(span_end_idx, len(segments))):
        seg = segments[i]
        text = _extract_segment_text(seg)
        if not text:
            continue

        tokens = _tokenize(text)
        if not tokens:
            continue

        keyword_count = sum(1 for t in tokens if t in span_keywords)
        density = keyword_count / len(tokens)

        seg_start = seg.get("start", 0.0)
        if (
            density > best_density
            or (abs(density - best_density) < 1e-9 and seg_start < best_start)
        ):
            best_density = density
            best_segment = text
            best_start = seg_start

    return best_segment if best_segment else ""


def build_topic_timeline(
    transcript_segments: list[dict],
    via_evidence: list[NormalizedVideoEvidence] | None = None,
    video_duration: float = 0.0,
    asset_id: str | None = None,
    config: TopicTimelineConfig | None = None,
) -> TopicTimeline:
    """
    Build a deterministic topic timeline from transcript + VIA evidence.

    Args:
        transcript_segments: List of dicts with start, end, text, words (from Whisper)
        via_evidence: Optional VIA normalized evidence for secondary signals
        video_duration: Optional video duration for clamping
        asset_id: Optional asset ID for topic ID stability
        config: Optional configuration (uses DEFAULT_TOPIC_CONFIG if None)

    Returns:
        TopicTimeline with ordered, non-overlapping TopicSpan entries.
    """
    config = config or DEFAULT_TOPIC_CONFIG
    via_evidence = via_evidence or []

    # Single-asset policy enforcement
    unique_via_assets = {v.asset_id for v in via_evidence}
    if asset_id is not None:
        # Explicit asset_id provided: filter strictly
        via_evidence = [v for v in via_evidence if v.asset_id == asset_id]
    else:
        # No explicit asset_id: enforce single-asset policy
        if len(unique_via_assets) > 1:
            raise ValueError(
                f"Multiple distinct asset_ids in VIA evidence ({sorted(unique_via_assets)}) "
                f"but no asset_id specified. Provide explicit asset_id to disambiguate."
            )
        # 0 or 1 unique asset_ids: OK to proceed
        if len(unique_via_assets) == 1:
            asset_id = unique_via_assets.pop()

    # Validate and sort transcript segments deterministically
    valid_segments = []
    for seg in transcript_segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        text = str(seg.get("text", "")).strip()

        # Skip invalid segments
        if start < 0 or end <= start or not text:
            continue

        # Clamp to video duration if provided
        if video_duration > 0:
            start = min(start, video_duration)
            end = min(end, video_duration)
            if end <= start:
                continue

        valid_segments.append({
            "start": start,
            "end": end,
            "text": text,
            "words": seg.get("words", []),
        })

    # Deterministic sort: start ASC, end ASC, text ASC
    valid_segments.sort(key=lambda s: (s["start"], s["end"], s["text"]))

    if not valid_segments:
        return TopicTimeline(
            asset_id=asset_id,
            spans=(),
            total_duration=video_duration,
            config=config,
        )

    n = len(valid_segments)

    # Compute boundary evidence at each potential boundary position
    boundary_evidence: list[float] = [0.0] * (n + 1)  # 0 and n are boundaries
    boundary_reasons: list[list[str]] = [[] for _ in range(n + 1)]

    # Boundaries 0 and n are always present (start/end of timeline)
    boundary_evidence[0] = 1.0
    boundary_reasons[0] = ["timeline_start"]
    boundary_evidence[n] = 1.0
    boundary_reasons[n] = ["timeline_end"]

    # Internal boundaries (between segments)
    for i in range(1, n):
        evidence, reasons = _compute_boundary_evidence(
            valid_segments, i, via_evidence, config
        )
        boundary_evidence[i] = evidence
        boundary_reasons[i] = reasons

    # Build spans from boundaries
    spans: list[TopicSpan] = []
    span_start_idx = 0

    for i in range(1, n + 1):
        # Check if we should split at boundary i
        # Use config.boundary_threshold as the split threshold (not hardcoded 0.5)
        should_split = boundary_evidence[i] >= config.boundary_threshold or i == n

        if should_split:
            span_end_idx = i  # exclusive

            # Aggregate keywords for this span
            span_tokens = Counter()
            for j in range(span_start_idx, span_end_idx):
                text = _extract_segment_text(valid_segments[j])
                tokens = _tokenize(text)
                span_tokens.update(tokens)

            # Top keywords: freq DESC, token ASC
            sorted_keywords = sorted(span_tokens.items(), key=lambda x: (-x[1], x[0]))
            keywords = [token for token, _ in sorted_keywords[:config.max_keywords_per_span]]

            # Span boundaries
            span_start = valid_segments[span_start_idx]["start"]
            span_end = valid_segments[span_end_idx - 1]["end"]

            # Representative text
            span_keyword_set = set(keywords)
            rep_text = _find_representative_segment(
                valid_segments, span_start_idx, span_end_idx, span_keyword_set
            )

            # Collect transcript refs (segment identifiers)
            transcript_refs = [
                f"{valid_segments[j]['start']:.3f}-{valid_segments[j]['end']:.3f}"
                for j in range(span_start_idx, span_end_idx)
            ]

            # Collect evidence refs
            evidence_refs = []
            for j in range(span_start_idx, span_end_idx):
                seg_via = _segment_has_via_evidence(valid_segments[j], via_evidence, asset_id)
                evidence_refs.extend([v.evidence_id for v in seg_via])

            # Confidence: lexical persistence + VIA support + coverage
            span_duration = span_end - span_start
            _ = sum(
                len(_extract_segment_text(valid_segments[j]))
                for j in range(span_start_idx, span_end_idx)
            )

            # Lexical persistence: average within-span similarity to neighbors
            # Simplified: use token diversity as proxy
            unique_tokens = len(span_tokens)
            total_tokens = sum(span_tokens.values())
            _ = unique_tokens / max(1, total_tokens)

            # VIA support - compute reliability-weighted support
            span_via_evidence = []
            for j in range(span_start_idx, span_end_idx):
                seg_via = _segment_has_via_evidence(valid_segments[j], via_evidence, asset_id)
                span_via_evidence.extend(seg_via)

            _ = len(evidence_refs)
            # Compute VIA reliability: average of (confidence_original * coverage_factor)
            via_reliability = 0.0
            if span_via_evidence:
                reliabilities = []
                for via in span_via_evidence:
                    coverage_factor = 1.0
                    if via.coverage_ratio is not None:
                        coverage_factor = via.coverage_ratio
                    elif via.partial:
                        coverage_factor = 0.5
                    elif via.failed_chunks:
                        coverage_factor = 0.7
                    # Penalize partial/low coverage
                    reliabilities.append(via.confidence_original * coverage_factor)
                via_reliability = sum(reliabilities) / len(reliabilities)
            _ = min(
            1.0, len(span_via_evidence) / 5.0
        ) * via_reliability if span_via_evidence else 0.0

            # Coverage: actual transcript coverage ratio
            span_duration = span_end - span_start
            if span_duration > 0:
                transcript_covered = sum(
                    valid_segments[j].get("end", 0.0) - valid_segments[j].get("start", 0.0)
                    for j in range(span_start_idx, span_end_idx)
                )
                transcript_coverage = min(1.0, transcript_covered / span_duration)
            else:
                transcript_coverage = 0.0

            # Lexical cohesion: average similarity between adjacent segments within span
            cohesion = 0.0
            if span_end_idx - span_start_idx > 1:
                similarities = []
                for j in range(span_start_idx, span_end_idx - 1):
                    text1 = _extract_segment_text(valid_segments[j])
                    text2 = _extract_segment_text(valid_segments[j + 1])
                    if text1 and text2:
                        tf1 = _compute_tf(_tokenize(text1))
                        tf2 = _compute_tf(_tokenize(text2))
                        similarities.append(_weighted_jaccard_tf(tf1, tf2))
                cohesion = sum(similarities) / len(similarities) if similarities else 0.0
            else:
                # Single segment: max cohesion by definition
                cohesion = 1.0

            # Combined confidence:
            # 0.4 * cohesion + 0.3 * via_reliability + 0.3 * transcript_coverage
            confidence = _clamp01(
                0.4 * cohesion
                + 0.3 * (via_reliability if span_via_evidence else 0.0)
                + 0.3 * transcript_coverage
            )

            topic_id = _stable_topic_id(asset_id, span_start, span_end, keywords)

            # Flatten boundary reasons for basis - use sorted for determinism
            flat_basis = []
            for i in range(span_start_idx + 1, span_end_idx):
                flat_basis.extend(boundary_reasons[i])
            basis = tuple(sorted(set(flat_basis)))

            spans.append(TopicSpan(
                topic_id=topic_id,
                start=span_start,
                end=span_end,
                keywords=tuple(keywords),
                representative_text=rep_text,
                confidence=confidence,
                basis=basis,
                transcript_refs=tuple(transcript_refs),
                evidence_refs=tuple(sorted(set(evidence_refs))),
            ))

            span_start_idx = span_end_idx

    # Post-process: merge adjacent spans with high lexical similarity
    # This prevents over-fragmentation of the same topic due to minor vocabulary variations
    merge_threshold = getattr(config, 'adjacent_merge_threshold', 0.2)
    spans = _merge_similar_spans(spans, merge_threshold=merge_threshold)

    # P1-2: Single-phrase excursion smoothing
    # If a span consists of exactly 1 segment and is surrounded by spans
    # with high lexical similarity, merge it into the surrounding topic
    spans = _smooth_single_phrase_excursions(spans, valid_segments, config)

    # P1-4: Large gap handling - ensure large gaps create boundaries
    # with explicit basis rather than silent merge
    spans = _handle_large_gaps(spans, valid_segments, via_evidence, asset_id, config)

    # Recalculate topic_ids after merging (since end/keywords may have changed)
    recalculated_spans = []
    for span in spans:
        if span.topic_id == "":
            # Recalculate with final start/end/keywords
            new_topic_id = _stable_topic_id(asset_id, span.start, span.end, list(span.keywords))
            span = TopicSpan(
                topic_id=new_topic_id,
                start=span.start,
                end=span.end,
                keywords=span.keywords,
                representative_text=span.representative_text,
                confidence=span.confidence,
                basis=span.basis,
                transcript_refs=span.transcript_refs,
                evidence_refs=span.evidence_refs,
            )
        recalculated_spans.append(span)

    return TopicTimeline(
        asset_id=asset_id,
        spans=tuple(recalculated_spans),
        total_duration=video_duration,
        config=config,
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


# ──────────────────────────────────────────────────────────────────────────────
# Convenience query functions (read-only, no mutations)
# ──────────────────────────────────────────────────────────────────────────────

def topic_for_time(timeline: TopicTimeline, time_s: float) -> TopicSpan | None:
    """Find the topic span containing a given timestamp."""
    for span in timeline.spans:
        if span.start <= time_s < span.end:
            return span
    # Check if exactly at end of last span
    if timeline.spans and abs(time_s - timeline.spans[-1].end) < 1e-9:
        return timeline.spans[-1]
    return None


def topic_for_range(timeline: TopicTimeline, start_s: float, end_s: float) -> list[TopicSpan]:
    """Find all topic spans overlapping with a time range."""
    result = []
    for span in timeline.spans:
        if span.end > start_s and span.start < end_s:
            result.append(span)
    return result