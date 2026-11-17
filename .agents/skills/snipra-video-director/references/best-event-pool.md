# Best Event Pool

Maintain one ranked, inspectable shortlist after the Quality Gate. Each entry retains candidate ID, source ranges, importance, coherence, completeness, payoff, evidence quality, outcome certainty, novelty, overlap/diversity, human labels when available, and eligible output intents.

Rank comparatively within a source/task before trusting independent absolute scores. Legacy VLM remains a secondary signal; structured features remain guardrails. Relative ranking is a promising experimental nomination signal, not active production logic. The pool feeds Auto and future Editor IA so intents select from shared event truth rather than rebuilding isolated detectors.

## Enhanced Event Pool (1.5H+)

`targeted_evidence.enhanced_event_pool` may be present when the Highlight Intelligence pipeline (1.5C–1.5G) has been executed upstream. It contains:

- **events**: Rank-ordered list (rank ASC per 1.5C authority) with full context:
  - `refined_start`, `refined_end` (1.5D deterministic boundaries)
  - `peak_at`, `payoff_window`, `reaction_window` (1.5D anchors)
  - `topic_contexts` with overlap ratios (1.5E temporal topics)
  - `score_components` trace (1.5C deterministic breakdown)
  - `evidence_refs` union across C/D/E

- **clusters**: Near-duplicate groups (1.5G representative-star clustering):
  - `representative_event_id` = lowest-rank member (structural, not editorial)
  - `relations` with overlap ratio, topic match, peak distance
  - `topic_ids` union

- **diverse_order_event_ids**: Representatives first (by cluster order), then remaining by rank ASC

**Authority contract**:
- `rank` = relative nomination order (1.5C), NOT selection mandate
- `overall_score` = comparative signal (1.5C), NOT quality gate
- `refinement_confidence` = boundary evidence strength (1.5D), NOT editorial merit
- `representative_event_id` = structural cluster center, NOT KEEP directive
- `diverse_order_event_ids` = inspection priority, NOT final selection

The Director retains sole editorial authority. Non-representatives remain selectable. Same-cluster selections in Montage are permitted. The Director may choose any event from the pool based on editorial judgment.