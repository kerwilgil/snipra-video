# Highlight director

A Highlight is one coherent event. Prefer `setup → escalation → payoff`, or `event → payoff` where context is already clear. Include reaction only when it adds meaning.

Do not join distant events merely to fill duration or temporal diversity. Menus, loading, dead aftermath, missing payoff, and unexplained jumps require rejection or boundary rework. If multiple independent moments are intended, change the intent to Montage.

## Enhanced Event Pool usage (1.5H+)

When `targeted_evidence.enhanced_event_pool` is present:

- **Refined ranges** (`refined_start`, `refined_end`): Use as evidence for boundary decisions. These are deterministic outputs from 1.5D, not editorial mandates.
- **Peak / payoff / reaction windows** (`peak_at`, `payoff_window_start/end`, `reaction_window_start/end`): Temporal anchors from 1.5D. Use to verify coherence of the selected event.
- **Topic contexts**: From 1.5E. Check `overlap_ratio` to understand topical coherence of the event.
- **Cluster awareness**: If the selected event is a non-representative member of a cluster, consider whether a near-duplicate representative better serves the highlight (but non-representatives remain selectable if editorially justified).
- **Representative is advisory**: `representative_event_id` in a cluster is the lowest-rank member by 1.5C score. It is a structural signal, not a KEEP directive. The Director may select any member.
- **Diverse order is advisory**: `diverse_order_event_ids` orders representatives first for variety inspection. It does not constrain selection.

**Selection rule**: Exactly one coherent event is required. The Director chooses based on editorial judgment using all available evidence (legacy pool, enhanced pool, VIA, transcript). The enhanced pool provides richer context; it does not replace editorial decision-making.