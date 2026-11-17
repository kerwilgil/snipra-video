# Event model

Conceptual `EventCandidate` fields are `id`, `start`, `peak`, `end`, `setup_start`, `event_start`, `payoff_timestamp`, `reaction_end`, `event_type`, `outcome`, `importance`, `coherence`, `completeness`, `payoff_strength`, `reaction_value`, `hook_potential`, `highlight_potential`, `montage_value`, `menu_contamination`, `dead_time`, `temporal_discontinuity`, `evidence`, and `evidence_quality`.

This is a future interface, not a production schema. Store observations, conflicts, and source ranges beside any claim. `outcome` can be domain-specific; when evidence cannot distinguish win/loss, kill/death, or another resolution, retain `uncertain`.
