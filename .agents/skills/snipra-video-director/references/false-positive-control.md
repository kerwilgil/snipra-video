# False-positive control and uncertainty

Treat menus, loading, idle, uninformative scoreboards, incomplete events, false payoff, excessive aftermath, unrelated joins, and abrupt discontinuity as negative evidence. A high legacy/VLM score is one observation, not a verdict.

Use structured features as guardrails: `menu_or_idle=true` raises a strong rejection/rework alert; `evidence_quality=insufficient` blocks strong semantic claims; `outcome=uncertain` withholds outcome-specific assertions; a clear, evidenced payoff can corroborate an event without ranking it alone. When signals conflict, preserve `uncertain` instead of forcing kill, death, victory, win, or loss.
