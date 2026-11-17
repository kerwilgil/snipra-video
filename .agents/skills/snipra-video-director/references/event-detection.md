# Event detection and temporal evidence

Inspect only signals Snipra exposes: candidate ranges, transcript, OCR, scene boundaries, audio peaks, score breakdown, legacy vision output, structured visual features, peak/evidence frames, and source ranges. Explicitly say when a signal is unavailable.

The current visual pass commonly samples about four 512px frames around 1/5, 2/5, 3/5, and 4/5 of a candidate. It can recognize states, but is weak at temporal relations such as setup/payoff, fatalities, combos, transformations, reveals, failures, and reactions.

Future hypothesis, not production behavior: retain the economical first pass, then inspect top-K events with an Event Filmstrip of roughly 8–12 chronological timestamped frames. Existing transcript, OCR, and audio markers may annotate it. Estimate roughly 2–3x image bytes/context or one extra VLM request per selected event; measure actual latency and human-ground-truth benefit before adoption.
