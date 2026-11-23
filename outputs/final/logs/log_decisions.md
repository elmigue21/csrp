# Decisions

| Date | Item | Rule | Decided by |
|---|---|---|---|
| 2026-11-09 | Order of work | Extract every signal with the current cleaning rules first; decide C1-C10 and D1-D10 from the results | team |
| 2026-11-09 | Mode | Not computed: undefined for continuous data without arbitrary rounding | team |
| 2026-11-11 | cv for gaze x/y | Computed and kept, tagged 'position': gaze angles have no true zero, so cv depends on how the headset sat; report with that caveat if it ranks highly | team |
| 2026-11-11 | Counts and durations | Not features: they measure how long the activity lasted (A4 runs longer), not load; rates are used instead | team |
| 2026-11-11 | Valid-sample % | Reported, never a feature: A4 loses more data, so it would reveal the label | team |
| 2026-11-09 | Feature-decision rules | Decide before any model runs and never change a decision because of results; drop only for data quality (blanks that follow the condition, stuck values, exact copies) | team |
| 2026-11-09 | D1 Blink duration (17 features) | Dropped: with the 20-value rule (D2) blank in 79 of 88 recordings (43 A1, 36 A4); 8 of the 9 with values are A4, so whether a value exists reveals the label; blink_rate is kept | team |
| 2026-11-09 | D6a saccade_duration__min | Dropped: one sample (4.17 ms) in 83 of 88 recordings, so it measures the tracker | team |
| 2026-11-09 | D6b Other saccade-duration statistics | Kept: rounded to 4.17 ms steps but still differ between recordings; stated as a limitation | team |
| 2026-11-09 | D7 fixation_duration__min | Kept: 42% of recordings at the 55 ms floor, the rest spread over 9 values | team |
| 2026-11-09 | D8 Variance (9 features) | Dropped: variance is SD squared, an exact copy that would split importance with SD | team |
| 2026-11-09 | D9 Range for 6 signals | Dropped for gaze speed, fixation duration and the 4 saccade signals: their minimum is fixed, so range copies max (rank correlation >= 0.999); kept for pupil and gaze x/y | team |
| 2026-11-09 | D10 Gaze position (20 features; 22 with gaze cv) | Kept and tagged 'position'; if one ranks highly, report it with a screen-content caveat | team |
| 2026-11-11 | D2 Minimum values per signal | A signal's statistics are blank when a recording has fewer than 20 values for it | team |
| 2026-11-11 | D3 min, max and range vs recording length | Kept; reported as a limitation (A4 runs longer than A1), checked against p05/p95 | team |
| 2026-11-11 | D4 Signal list | Kept the 10 signals; per-eye pupil and pupil rate of change not added | team |
| 2026-11-11 | Per-person standardization | Not done: the feature table holds each feature's value as extracted; any scaling a model needs is fitted on the training people inside each LOPO fold | team (post-defense) |
| 2026-11-11 | A2 and A3 | Left out of the feature table: they have no label and were only used for per-person standardization; they still count in the C9 event-detection check | team (post-defense) |
| 2026-11-11 | C1 Line up the files in time | Kept: pupil samples and blinks outside the gaze time range are removed (mainly P18 A4, whose pupil file runs up to 2 hours past the activity) | team |
| 2026-11-11 | C2 Low-confidence samples | Kept: gaze and pupil samples with confidence below 0.8 are treated as missing (6.7% of gaze samples; 0.6 or 0.7 would give 6.3-6.4%) | team |
| 2026-11-11 | C3-C10 Remaining cleaning steps | Kept as defended: >35% invalid gaze drops a recording (C3); gaps over 1 s not counted (C4); blinks 50-500 ms, merged under 100 ms (C5); gaze gaps under 75 ms filled (C6); pupil cleaning (C7); gaze as both-eye angles (C8); I-VT 45 deg/s, fixations >= 55 ms (C9); valid-sample share reported, never a feature (C10) | team |
| 2026-11-11 | Pupil quality | No extra exclusion: 7 A4 recordings keep under 80% of pupil data and P44 A4's right eye reads too small; C7 already removes impossible values, so they are kept and reported as a limitation | team |
| 2026-10-01 | Recording exclusion (C3) | Drop a recording with more than 35% invalid gaze; drop a participant missing A1 or A4 | team (defended version) |
| 2026-10-15 | P31 | Left out: its files are a byte-identical copy of P28's (R5-1) | team (defended version) |
