# RRL review for the Pina draft (v5)

**Date:** 2026-11-18. **Checked:** every citation in `CSRP-Chapter-1-to-5-draft-Pina-v5.md` (54 keys), against
the switch from COLET (Pupil Core, screen, chin rest, whole recordings) to Pina et al. (Varjo XR-4 VR headset,
head free, 30 s windows) and the method changes (event features dropped, one-feature baseline added).
`{n}` keys are the **v4-6 numbers** (the build script renumbers them); `rrl-by-topic.md` uses the same numbers.

## Summary
- **Keep as is:** 42. They are about the construct, models, validation, metrics or importance, which did not change.
- **Change the sentence:** 4 ({18}, {23}, {43}, {44}).
- **Replace (Pupil Core / screen-specific, no longer fits):** 2 ({30}, {71}).
- **Move to a better place:** 2 ({66} Nenna is a VR workload study; {72} Kothari is about eye and head moving together).
- **Add (needed, not yet in the list):** {pina}, {bielsa}; 1–2 VR eye-tracking workload sources (candidates below).
- **Already gone from v5 (COLET-only):** Fuhl, Wibirama, Dell'Acqua, Fenoglio, Skaramagkas, Kassner (Pupil), Hausamann,
  Trabulsi, Dormann, Foulsham, Wu, Bargary, Xu, Hogervorst, Van Calster, Demšar, and others in the COLET section.
- How to explain it: "We kept the sources about cognitive load, the models and the validation, because those did not
  change. We removed or replaced the sources that were about the old tracker and the old screen setup, and added
  sources about eye tracking in VR."

## Needs action
| Key | Source | Where in v5 | Problem | Action |
|---|---|---|---|---|
| {30} | Faraji et al. 2023, Pupil Labs Core toolkit | Table 3: sample validity, gaze gaps | About Pupil Core; Varjo marks validity itself | **Replace** validity row source with {bielsa} (describes the Varjo data) or mark as a design choice; keep {30} off |
| {71} | Velisar & Shanidze 2024, Pupil Core noise | Table 3: gaze direction | Pupil Core-specific | **Drop**; {70} and {72} cover the row |
| {66} | Nenna et al. 2023, VR human–robot workload | Table 3: window exclusion (35%) | Fits better as a VR workload study | **Keep for the 35% rule and also cite in Ch. 2 VR section** (check that it states the 35% rule; v4-6 used it for that) |
| {72} | Kothari et al. 2020, Gaze-in-wild (eye + head) | Table 3: gaze direction | Now the key source for "eye-in-head gaze is distorted by head movement" | **Also cite** in Ch. 3 Fixation check and Ch. 4 Discussion |
| {18} | Ktistakis et al. 2022, COLET | Ch. 1 dataset switch only | Fine, but now one sentence | Keep (needed for the disclosure) |
| {23} | Bentivoglio et al. 1997, speech raises blinks | Ch. 1 switch reason; Ch. 2 | Pina has no speech, so only the reason for switching | Keep; delete any limitation that says the high-load task involves speech |
| {43} | Andersson et al. 2017, event-detection algorithms | Ch. 3 I-VT settings | Validated on screen data with a still head | Keep, and add "for a still head" — that is the point of the failed check |
| {44} | Komogortsev et al. 2010 | Ch. 2 | Same | Keep; same note |
| {pina} | Pina et al. 2025, "A comprehensive analysis of the influence of cognitive load on physiological signals in virtual reality", IEEE conf. 2025 (cited in Bielsa ref [6]) + Zenodo 18669059 | Ch. 1–3 | **Missing from the list** | Add; get full venue, pages and DOI from the paper itself |
| {bielsa} | Bielsa, Pina, Serrano & Martin, "An analysis of gaze behavior under multisensory cognitive load in immersive environments", Comput. Graph., vol. 137, 104613, 2026, doi:10.1016/j.cag.2026.104613 | Ch. 1–5 | **Missing from the list** | Add (verified: full text read) |

## Keep as is (42)
{1} Kahneman, {2} Paas, {3} Young, {4} Hart & Staveland, {5} Tao, {6} Matthews, {7} Bishop, {8} Kaczorowska,
{9} Chen & Guestrin, {10} Grinsztajn, {11} Rolon-Merette, {12} Nerella (head-mounted AR, worth noting in Ch. 2 VR),
{13} Shao, {14} Saeb, {15} Kapoor, {16} Božak, {19} Lundberg & Lee, {20} Lundberg et al., {24} UN SDGs,
{26} Christodoulou, {28} Varoquaux 2018, {29} Kret, {31} Salvucci, {33} Cawley, {34} Hart 2006, {35} Mathôt (now
also the source for the brightness limitation), {36} Recarte 2008, {37} May, {40} Liu, {41} Wang, {42} Recarte &
Nunes, {50} Zou & Hastie, {51} Walocha, {53} Little, {54} Varma, {55} Varoquaux 2017, {62} Molnar, {67} Hershman,
{68} Steinhauer, {69} Petersch & Dierkes (head-mounted pupil, still fits), {70} Hooge, {78} Fawcett,
{79} Niculescu-Mizil, {82} Ojala, {83} Nogueira, {guyon} Guyon & Elisseeff.

## Claims in v5 with no source yet
| Claim | Fix |
|---|---|
| Table 1 row: Shao et al. validation `[TODO: check]` | From `rrl-master-list.md` X12: random 80/20 split. Fill in "Random 80/20" |
| 30 s windows, 8 per segment | Design choice: add to a short Design Choices line |
| Varjo XR-4 at 200 Hz | {bielsa} (states it) |
| Accuracy, precision, recall, F1 definitions | {78} Fawcett covers them |
| Fixation/saccade detection in VR needs head pose | {bielsa} (ray-casting from headset pose) and {72} Kothari |

## VR eye-tracking workload candidates (cited by Bielsa; NOT yet verified — read before adding)
| Candidate | Why it would help |
|---|---|
| Lee et al. 2024, "Measuring cognitive load in VR training via pupillometry", IEEE Trans. Learn. Technol. 17:704–10, doi:10.1109/TLT.2023.3326473 | Pupil as a VR load measure (supports the pupil result and brightness caveat) |
| Nasri et al. 2024, "Exploring eye tracking to detect cognitive load in complex VR training", ISMAR-Adjunct | Eye-tracking load detection in VR (closest to this study) |
| Setu et al. 2025, "Predicting and explaining cognitive load ... in virtual multitasking", IEEE TVCG 31(5):3014–24 | VR multitasking load prediction with explanations |
| Hebbar et al. 2023, "Cognitive load estimation in VR flight simulator", J. Eye Mov. Res. 15(3) | VR eye-based load estimation |
Pick one or two after reading them (memory rule: verified peer-reviewed sources only).

## Other RRL files
- `rrl-colet-list.md` and section 3 of `rrl-by-topic.md` ("COLET and studies that used it") are COLET-only: archive.
- `rrl-master-list.md`, `rrl-decision-log.md`, `rrl-recheck-post-defense.md`: keep as history; this file supersedes them for v5.
- `build_v4_8_docx.py` holds `NEW_REFS` (only `guyon`): a v5 build needs `pina` and `bielsa` added there.
