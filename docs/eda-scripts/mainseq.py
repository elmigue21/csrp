"""The main sequence: the standard validity test for saccade detection.

For genuine saccades, peak velocity rises with amplitude in a tight, highly
predictable way (log-log near-linear, saturating above ~20 deg). It is the
canonical check in the eye-tracking methodology literature: if a detector's
output does not reproduce the main sequence, the events it labels 'saccades'
are not saccades. This does not depend on my judgement of what is plausible.
"""
import numpy as np
import pandas as pd
import warnings
from scipy import stats

warnings.filterwarnings("ignore")
d = pd.read_parquet("eye.parquet")

s = d.dropna(subset=["saccade_amplitude_degree", "saccade_velocity_degree/s"])
A = s.saccade_amplitude_degree.values
V = s["saccade_velocity_degree/s"].values

def report(mask, name):
    a, v = A[mask], V[mask]
    if len(a) < 50:
        print(f"  {name:<34} n={len(a)} -- too few"); return
    r_log = stats.pearsonr(np.log10(a), np.log10(v))[0]
    rho = stats.spearmanr(a, v)[0]
    print(f"  {name:<34} n={len(a):>6,}  log-log r={r_log:+.3f}  R2={r_log**2:.3f}  rho={rho:+.3f}")

print("=== MAIN SEQUENCE: amplitude vs peak velocity ===")
print("  (healthy saccade detection: log-log R2 typically > 0.8, rho strongly positive)\n")
report(np.ones(len(A), bool),               "ALL detected events")
report(A <= 90,                             "plausible subset (amp <= 90 deg)")
report(A > 90,                              "artifact subset (amp > 90 deg)")
report((A <= 20) & (V <= 900),              "small saccades (amp<=20, vel<=900)")

print("\n=== is the relationship even monotonic where it should be? ===")
sub = (A <= 20) & (V <= 900)
a, v = A[sub], V[sub]
bins = pd.cut(a, [0, 2, 4, 6, 8, 10, 15, 20])
tab = pd.DataFrame({"amp_bin": bins, "vel": v}).groupby("amp_bin").vel.agg(["count", "median"])
print(tab.round(1).to_string())
print("\n  expected: median peak velocity should climb steadily with amplitude")

print("\n=== independent red flag: one amplitude per 250 ms window ===")
multi = d[(d.saccade_count > 1)].dropna(subset=["saccade_amplitude_degree"])
print(f"  windows containing >1 saccade but carrying a single amplitude value: {len(multi):,}")
print(f"  ({100*len(multi)/len(s):.1f}% of usable saccade windows)")
print("  -> for these, the amplitude/velocity columns are ambiguous by construction")
rep = d.sort_values(["pid", "task", "timestamps_start_ms"]).groupby(["pid", "task"]) \
       .saccade_amplitude_degree.apply(lambda x: (x.diff() == 0).mean()).mean()
print(f"  mean fraction of consecutive windows repeating the SAME amplitude: {100*rep:.1f}%")
print("  -> consistent with a value being carried across windows rather than re-measured")
