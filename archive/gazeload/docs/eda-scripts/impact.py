import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
d = pd.read_parquet("eye.parquet")

EPOCH_MS = 30000
start0 = d.groupby(["pid", "task"]).timestamps_start_ms.transform("min")
d["epoch"] = ((d.timestamps_start_ms - start0) // EPOCH_MS).astype(int)
k = ["pid", "task", "epoch"]
g = d.groupby(k)
ep = pd.DataFrame({"n": g.size()})
ep = ep[ep.n / (EPOCH_MS / 250) >= 0.8]
print("epochs at 30 s with >=80%% coverage: %d  (methodology.md reports 1,087)" % len(ep))

art = (d.saccade_amplitude_degree > 90) | (d["saccade_velocity_degree/s"] > 900)
d["amp_clean"] = d.saccade_amplitude_degree.where(~art)
d["vel_clean"] = d["saccade_velocity_degree/s"].where(~art)

a = g.saccade_amplitude_degree.mean().reindex(ep.index)
ac = g.amp_clean.mean().reindex(ep.index)
v = g["saccade_velocity_degree/s"].mean().reindex(ep.index)
vc = g.vel_clean.mean().reindex(ep.index)

print("\n=== effect of plausibility cleaning at the 30 s epoch scale ===")
print("saccade_amplitude_degree_mean   raw: median %7.1f deg   cleaned: median %5.2f deg" % (a.median(), ac.median()))
print("saccade_velocity_mean           raw: median %7.1f d/s   cleaned: median %5.1f d/s" % (v.median(), vc.median()))
print("\ncorr(raw epoch mean, cleaned epoch mean): amplitude r=%.3f   velocity r=%.3f" % (a.corr(ac), v.corr(vc)))
print("  -> the raw epoch feature is largely a DIFFERENT quantity, not a noisy version")
print("\nepochs where raw mean amplitude exceeds 90 deg (i.e. the")
print("  epoch average is itself anatomically impossible): %d of %d (%.1f%%)" % (
    (a > 90).sum(), len(ep), 100 * (a > 90).mean()))

print("\n=== redundancy inside the 37-feature matrix ===")
sm = g.saccade_amplitude_degree.apply(lambda s: s.isna().mean()).reindex(ep.index)
fm = g.FDI.apply(lambda s: s.isna().mean()).reindex(ep.index)
fz = g.fixation_count.apply(lambda s: (s == 0).mean()).reindex(ep.index)
print("fdi_missing_frac vs fixation_zero_frac : identical in %d/%d epochs (r=%.6f)" % (
    int(np.isclose(fm, fz).sum()), len(ep), fm.corr(fz)))
sc = g.saccade_count.mean().reindex(ep.index)
sr = g.SaccRate.mean().reindex(ep.index)
print("SaccRate_mean vs saccade_count_mean    : r=%.6f  (SaccRate = 4 x count exactly)" % sc.corr(sr))
scs = g.saccade_count.std().reindex(ep.index)
srs = g.SaccRate.std().reindex(ep.index)
print("SaccRate_std  vs saccade_count_std     : r=%.6f" % scs.corr(srs))

print("\n=== gaze validity at epoch scale ===")
gi = g.EyeGaze_z.apply(lambda s: (s < 0).mean()).reindex(ep.index)
print("epochs with >50%% invalid gaze (EyeGaze_z<0): %d of %d (%.1f%%)" % (
    (gi > .5).sum(), len(ep), 100 * (gi > .5).mean()))
p7 = gi[gi.index.get_level_values("pid") == 7]
print("participant 07: %d epochs, median invalid-gaze fraction %.2f" % (len(p7), p7.median()))
print("dropping participant 07 costs %d of %d epochs (%.1f%%)" % (len(p7), len(ep), 100 * len(p7) / len(ep)))
