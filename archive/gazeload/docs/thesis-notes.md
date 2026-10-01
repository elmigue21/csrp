# Thesis Notes

Working notes for the defense of "Comparative Analysis of Logistic Regression
and XGBoost for Eye-Tracking-Based Cognitive Load Detection with Feature
Importance Analysis." Each section records a decision, the reason for it, and
the current state of its support in the review of related literature.

## 1. The 30-Second Aggregation

### 1.1 What it is

The released GAZELOAD metrics files provide one row per 250 ms window. The
models in this study do not consume those rows directly. Consecutive windows are
grouped into fixed 30-second epochs, and each group is reduced to a single row by
summarizing every column across the group.

Because 30 seconds divided by 250 ms is 120, one epoch contains up to 120
windows. Each of the sixteen base features contributes its mean and its standard
deviation across those windows, and five further features are derived as
proportions. The result is 135,499 windows reduced to 1,087 epochs, and each
epoch becomes one training example.

### 1.2 Why aggregation is necessary

The measures named in the review of related literature are rates: blink rate,
fixation rate, saccade rate, dwell time. A rate is a count over a period of time
and is undefined inside a single 250 ms sample. Aggregation is therefore not a
convenience step. It is the step that converts the released per-window values
into the constructs the literature actually measures.

The dataset confirms this empirically. At the published 250 ms resolution, gaze
transition entropy is exactly zero in 98.78 percent of windows and the blink flag
fires in only 0.43 percent. Neither column carries usable variance until windows
are combined.

### 1.3 How the epochs are cut

Two implementation rules matter for defensibility, both in `src/data.py`.

First, epoch boundaries are computed from elapsed time within each recording
rather than from row position. The epoch index is the elapsed milliseconds since
the start of that recording divided by the epoch length. Counting by row would
allow a tracker dropout to join gaze from before and after the gap into what
would then be falsely reported as thirty continuous seconds. Counting by the
clock cannot do this, and it also guarantees that an epoch never spans a
recording boundary.

Second, an epoch is retained only if at least 80 percent of its 120 nominal
windows are present. This discards the partial trailing fragment at the end of
each recording and any epoch heavily affected by dropout.

### 1.4 Why thirty seconds

The choice is a trade-off. A shorter epoch contains too few oculomotor events for
a rate or an entropy to be meaningful, which is the original problem at 250 ms. A
longer epoch yields fewer training examples and blurs over genuine variation in
effort within a recording. Thirty seconds was selected as the balance between
these two pressures.

This specific value is not taken from any source in the review of related
literature. It is a judgment made from the exploratory data analysis in this
project, and Chapter 3 should state it as such rather than imply a citation
exists.

### 1.5 State of RRL support

Support for the aggregation is partial and should be described precisely.

Supported: the principle that eye-tracking workload features are computed over
extended segments rather than over milliseconds. Kaczorowska, Plechawska-Wójcik
and Tokovarov compute their feature set over whole task segments of 90 and 180
seconds. Their features are also structured as a mean and a standard deviation
per base measure, which is the same structure used here.

Supported: the underlying measures. Tao and colleagues reviewed 91 studies and
identified thirteen eye movement measures of mental workload, including blink
rate, fixation rate, saccade rate, saccadic amplitude and fixation duration. Most
of the base features aggregated here appear on that list.

Not supported: the value of thirty seconds itself. No source in the current
reference list justifies this particular window length. Ktistakis et al. (COLET)
and Aksu et al. are the two remaining candidates worth checking for a stated
window length; neither has been verified yet.

### 1.6 Note on a structural difference from Kaczorowska

Kaczorowska et al. compute the mean and standard deviation of event durations,
such as mean fixation duration and mean saccade amplitude per event. GAZELOAD
does not publish per-event records; it publishes per-window counts. The feature
`fixation_count_mean` in this study is therefore the average number of fixations
per 250 ms window, not an average fixation duration.

The aggregation structure is borrowed from that work; the underlying quantities
differ. Chapter 3 should state this rather than allow the reader to assume the
features are equivalent.

### 1.7 Consequence for the derived features

Both `mean` and `std` skip missing values. As a result, `saccade_amplitude_
degree_mean` averages only the windows in which a saccade actually occurred, and
carries no information about how often saccades failed to occur. Since saccade
amplitude and velocity are missing in exactly the 23.9 percent of windows where
`saccade_count` is zero, that missingness is a measurement rather than a gap.
This is the reason `sacc_missing_frac` and `fixation_zero_frac` exist as separate
features, and the justification is a property of the aggregation step rather than
a citation.

References:
https://pmc.ncbi.nlm.nih.gov/articles/PMC7914927/
https://pmc.ncbi.nlm.nih.gov/articles/PMC6696017/

## 2. The Dataset and What Was Done With It

### 2.1 What GAZELOAD is

GAZELOAD is a multimodal eye-tracking dataset recorded during industrial
human-robot collaboration, published by Karbouj, Gaaloul and Krüger in 2025.
Participants performed assembly work alongside a robot while wearing Meta Aria
smart glasses, which recorded their gaze. Ambient illuminance was logged
alongside the gaze stream.

Twenty-six participants each completed five tasks, giving a complete 26 by 5
grid of 130 recordings with no missing cells. The tasks increase in demand by
experimental design. After each task the participant rated how mentally demanding
it felt on a scale from 1 to 10, and that self-report is the target variable of
this study.

The `Tasks_Rating.xlsx` metadata file records 16 male and 10 female participants,
which does not agree with the dataset title. This should be resolved before the
sample composition is cited.

### 2.2 Shape of the released data

This study uses the `04_eye-metrics` directory, which contains one CSV per
recording, named by participant and task, for example `01_2_Metrics_withLux.csv`.
Each row is one non-overlapping 250 ms window.

    130 CSV files
    135,499 rows total, roughly 1,000 per recording
    22 columns

The 22 columns divide into three groups. Only the third group contains actual
measurements.

| Group | Count | Columns |
|---|---|---|
| Target | 1 | `selfreport_mental_load(1-10)` |
| Keys and timing | 4 | `tasks`, `Participant_ID`, `timestamps_start_ms`, `timestamps_end_ms` |
| Measurements | 17 | `fixation_count`, `saccade_count`, `saccade_amplitude_degree`, `saccade_velocity_degree/s`, `EyeGaze_x/y/z`, `std_EyeGaze_x/y/z`, `gaze_x_scene_mean`, `gaze_y_scene_mean`, `GTE`, `FDI`, `SaccRate`, `blink_flag_any`, `lux_interpolated` |

A representative extract, showing the first four windows of recording `01_1` and
a subset of the columns:

    fixation_count  saccade_count  saccade_amplitude  FDI     GTE  blink_flag_any  lux
                 2              1              5.065  21.659  0.0               0  234.789
                 1              1             72.375   0.000  0.0               0  248.626
                 0              1             72.375     NaN  0.0               0  262.464
                 2              4             19.298  21.742  0.0               0  276.301

Two properties of the released format are visible in this extract and both shape
the method. The third row has `FDI` missing because `fixation_count` is zero;
this missingness is a measurement rather than a recording fault. And `GTE` is
zero in all four rows, which is typical rather than exceptional at this
resolution.

The single most consequential property of the format is that the target is
constant within a recording. Every row of a file carries the same rating.
The 135,499 rows therefore represent only 130 independent label observations,
and this fact drives the evaluation protocol described in §1 and in
`methodology.md`.

### 2.3 Columns dropped

Four of the 22 columns are excluded from the feature matrix. No measurement
column is dropped; all 17 are used.

| Column | Reason |
|---|---|
| `tasks` | The task index alone predicts the label at ROC-AUC 0.93. Retaining it would let both classifiers read the experimental condition instead of the participant's eyes. |
| `Participant_ID` | A grouping key rather than a feature, and corrupted in six recordings, where it holds the strings `Hedi` and `Wassim` instead of numeric identifiers. Participant identity is taken from the filename, which is internally consistent with the `tasks` column across all 130 recordings. |
| `timestamps_start_ms` | Used to construct epoch boundaries, then discarded. Absolute recording time is not a workload feature. |
| `timestamps_end_ms` | As above. |

Two further columns deserve mention because they are retained but contested.
`lux_interpolated` is an environmental rather than an oculomotor variable, and
mean session illuminance differs by task, so it functions as an indirect route
to the task identity that the `tasks` exclusion was designed to block. It is
retained in the main configurations and removed in the ablation configurations
so that its contribution can be measured rather than assumed.

Pupil diameter and pupil dilation are absent from the release entirely. The
authors' `01_Metadata/pupil.py` derives them from raw Aria `.vrs` recordings,
which are not part of the public distribution. Because pupillometry is the
strongest single eye-based indicator of workload in the reviewed literature,
reported as significant in 79 percent of the studies surveyed by Tao et al.,
its absence is a limitation of this study rather than a design choice.

### 2.4 Data transformation

The transformation from released format to model input proceeds in five steps.

1. Read. The 130 CSVs are read and concatenated, with participant and task
   parsed from the filename. Rows are sorted by start timestamp within each
   recording.
2. Epoch. Windows are grouped into 30-second epochs cut on elapsed time, and
   epochs below 80 percent coverage are discarded. This is the step described in
   full in §1.
3. Aggregate. Each of the sixteen base features contributes its mean and its
   standard deviation across the windows of the epoch, and five proportion
   features are derived. See §2.5.
4. Label. The 1 to 10 rating is converted to a binary target under two schemes:
   an absolute threshold at 4 or above, which retains all 1,087 epochs; and a
   within-participant threshold at each participant's own median, which excludes
   the 319 epochs sitting exactly at the median and retains 768.
5. Scale. Features are either left raw or standardized within each participant.
   Configurations that ablate illuminance drop the two lux columns at this point.

The resulting epoch table has this shape:

    1,087 rows
    43 columns: 37 features, plus participant, task, epoch, rating,
                n_windows and coverage as bookkeeping

Reduction summary:

| Stage | Rows | Feature columns |
|---|---|---|
| Released windows | 135,499 | 17 measurements |
| After epoching and aggregation | 1,087 | 37 |
| After within-participant labelling | 768 | 37 |

### 2.5 Key features added

No feature in the model is used exactly as published, because the model's unit of
observation is a 30-second epoch while the released unit is a 250 ms window. Every
column must therefore be summarized to cross that boundary. Thirty-two of the 37
features are the mean and standard deviation of the sixteen base measurements,
and this structure follows Kaczorowska et al., who report the mean and standard
deviation of each base measure in the same way.

Five features are constructed rather than aggregated. Each is the proportion of
windows in the epoch satisfying a condition, and each exists because averaging
the source column destroys the information in question.

| Feature | Source column | Justification |
|---|---|---|
| `blink_rate` | `blink_flag_any` | The source is a 0/1 flag firing in 0.43 percent of windows. Its mean over an epoch is the blink rate, which Tao et al. report as showing significant workload differences in 71 percent of the studies reviewed. The standard deviation of a binary column is a deterministic function of its mean and would be redundant, so only the mean is taken. |
| `sacc_missing_frac` | `saccade_amplitude_degree` | The column is missing in exactly the 23.9 percent of windows where `saccade_count` is zero. Aggregation skips missing values, so the mean cannot express how often a saccade failed to occur. Supported by the EDA, not by the RRL. |
| `fixation_zero_frac` | `fixation_count` | Proportion of windows with no fixation. Supported by the EDA, not by the RRL. |
| `gte_nonzero_frac` | `GTE` | Proportion of windows with non-zero gaze transition entropy, since GTE is exactly zero in 98.78 percent of windows. Not supported: the 91-study review by Tao et al. does not list gaze entropy among the thirteen eye movement measures identified. At epoch scale this feature also correlates 0.997 with `GTE_mean`, so it adds little. |
| `fdi_missing_frac` | `FDI` | Proportion of windows with no fixation. Identical to `fixation_zero_frac` in all 1,087 epochs, because `FDI` is missing exactly when `fixation_count` is zero. Scheduled for removal. |

Two duplicate pairs remain in the feature set and should be resolved before the
importance tables in Chapter 4 are quoted. `SaccRate_mean` correlates 1.000000
with `saccade_count_mean`, because `SaccRate` equals four times `saccade_count`
in every row, and `fdi_missing_frac` is identical to `fixation_zero_frac` as
noted above. Perfect collinearity makes logistic regression coefficients
arbitrary within the pair and splits XGBoost gain and SHAP attribution across two
interchangeable columns, halving the apparent importance of the underlying
construct.

A correction to prior wording: `methodology.md` §3.4 and the comment in
`src/data.py` state that `FDI` is missing when `fixation_count` is below two. It
is missing when `fixation_count` is exactly zero. The code is correct because it
tests `FDI.isna()` directly; only the stated mechanism is wrong.

References:
https://pmc.ncbi.nlm.nih.gov/articles/PMC7914927/
https://pmc.ncbi.nlm.nih.gov/articles/PMC6696017/
https://data.mendeley.com/datasets/9smd7nbtwc/1
