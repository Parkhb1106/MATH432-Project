# Entropy Meets Topology: Transferring Atrial Fibrillation Detection from Clinical ECG to Wrist-Worn PPG via Nonlinear Dynamical Features

*Draft of Section 1 (Introduction) and Section 2 (Data)*

---

## 1. Introduction

### 1.1 Background

Atrial fibrillation (AF) is the most common sustained cardiac arrhythmia. During AF, the atria do not contract in an organised way. Electrical impulses reach the ventricles at irregular times, so **the interval between consecutive heartbeats becomes irregular**. AF is associated with an approximately five-fold increase in the risk of stroke [1]. It is also often paroxysmal (it comes and goes) and asymptomatic, so a short ECG recorded in a clinic can easily miss it. For these reasons, long-term and low-cost rhythm monitoring is clinically valuable.

Consumer wearables now make such monitoring possible. A smartwatch carries a **photoplethysmography (PPG)** sensor: an LED measures changes in blood volume under the skin of the wrist. From this signal the watch can estimate the time between pulses passively and continuously. Large studies such as the Apple Heart Study [2] have shown that PPG-based irregular-rhythm notifications can be used for AF screening at population scale. However, two problems make AF detection from wrist PPG harder than from ECG:

1. **Scarce labels.** Annotating AF requires a simultaneous reference ECG reviewed by clinicians. There are therefore far fewer labelled wearable recordings than labelled clinical ECG recordings, such as the MIT-BIH databases on PhysioNet [3].
2. **Domain shift.** A PPG pulse is smoother than an ECG QRS complex, so beat timing from PPG is less precise. PPG is also very sensitive to motion artifacts. A model trained on clinical ECG may therefore fail on wrist PPG.

A natural bridge between the two domains is the **beat-to-beat interval series**. From ECG we obtain RR intervals (between R-peaks); from PPG we obtain pulse-to-pulse intervals (PPI). Both measure essentially the same quantity, the time from one heartbeat to the next. Since the defining symptom of AF is the irregularity of exactly this series, features computed from it should keep their meaning across domains.

The interval series is the output of a nonlinear dynamical system (the cardiac conduction system under autonomic control). Classical HRV statistics, such as the standard deviation or RMSSD, summarise only its spread. Three families of mathematical tools describe its **dynamics** instead:

- **Scaling laws.** Detrended fluctuation analysis (DFA) [4] measures the fractal correlation of the series. Healthy sinus rhythm shows 1/f-type scaling (exponent α ≈ 1), while an uncorrelated series gives α ≈ 0.5.
- **Entropy.** Sample entropy [5] and its AF-specific variant COSEn [6] quantify how unpredictable the next interval is. Permutation entropy [7], computed at several time scales [8], measures the diversity of up/down ordering patterns.
- **Topology.** Takens' theorem [9] says that a delay embedding of a scalar series reconstructs the geometry of the underlying attractor. Persistent homology [10] then describes the shape of the resulting point cloud (connected components, loops) at all scales at once. The stability theorem [11] guarantees that this description changes only a little when the data are perturbed a little, which is an attractive property for noisy wearable signals. Persistence images [12] and persistence landscapes [13] turn these descriptions into vectors that a classifier can use.

### 1.2 Problem statement

This project asks whether **AF detection learned from clinical ECG can be transferred to wrist-worn PPG** by using mathematically motivated features of the beat-interval series. We formulate four research questions:

| | Research question |
| --- | --- |
| **RQ1** | How well do scaling, entropy and topological features distinguish AF from non-AF on clinical ECG? Are the three families complementary to each other, and do they add information beyond standard HRV statistics? |
| **RQ2** | Does a model trained on clinical ECG still work on wrist PPG? How large is the domain gap? |
| **RQ3** | When only a few patients with labelled wrist data are available, does adding clinical ECG data improve performance? |
| **RQ4** | Which features remain reliable under real motion artifacts recorded during daily life? |

The task is binary classification of fixed-length segments of the interval series into **AF** and **non-AF**. All evaluations are patient-independent: no patient contributes data to both the training and the test set.

### 1.3 Why this problem matters

- **Clinical relevance.** Earlier detection of asymptomatic AF can lead to earlier treatment and prevention of stroke. Wrist PPG is the only signal that a wearable can record continuously without any action from the user.
- **Data efficiency.** If clinical ECG databases, which are public and well annotated, can compensate for scarce wearable labels, new wearable devices could be deployed with much less data collection.
- **Interpretability and robustness.** Commercial detection algorithms are proprietary, and recent deep-learning approaches are hard to interpret. Each feature in this project has a clear mathematical meaning, for example "the rhythm has lost its fractal correlation" or "the next interval is unpredictable". The stability theorem of persistent homology also gives a theoretical reason to expect robustness to noise, which we test directly in RQ4.
- **Connection to the course.** The project combines empirical scaling laws, entropy theory, topological data analysis and classification on one real-world problem.

---

## 2. Data

We use two public datasets. Both are reduced to the same representation: a series of beat intervals in seconds, with an AF / non-AF label for each interval.

| Dataset | Signal | Size | Role in this project |
| --- | --- | --- | --- |
| **MIT-BIH Atrial Fibrillation Database (AFDB)** [3, 14] | 2-lead Holter ECG, 250 Hz | 25 patients, about 10 h each | Stage 1: feature comparison on clinical ECG (RQ1); source domain for transfer (RQ2, RQ3) |
| **Long-term ECG and wrist PPG with annotated AF episodes** (Zenodo, v6) [15] | Wrist PPG 100 Hz, reference ECG 500 Hz, 3-axis accelerometers | 39 patients (35 with PPG), 5–8 days each, 261 days in total | Stages 2–3: target domain for transfer (RQ2, RQ3); motion robustness (RQ4) |

### 2.1 MIT-BIH Atrial Fibrillation Database (AFDB)

**Source.** PhysioNet, https://physionet.org/content/afdb/1.0.0/ (open access). We download it from PhysioNet's official open-data mirror and verify every file against the published SHA-256 checksums.

**Content.** The database contains 25 long-term ECG recordings from patients with AF, mostly paroxysmal. Each recording is about 10 hours long and has two ECG leads sampled at 250 Hz with 12-bit resolution. Two recordings (00735, 03665) contain annotations only, with no signal. This does not affect us, because we use only the beat times.

**Raw file structure** (WFDB format, one set of files per record):

| File | Content |
| --- | --- |
| `.hea` | Header: number of channels, sampling frequency, number of samples, gain |
| `.dat` | ECG signal (binary, two channels) |
| `.atr` | Rhythm annotations made by cardiologists: the sample index where each rhythm episode begins, with a label `(N` (normal), `(AFIB` (AF), `(AFL` (atrial flutter) or `(J` (AV junctional rhythm) |
| `.qrs` | Beat (QRS) locations from an automatic detector |
| `.qrsc` | Manually corrected beat locations (only records 05091 and 07859) |

*Sample — record 04746:*

```
04746.hea   04746 2 250 9205760 22:00:00          # 2 channels, 250 Hz, 9,205,760 samples (10.2 h)
            04746.dat 212 0 12 0 -57 10750 0 ECG1
            04746.dat 212 0 12 0 -68 -18834 0 ECG2

04746.dat   (first 5 samples, mV)   ECG1    ECG2
                                   -0.285  -0.340
                                   -0.300  -0.320
                                   -0.310  -0.325
                                   -0.300  -0.315
                                   -0.300  -0.310

04746.atr   sample      rhythm      # rhythm changes
            62          (N
            1,097,510   (AFIB
            1,098,684   (N
            2,284,261   (AFIB

04746.qrs   sample: 124, 357, 583, 809, 1033, 1263, ...   # beat locations
```

**From annotations to labelled intervals.** For each record:

1. Take the beat locations from `.qrsc` if available, otherwise from `.qrs`.
2. Compute RR intervals, rr[i] = (beat[i+1] − beat[i]) / 250 seconds.
3. Give each interval the rhythm label in force at the beat that ends it.

Intervals outside 0.25–2.5 s are flagged as implausible. In total this gives 1,221,650 intervals (249 h). By time, 59.9% is normal rhythm, 37.5% AF, 2.1% junctional rhythm and 0.6% atrial flutter.

**Segment dataset.** The interval series is cut into non-overlapping segments of **60 consecutive intervals**. We use a fixed number of beats rather than a fixed duration, so that every segment gives the same number of points to the entropy and topology computations regardless of heart rate. The labelling rules are:

- A segment is dropped if it contains any implausible interval, any atrial flutter, or any interval without a rhythm label.
- AF is the AFIB label only. Normal and junctional rhythm count as non-AF.
- Main label: `label_50 = 1` if at least 50% of the segment's intervals are AF.
- Secondary label for a sensitivity analysis: `label_excl` = 1 if at least 90% AF, 0 if at most 10% AF, otherwise excluded (−1).

Record 07162 is excluded. Its reference beat locations do not match the QRS complexes in its signal: an independent detector agreed with only about 30% of its beats, while the other records agreed at around 99%.

The resulting dataset has **18,925 segments from 24 patients**. Of these, 7,724 (40.8%) are AF under `label_50`; 387 segments (2.0%) are ambiguous under `label_excl`.

| Field | Shape | Meaning |
| --- | --- | --- |
| `rr` | (18925, 60) | The 60 RR intervals of the segment, in seconds |
| `af_frac` | (18925,) | Fraction of the segment's intervals annotated as AF |
| `label_50` | (18925,) | Main label: 1 = AF, 0 = non-AF |
| `label_excl` | (18925,) | Strict label: 1 / 0 / −1 (ambiguous) |
| `record` | (18925,) | Patient (record) ID, used for patient-level splits |
| `start_samp`, `end_samp` | (18925,) | Sample positions of the first and last beat in the original signal |
| `beat_source` | (18925,) | Which beat file was used (`qrs` or `qrsc`) |

*Sample — two segments from the same patient (first 8 of 60 intervals shown):*

| record | af_frac | label_50 | label_excl | start_samp | rr[0:8] (s) |
| --- | --- | --- | --- | --- | --- |
| 04746 | 0.0 | 0 | 0 | 124 | 0.932, 0.904, 0.904, 0.896, 0.920, 0.944, 0.936, 0.928 |
| 04746 | 1.0 | 1 | 1 | 2,609,859 | 0.596, 0.700, 0.480, 0.464, 0.616, 0.676, 0.556, 0.584 |

The non-AF segment is regular, with intervals around 0.92 s. The AF segment is shorter on average and visibly irregular.

**Patient-level folds.** The proportion of AF differs greatly between patients: two records are 100% AF, while another is 0.2% AF. We therefore split the 24 patients into 5 folds with `StratifiedGroupKFold`, grouped by patient and stratified by label, so that every fold has a similar AF share (36–46%). The fold assignment is fixed and shared across all experiments.

### 2.2 Long-term ECG and wrist PPG with annotated AF episodes (Zenodo)

**Source.** Zenodo, version 6 (published 3 November 2026), https://zenodo.org/records/23921320, DOI 10.5281/zenodo.23921320. It is open access under the CC BY-NC-SA 4.0 licence, so the data are used for non-commercial research and are not redistributed in our repository. The data were collected at Vilnius University and Kaunas University of Technology.

**Content.** 39 patients were monitored for 5–8 days each, 261 days in total, during their daily life. Each patient wore two devices at the same time:

- a **reference ECG** (Bittium Faros 180 chest device): single lead, 500 Hz, with a 25 Hz chest accelerometer;
- a **wrist device**: green-light PPG at 100 Hz, an ambient-light channel, and a 50 Hz wrist accelerometer.

AF episodes were detected automatically on the ECG, then reviewed and corrected by clinicians. The label is given **for every RR interval**. Four patients (024, 025, 034, 037) have no PPG data, so 35 patients can be used for the PPG experiments. The full dataset is 62.7 GB, so we download and process a selected subset of patients.

**File structure** (MATLAB v7.3 / HDF5, read with `h5py`):

`XXX_ECG.mat`, one per patient:

| Variable | Shape | Meaning |
| --- | --- | --- |
| `ECG` | (N, 1) | ECG signal, 500 Hz, µV. Gaps between recording sessions are filled with NaN |
| `Accelerometer_X/Y/Z` | (N/20, 1) | Chest acceleration, 25 Hz, mg |
| `QRSindex` | (n_beats, 1) | R-peak sample positions (1-based) |
| `rr` | (n_beats − 1, 1) | RR intervals in seconds (= diff(QRSindex) / 500) |
| `AF_annotation` | (n_beats − 1, 1) | Label of each RR interval: 1 = AF, 0 = non-AF |
| `recording_startday`, `recording_starttime` | text | Start of the recording (day number, clock time) |

`XXX_PPG.mat`, one per patient: every variable is a list with one entry per continuous wrist recording ("PPG segment").

| Variable | Content of each entry |
| --- | --- |
| `PPG_GREEN` | Green-light PPG waveform, 100 Hz, ADC units |
| `PPG_AMBIENT` | Ambient-light channel, 100 Hz |
| `Accelerometer_X/Y/Z` | Wrist acceleration, 50 Hz, g |
| `recording_startday`, `recording_starttime` | Start of the PPG segment on the ECG clock |
| `sync_method` | How the PPG segment was aligned to the ECG: `beat` (beat-level alignment, about ±0.05 s) or `coarse` (motion-based alignment only, about ±1 s) |

A file `subject_info.xlsx` also gives demographics (sex, age, height, weight), lifestyle, comorbidities and medication for each patient.

*Sample — patient 006:*

| | Value |
| --- | --- |
| ECG | 183.3 h, 864,623 beats, 25.3% of the time in AF, 2 AF episodes |
| First beat | `QRSindex[0] = 131` (0.26 s after the start of the recording) |
| Implausible RR (outside 0.25–2.5 s) | 0.56% (e.g. long gaps where the electrodes lost contact) |

| PPG segment | Start (from ECG start) | Length | `sync_method` | ECG beats covered | AF share |
| --- | --- | --- | --- | --- | --- |
| 0 | +18.34 h | 2.39 h | beat | 9,974 | 0% |
| 1 | +22.08 h | 0.80 h | coarse | 3,274 | 0% |
| 2 | +45.37 h | 2.95 h | beat | 17,495 | 100% |

This sample shows an important property of the dataset: **PPG covers only part of the ECG monitoring time** (6.1 of 183.3 hours for patient 006). The amount of usable labelled PPG is therefore much smaller than the dataset size suggests.

**Transferring ECG labels to PPG.** Labels exist only on the ECG. Because the PPG start times are given on the ECG clock, each PPG segment is matched to the ECG beats that fall within its time span, and those beats' `AF_annotation` values become the labels of the PPG segment. Pulse peaks detected in `PPG_GREEN` give the PPI series. This series is then cut into the same 60-interval segments and labelled with the same rule as for AFDB (`label_50`). We prefer PPG segments with `sync_method = beat`, since `coarse` alignment (±1 s) is too loose for beat-level labels. The wrist accelerometer is aligned in the same way and is used in RQ4 to group segments into rest, low-activity and high-activity conditions.

### 2.3 Summary of the common representation

| | AFDB (source) | Zenodo ECG (reference) | Zenodo PPG (target) |
| --- | --- | --- | --- |
| Beat times | Annotated QRS | `QRSindex` | Detected pulse peaks |
| Interval series | RR | RR | PPI |
| Label | Cardiologist rhythm annotation | Per-RR `AF_annotation` | Transferred from simultaneous ECG |
| Unit of analysis | 60-interval segment, AF / non-AF | same | same |
| Split | 5 patient-level folds | patient-level | patient-level |

---

## References

[1] Wolf, P. A., Abbott, R. D., & Kannel, W. B. (1991). Atrial fibrillation as an independent risk factor for stroke: the Framingham Study. *Stroke*, 22(8), 983–988.
[2] Perez, M. V., et al. (2019). Large-scale assessment of a smartwatch to identify atrial fibrillation. *New England Journal of Medicine*, 381(20), 1909–1917.
[3] Goldberger, A. L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*, 101(23), e215–e220.
[4] Peng, C.-K., Havlin, S., Stanley, H. E., & Goldberger, A. L. (1995). Quantification of scaling exponents and crossover phenomena in nonstationary heartbeat time series. *Chaos*, 5(1), 82–87.
[5] Richman, J. S., & Moorman, J. R. (2000). Physiological time-series analysis using approximate entropy and sample entropy. *American Journal of Physiology – Heart and Circulatory Physiology*, 278(6), H2039–H2049.
[6] Lake, D. E., & Moorman, J. R. (2011). Accurate estimation of entropy in very short physiological time series: the problem of atrial fibrillation detection in implanted ventricular devices. *American Journal of Physiology – Heart and Circulatory Physiology*, 300(1), H319–H325.
[7] Bandt, C., & Pompe, B. (2002). Permutation entropy: a natural complexity measure for time series. *Physical Review Letters*, 88(17), 174102.
[8] Costa, M., Goldberger, A. L., & Peng, C.-K. (2002). Multiscale entropy analysis of complex physiologic time series. *Physical Review Letters*, 89(6), 068102.
[9] Takens, F. (1981). Detecting strange attractors in turbulence. In *Dynamical Systems and Turbulence*, Lecture Notes in Mathematics, 898, 366–381.
[10] Edelsbrunner, H., & Harer, J. (2010). *Computational Topology: An Introduction*. American Mathematical Society.
[11] Cohen-Steiner, D., Edelsbrunner, H., & Harer, J. (2007). Stability of persistence diagrams. *Discrete & Computational Geometry*, 37(1), 103–120.
[12] Adams, H., et al. (2017). Persistence images: a stable vector representation of persistent homology. *Journal of Machine Learning Research*, 18(8), 1–35.
[13] Bubenik, P. (2015). Statistical topological data analysis using persistence landscapes. *Journal of Machine Learning Research*, 16, 77–102.
[14] Moody, G. B., & Mark, R. G. (1983). A new method for detecting atrial fibrillation using R-R intervals. *Computers in Cardiology*, 10, 227–230.
[15] Bacevičius, J., et al. (2026). Long-term electrocardiogram and wrist-based photoplethysmogram recordings with annotated atrial fibrillation episodes (Version v6) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.23921320
