# signal-processing/pupillometry

---

IMPORTANT NOTICE

These pipelines should never be copied and applied as-is to a new dataset.
Every preprocessing decision documented here -- the confidence threshold,
the gap expansion window, the Gaussian sigma, the baseline duration, the
outlier criteria -- was made for a specific paradigm, population, tracker,
and research question. Applying these parameters to a different study without
understanding and justifying each choice is methodologically incorrect and
will likely produce invalid results.

Before using any of these scripts, read the full header of the relevant file,
understand why each parameter was chosen for that paradigm, inspect your own
raw data to determine appropriate values for your context, and modify the
pipeline accordingly. The code is shared as a methodological reference, not
as a ready-to-run toolbox.

If you need guidance adapting these pipelines to your own paradigm or data,
do not hesitate to reach out:

    ricardo.antonio.martinezf@gmail.com

---

## Scripts

### pupil_sapiens.py

Cognitive load assessment in healthy young adults.
9 conditions of varying demand (basal to N-back).
Pupil Labs hardware (Pupil Invisible / Neon), 60 Hz.
Confidence-based filtering, Gaussian smoothing, AUC extraction.
Pontificia Universidad Catolica de Valparaiso, Chile.

Input: one CSV per participant-condition, named {pid}_{code}.csv
Output: metrics_long.csv, interpolation_report.csv,
        group_curves.png, auc_bars.png

Paradigm-specific decisions:
- Confidence gate at 0.6: Pupil Labs model_confidence below this
  value reliably indicates tracking loss (blinks, occlusions).
- Edge padding (PAD_MS): transitions into and out of blinks corrupt
  signal on both sides. Value set by user based on data inspection.
- Gaussian sigma set by user: choose based on the temporal scale of
  the effect you expect. A sigma of N samples at 60 Hz corresponds
  to a low-pass cutoff of ~1/(2*pi*N/60) Hz.
- AUC on positive portion of baseline-corrected signal: negative
  values are pupil constriction below baseline and are not part of
  the cognitive load response being measured.
- N-back trial selection: one trial per participant, selected based
  on signal quality (lowest % interpolation, cleanest baseline).
- Outlier detection (Z_OUTLIER): set by user. Participants whose
  mean AUC exceeds this z-score are excluded from group figures.

### pupil_cogniaccion.py

Reading comprehension dynamics in adolescents.
3 counterbalanced texts of matched syntactic difficulty.
Tobii (300 Hz, or resampled). Exported as single CSV.
Focus is on temporal trajectory across reading epoch, not AUC.
Pontificia Universidad Catolica de Valparaiso, Chile.

Input: one CSV with all participants.
       Columns: Participant name, text_id, condition,
       Recording timestamp, Pupil diameter left/right,
       Validity left/right
Output: interpolation_report.csv, individual_curves.png,
        group_mean_curves.png

Paradigm-specific decisions:
- Gap expansion (EXPAND_MS): Tobii validity codes do not cover the
  full post-blink recovery artifact. Value set by user.
- Gaussian sigma (SIGMA_MS): set by user in ms, auto-converted to
  samples. Larger values smooth more but blur temporal features.
- Percentage-time normalization: reading duration varies across
  participants. Normalization allows cross-participant comparison
  of the temporal trajectory shape.
- IQR outlier filter (IQR_THRESH): set by user. See Leys et al.
  (2013) for guidance on choosing k.
- Hierarchical averaging: text -> participant -> group.
  Prevents participants with more valid samples from dominating.

### pupil_eyeAD.py

Visual oddball paradigm (target/distractor words) in MCI and controls.
Tobii (33 Hz). Event-based: 2000 ms epochs time-locked to stimulus onset.
Clinical population: CSF and blood-based biomarker-classified MCI
and healthy older adults.
University of Barcelona / Hospital del Mar / Hospital Clinic de Barcelona.

Input: one CSV with all trials from all participants.
       See script header for required column names.
Output: pupil_trials_clean.csv, pupil_oddball.png

Note: vergence computation is not included in this script.

Paradigm-specific decisions:
- Absolute time (ms) rather than percentage: trial duration is fixed
  at 2000 ms, so percentage normalization is not needed and would
  obscure millisecond-level temporal dynamics.
- Gaussian smoothing (GAUSSIAN_SIGMA): set by user in samples at
  33 Hz. At 33 Hz, 1 sample = ~30 ms. Choose to preserve the
  temporal resolution needed for onset and peak analysis.
- MAD-based amplitude outlier filter (QC['mad_mult']): adapts to
  session-level variance. A fixed absolute threshold would be too
  liberal for participants with small baseline pupils and too
  conservative for others. Value set by user.
- QC thresholds (max_interp_ratio, max_baseline_sd, min_valid_pts,
  velocity_thresh): all set by user. See QC dict in script.

---

## Common QC steps across all three pipelines

1. Validity flagging: mark samples as NaN before any computation.
2. Physiological range [2-8 mm]: values outside are artifacts.
3. Derivative filter: removes jumps inconsistent with genuine dynamics.
4. Per-eye processing before binocular averaging.
5. Interpolation over gaps after all NaN marking is done.
6. Gaussian smoothing as the final preprocessing step.
7. Baseline correction using median (more robust than mean to residual
   spikes that survive earlier QC).

---

## Related publications

The pipelines in this folder were developed in the context of the
following published and preprint works. If you use or adapt any
script, please cite the relevant publication.

**Cogni-Action Project (pupil_cogniaccion.py)**

Paper 1 (published):
  Martinez-Flores, R., Julio, C., Ibanez, R., Campos-Rojas, C., Jarpa, M.,
  Super, H., Tari, B., & Cristi-Montero, C. (2025).
  Impact of physical activity modalities on text processing and comprehension
  in adolescents. The Journal of Experimental Education, 1-16.
  https://doi.org/10.1080/00220973.2025.2513244

Paper 2 (preprint; under review):
  Martinez-Flores, R., Super, H., Sanchez-Martinez, J., Solis-Urra, P.,
  Ibanez, R., Herold, F., Paas, F., Mavilidi, M., Zou, L., &
  Cristi-Montero, C. (2026).
  Gray matter volume modulates the effect of acute physical activity on
  reading comprehension and cognitive load in adolescents.
  The Cogni-Action Project. bioRxiv.
  https://doi.org/10.64898/2026.03.31.715252

Paper 3 (in preparation):
  Martinez-Flores, R., et al. Functional pupillary dynamics during reading
  comprehension across physical activity conditions. Cogni-Action Project.

**Eye-AD Project (pupil_eyeAD.py)**

Paper 1 (preprint; under review):
  Martinez-Flores, R., Martin-Sobrino, I., Falgas, N., Grau-Rivera, O.,
  Suarez-Calvet, M., Cristi-Montero, C., Ibanez, A., & Super, H. (2026).
  Cognitive vergence and pupil response during oddball task are associated
  with Alzheimer's disease cerebrospinal fluid neurodegenerative biomarkers.
  bioRxiv. https://doi.org/10.64898/2026.04.10.717637

Paper 2 (preprint; under review):
  Martinez-Flores, R., Martin-Sobrino, I., Falgas, N., Grau-Rivera, O.,
  Suarez-Calvet, M., et al. (2026).
  Cognitive vergence and pupillary responses as functional oculomotor
  signatures to differentiate AT(N) biological profiles.
  bioRxiv. https://doi.org/10.64898/2026.04.14.718456

Paper 3 (under review):
  Martinez-Flores, R., et al. Attentionally-directed Oculomotor Training Produces 
  Disease-stage-dependent Effects on Plasma Tau, Cognition, and Oculomotor Signatures 
  in Patients with Mild Cognitive Impairment

**SAPIENS Project (pupil_sapiens.py)**

Manuscripts in preparation.

---

## Requirements

    pip install pandas numpy scipy matplotlib

---

## License

This repository is licensed under Creative Commons Attribution 4.0
International (CC BY 4.0). You are free to use and adapt this code
for any purpose, provided you cite the original author and include
a link to this repository.

Ricardo Martinez-Flores -- ricardo.antonio.martinezf@gmail.com
https://github.com/ricardo-martinez-flores/oculomotor-signal-analysis
