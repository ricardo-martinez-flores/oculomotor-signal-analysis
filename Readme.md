# oculomotor-signal-analysis

Signal processing pipelines and statistical analysis scripts for
pupillometry and oculomotor research across three independent projects.

Shared here as methodological references. Scripts document the reasoning
behind each preprocessing and analytical decision. They are not
ready-to-run tools and should always be adapted to the specific
paradigm, population, and research question of the user.

---

## Author

**Ricardo Martinez-Flores**
PhD(c), Institute of Neurosciences, University of Barcelona (UBNeuro)
Vision and Control of Action Group, Department of Cognition, Development
and Educational Psychology, University of Barcelona, Barcelona, Spain
ANID Doctoral Fellow (Abroad) and National Master's Fellow

Contact: ricardo.antonio.martinezf@gmail.com

---

## Projects

### Eye-AD
Pupillary response and cognitive vergence during a visual oddball paradigm
(target / distractor words) in MCI and cognitively healthy older adults.
Participants classified by CSF and blood-based biomarkers. 
Clinical partners: Hospital del Mar, Hospital Clinic and Sant Roc Residence
de Barcelona.

Two research lines: (1) oculomotor temporal dynamics as non-invasive
diagnostic proxies for Alzheimer's neuropathological burden, and (2)
oculomotor training as a non-pharmacological intervention to improve
neurodegenerative protein levels and cognition in MCI.

Related manuscripts:
- Martinez-Flores, R., et al. (2026). Cognitive vergence and pupil response
  during oddball task are associated with Alzheimer's disease CSF
  neurodegenerative biomarkers. bioRxiv. Under review.
  https://doi.org/10.64898/2026.04.10.717637
- Martinez-Flores, R., et al. (2026). Cognitive vergence and pupillary
  responses as functional oculomotor signatures to differentiate AT(N)
  biological profiles. bioRxiv. Under review.
  https://doi.org/10.64898/2026.04.14.718456
- Martinez-Flores, R., et al. Attentionally-directed Oculomotor Training Produces 
  Disease-stage-dependent Effects on Plasma Tau, Cognition, and Oculomotor Signatures 
  in Patients with Mild Cognitive Impairment. Under review.

### SAPIENS
Pupillary dynamics across 9 cognitive load conditions (basal, scroll,
reality, music, reading, podcast, Tetris, documentary, N-back) in
healthy young adults. Tracker: Pupil Labs (60 Hz).
Pontificia Universidad Catolica de Valparaiso, Chile.

Manuscripts in preparation.

### Cogni-Action
Pupillary temporal dynamics during reading comprehension in adolescents
across three physical activity conditions (C-HIIT, MICT, SC).
Texts counterbalanced for syntactic difficulty. Tracker: Tobii (33 Hz).
Pontificia Universidad Catolica de Valparaiso, Chile.
Project led by Carlos Cristi-Montero.

Related manuscripts:
- Martinez-Flores, R., et al. (2025). Impact of physical activity modalities
  on text processing and comprehension in adolescents. The Cogni-Action
  Project. Journal of Experimental Education.
  https://doi.org/10.1080/00220973.2025.2513244
- Martinez-Flores, R., et al. (2026). Gray matter volume modulates the
  effect of acute physical activity on reading comprehension and cognitive
  load in adolescents. The Cogni-Action Project. bioRxiv. Under review.
  https://doi.org/10.64898/2026.03.31.715252
- Martinez-Flores, R., et al. Functional pupillary dynamics during reading
  comprehension across physical activity conditions. In preparation.

---

## Repository structure

    signal-processing/
        pupillometry/
            pupil_sapiens.py        -- cognitive load, 9 conditions, Pupil Labs
            pupil_cogniaccion.py    -- reading dynamics, Tobii, adolescents
            pupil_eyeAD.py          -- visual oddball, Tobii, MCI / older adults

    statistical-analysis/
        permutation_analysis.R      -- general pre-post / between-group framework
        lmm_eyeAD.R                 -- LMM: maximal random effects + permutation
        gamm_cogniaccion.R          -- GAMM + cluster permutation + BCa

---

## Methods

**Signal processing (Python)**
Per-eye preprocessing before binocular averaging. Pipeline order:
validity/confidence flagging, physiological range filter [2-8 mm],
derivative filter, gap expansion, linear interpolation, Gaussian
smoothing, baseline correction (median). Paradigm-specific parameters
are set to None and must be defined by the user based on their own data.

**Statistical analysis (R)**
- LMM: keep-it-maximal structure (Barr et al. 2013), principled
  step-down on singular fit; REML for estimation, ML for permutation
  t-statistic
- GAMM: smooth-by-condition (s:by) as primary model; tensor product
  interaction (te/ti) as alternative with AIC comparison
- Permutation: Freedman-Lane for covariate control, within-participant
  exchangeability for hierarchical data
- FWER: Westfall-Young Max-T with joint permutation
- Effect size: Hedges g with BCa bootstrap CI
- Cluster-based permutation on GAMM pointwise t-statistic curve

---

## Important notice

The code in this repository should never be copied and applied directly
to a new dataset without careful adaptation. Every preprocessing parameter
and statistical decision was made for a specific paradigm, population,
tracker, and research question. Applying these choices to a different
study without understanding and justifying each one will produce
methodologically incorrect results.

For guidance adapting these pipelines or analyses to your own study,
contact: ricardo.antonio.martinezf@gmail.com

---

## Citation

If you use or adapt code from this repository, please cite the associated
publications listed above and include a link to this repository.
A citable DOI will be assigned upon manuscript acceptance via Zenodo.

---

## License

This repository is licensed under Creative Commons Attribution 4.0
International (CC BY 4.0). You are free to use and adapt this code
for any purpose, provided you cite the original author and include
a link to this repository.

Full license: https://creativecommons.org/licenses/by/4.0/

---

## Requirements

Python: pandas numpy scipy matplotlib

R: lme4 lmerTest dplyr tidyr ggplot2 patchwork boot emmeans mgcv readxl
