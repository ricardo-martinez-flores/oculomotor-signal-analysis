# statistical-analysis

---

IMPORTANT NOTICE

These scripts should never be replicated exactly and applied to a new dataset
without careful adaptation. Every methodological choice documented here -- the
permutation scheme, the random effects structure, the cluster-forming threshold,
the bootstrap resampling strategy, the FWER correction method -- was made for
a specific design, sample size, outcome type, and research question. Blindly
reusing these scripts on a different study without understanding each decision
is not just suboptimal: it will produce statistically incorrect results.

Before using any of these scripts, read the full header of the relevant file,
understand the rationale behind each method, assess whether those assumptions
hold for your own data and design, and modify the code accordingly. In
particular: the random effects structure in the LMM must reflect your own
design; the cluster-forming threshold in the GAMM depends on your degrees of
freedom; the permutation scheme assumes exchangeability that must be justified
for your own data.

These scripts are shared as methodological references and worked examples,
not as general-purpose tools to be run without modification.

If you need guidance adapting these analyses to your own study design,
do not hesitate to reach out:

    ricardo.antonio.martinezf@gmail.com

---

## Scripts

### permutation_analysis.R

General-purpose permutation framework for pre-post and between-group designs.
Adaptable to any continuous outcome by editing the DATA CONFIGURATION block.

Methods:
- Sign-flip permutation (no covariate)
- Freedman-Lane permutation (with covariate)
- Westfall-Young Max-T (FWER for multiple outcomes)
- Hedges g + BCa bootstrap CI (noncentral-t fallback when n < 10)
- Optional spaghetti + violin figure and Word table

Designed for: biomarker comparisons (pre-post intervention),
between-group effect sizes, any design with one outcome at a time
or multiple outcomes requiring FWER control.

### lmm_eyeAD.R

Linear mixed model analysis for oculomotor features (Eye-AD project).
Features: AUC, peak amplitude, peak latency, mean dilation.

Model 1 -- Keep-it-maximal (Barr et al. 2013):
  Applied to AUC with stimulus (Target/Distractor) x timepoint (pre/post).
  Starts with the maximal random effects structure justified by the design.
  Step-down only if convergence fails or singular fit is detected.
  Rationale: anti-conservative Type I error results from under-specifying
  random effects when predictors vary within participants.

Model 2 -- Multiple features, permutation inference:
  All oculomotor features tested simultaneously against a single predictor.
  Freedman-Lane permutation for inference (within-participant residual
  permutation, respects hierarchical structure).
  REML -> ML switch: REML estimates are not comparable across models
  with different fixed effects; ML is used for the permutation t-statistic.
  The reported coefficients and CIs come from the final REML fit.
  Max-T (Westfall-Young, joint permutation) for FWER across features.
  BCa bootstrap with participant-level resampling for beta CIs.

Figures:
- Estimated marginal means from Model 1 (stimulus x timepoint, AUC)
- Forest plot of beta coefficients with BCa CIs (Model 2)
- Observed means +/- SE per feature (complement to EMM figure)

### gamm_cogniaccion.R

Generalized additive mixed model + cluster-based permutation.
Pupillary dynamics during reading comprehension (Cogni-Action project).
Three conditions (SC, MICT, C-HIIT), temporal analysis across reading %.

Two model types explained and implemented:

  A) Smooth-by-condition (primary model):
     Pupil ~ Condition + s(Time_pct) + s(Time_pct, by=Condition) + s(Participant, re)
     Each condition gets a smooth deviation from the reference (SC).
     Best when conditions are categorical and you want per-condition
     deviations relative to a reference level.

  B) Tensor product interaction (alternative, set RUN_TENSOR = TRUE):
     Pupil ~ Condition + s(Time_pct) + ti(Time_pct, Condition_num) + s(Participant, re)
     Models the time x condition interaction as a 2D smooth surface.
     Does not assume a reference level. Requires numeric condition coding.
     Better when conditions are ordered or when the interaction shape
     is complex and not decomposable into additive per-condition offsets.
     AIC comparison between models is printed when RUN_TENSOR = TRUE.

Inference: cluster-based permutation on the GAMM-derived pointwise
t-statistic curve. Clusters are contiguous windows where |t| > t_crit.
Null distribution built by permuting condition labels within participants
(preserves within-participant correlation under H0).

Effect size: Hedges g on AUC within each significant cluster window.
BCa bootstrap CI on the AUC-based g.

Figures:
- GAM-smoothed curves per condition with 95% CI ribbon
- Pointwise differences with cluster shading and Hedges g annotation
- Forest plot of AUC-based Hedges g per significant cluster

---

## When to use which script

  permutation_analysis.R:
    Single or multiple continuous outcomes, pre-post or between-group,
    no temporal structure. Biomarkers, cognitive scores, summary metrics.

  lmm_eyeAD.R:
    Event-locked features (AUC, amplitude, latency) from discrete trials.
    Crossed within-participant design (stimulus x timepoint).
    Need to account for random slopes and test multiple features.

  gamm_cogniaccion.R:
    Continuous temporal signal (pupil over time %).
    Need to capture nonlinear time courses and compare their shapes.
    Cluster-based inference on the full time series.

---

## Statistical decisions shared across scripts

Freedman-Lane permutation:
  When a covariate must be controlled, residualizing on it under H0
  and permuting the residuals correctly separates the covariate
  effect from the effect being tested. Standard label permutation
  would confound the two.

Westfall-Young Max-T:
  Joint permutation preserves the correlation between test statistics.
  More powerful than Bonferroni when outcomes are correlated.
  More conservative than BH-FDR but strongly controls FWER.

BCa bootstrap:
  Corrects for both bias and skewness in the bootstrap distribution.
  More accurate than percentile bootstrap for small n or skewed data.
  Participant-level resampling (not row-level) preserves the
  within-participant correlation structure.

Hedges g vs Cohen d:
  Hedges g applies the small-sample correction factor J = 1 - 3/(4df-1).
  For paired data: g = (mean_diff / SD_diff) * J  (d_z formulation).
  For independent samples: g = (mean_diff / SD_pooled) * J.

---

## Related publications

The scripts in this folder were developed and applied in the context of
the following published and preprint works. If you use or adapt any
script, please cite the relevant publication.

**permutation_analysis.R** (pre-post biomarker analysis, intervention design)

Paper 3 -- Eye-AD (under review, GeroScience):
  Martinez-Flores, R., et al. Oculomotor training as a non-pharmacological
  intervention to improve neurodegenerative biomarkers and cognition in MCI.
  Uses: Freedman-Lane + BCa Hedges g + between-group Max-T and LMM with permutation inference

**lmm_eyeAD.R** (multi-feature LMM with permutation inference)

Paper 1 -- Eye-AD (preprint):
  Martinez-Flores, R., Martin-Sobrino, I., Falgas, N., Grau-Rivera, O.,
  Suarez-Calvet, M., Cristi-Montero, C., Ibanez, A., & Super, H. (2026).
  Cognitive vergence and pupil response during oddball task are associated
  with Alzheimer's disease cerebrospinal fluid neurodegenerative biomarkers.
  bioRxiv. https://doi.org/10.64898/2026.04.10.717637

Paper 2 -- Eye-AD (preprint):
  Martinez-Flores, R., Martin-Sobrino, I., Falgas, N., Grau-Rivera, O.,
  Suarez-Calvet, M., et al. (2026).
  Cognitive vergence and pupillary responses as functional oculomotor
  signatures to differentiate AT(N) biological profiles.
  bioRxiv. https://doi.org/10.64898/2026.04.14.718456

**gamm_cogniaccion.R** (GAMM + cluster permutation, reading dynamics)

Paper 2 -- Cogni-Action (preprint):
  Martinez-Flores, R., Super, H., Sanchez-Martinez, J., Solis-Urra, P.,
  Ibanez, R., Herold, F., Paas, F., Mavilidi, M., Zou, L., &
  Cristi-Montero, C. (2026).
  Gray matter volume modulates the effect of acute physical activity on
  reading comprehension and cognitive load in adolescents.
  The Cogni-Action Project. bioRxiv.
  https://doi.org/10.64898/2026.03.31.715252

Paper 3 -- Cogni-Action (in preparation):
  Martinez-Flores, R., et al. Functional pupillary dynamics during reading
  comprehension across physical activity conditions. Cogni-Action Project.

---

## Requirements

    install.packages(c("lme4", "lmerTest", "dplyr", "tidyr",
                       "ggplot2", "patchwork", "boot", "emmeans",
                       "mgcv", "readxl", "tidyverse"))

---

## License

This repository is licensed under Creative Commons Attribution 4.0
International (CC BY 4.0). You are free to use and adapt this code
for any purpose, provided you cite the original author and include
a link to this repository.

Ricardo Martinez-Flores -- ricardo.antonio.martinezf@gmail.com
https://github.com/ricardo-martinez-flores/oculomotor-signal-analysis
