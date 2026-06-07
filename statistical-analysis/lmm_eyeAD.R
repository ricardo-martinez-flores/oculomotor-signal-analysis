# lmm_eyeAD.R
#
# Linear mixed model analysis -- Eye-AD project
# Oculomotor features as biomarkers in MCI (pre-post intervention design)
#
# Features analyzed (derived from pupil_eyeAD.py output):
#   - AUC (area under the baseline-corrected curve, positive portion)
#   - Peak amplitude (maximum dilation post-stimulus)
#   - Peak latency (time of maximum dilation, ms)
#   - Mean dilation (mean of post-stimulus window)
#
# Two model types:
#
#   Model 1 -- Keep-it-maximal (Barr et al. 2013)
#     Single outcome, maximal random effects structure.
#     Applied here to vergence/pupil amplitude (Target vs Distractor x Pre vs Post).
#     Rationale: when the fixed effects structure is fully crossed within
#     participants, random slopes for all within-participant predictors
#     should be included to avoid anti-conservative Type I error rates.
#     Reduction is applied only when the maximal model fails to converge
#     or produces a singular fit, following a principled step-down sequence.
#
#   Model 2 -- Multiple features, permutation inference
#     Multiple oculomotor features tested simultaneously.
#     Freedman-Lane permutation for inference (REML -> ML switch for
#     likelihood-ratio based t-statistics, preserving hierarchical structure).
#     Max-T correction (Westfall-Young) for FWER across features.
#     BCa bootstrap for beta coefficients and CIs.
#     Optional covariate (medication, diagnosis group) via Freedman-Lane.
#
# Why permutation rather than parametric p-values from lmer:
#   LMM p-values rely on approximate df (Satterthwaite or Kenward-Roger).
#   With small n or unbalanced designs these approximations can be liberal.
#   Permutation makes no distributional assumptions and controls Type I error
#   exactly under the exchangeability assumption.
#
# Why REML -> ML for the permutation t-statistic:
#   REML estimates are not comparable across models with different fixed
#   effects. Switching to ML for the permutation loop allows likelihood-ratio
#   based inference while keeping REML for the final parameter estimates.
#   The permuted t-statistic is computed from the ML fit; the reported
#   coefficients and CIs come from the REML fit.
#
# Author: Ricardo Martinez-Flores
# Contact: ricardo.antonio.martinezf@gmail.com
# License: MIT

library(lme4)
library(lmerTest)    # Satterthwaite df for model summary (not used for inference)
library(dplyr)
library(tidyr)
library(ggplot2)
library(patchwork)
library(boot)
library(emmeans)

# Resolve namespace conflicts
select    <- dplyr::select
filter    <- dplyr::filter
mutate    <- dplyr::mutate
summarise <- dplyr::summarise

set.seed(42)


# =============================================================================
# CONFIGURATION
# =============================================================================

N_PERM <- 5000
N_BOOT <- 5000
ALPHA  <- 0.05

output_dir <- "~/Desktop/lmm_eyeAD_outputs"
dir.create(path.expand(output_dir), recursive = TRUE, showWarnings = FALSE)
out <- function(f) file.path(path.expand(output_dir), f)

# Feature column names in the data file (output of pupil_eyeAD.py aggregated)
# Expected: one row per participant x condition x timepoint
# Columns: participant, condition (Target/Distractor), timepoint (pre/post),
#          auc, peak_amplitude, peak_latency_ms, mean_dilation
FEATURES <- c("auc", "peak_amplitude", "peak_latency_ms", "mean_dilation")

# Covariate column (set to NULL to disable Freedman-Lane)
COL_COVARIATE <- "diagnosis"   # 0 = healthy, 1 = MCI; set NULL if not used

# Font for figures
FONT <- "Times New Roman"
plt_theme <- theme_bw(base_size = 14, base_family = FONT) +
  theme(
    plot.title         = element_text(face = "bold", hjust = 0.5, size = 16),
    axis.title         = element_text(face = "bold", size = 14),
    axis.text          = element_text(face = "bold", size = 13),
    panel.grid.minor   = element_blank(),
    panel.grid.major.x = element_blank(),
    legend.position    = "right",
    legend.text        = element_text(size = 12)
  )


# =============================================================================
# DATA LOADING
# =============================================================================

# Expected CSV: output from pupil_eyeAD.py aggregated to trial-mean per
# participant x stimulus x condition (pre/post). Columns:
#   participant, stimulus (Target/Distractor), timepoint (pre/post),
#   diagnosis (0/1), auc, peak_amplitude, peak_latency_ms, mean_dilation
#
# To aggregate from the raw trial output:
#   df_raw <- read.csv("pupil_trials_clean.csv", sep=";", dec=",")
#   df <- df_raw %>%
#     group_by(participant, stimulus, condition) %>%
#     summarise(across(c(auc, peak_amplitude, peak_latency_ms, mean_dilation),
#                      mean, na.rm=TRUE), .groups="drop") %>%
#     rename(timepoint = condition)

data_path <- "~/Desktop/Palabras/outputs/pupil_features_aggregated.csv"
df_raw <- read.csv(data_path)

df <- df_raw %>%
  mutate(
    participant = factor(participant),
    stimulus    = factor(stimulus,  levels = c("Distractor", "Target")),
    timepoint   = factor(timepoint, levels = c("pre", "post"))
  )

if (!is.null(COL_COVARIATE) && COL_COVARIATE %in% names(df)) {
  df[[COL_COVARIATE]] <- as.numeric(df[[COL_COVARIATE]])
}

cat(sprintf("Participants: %d\n", n_distinct(df$participant)))
cat(sprintf("Stimulus levels: %s\n", paste(levels(df$stimulus),   collapse=", ")))
cat(sprintf("Timepoint levels: %s\n", paste(levels(df$timepoint), collapse=", ")))


# =============================================================================
# MODEL 1 -- KEEP-IT-MAXIMAL (Barr et al. 2013)
# Applied to AUC as the primary outcome
# Fixed effects: stimulus x timepoint interaction
# Random effects: maximal structure for within-participant predictors
# =============================================================================

cat("\n", strrep("=", 70), "\n")
cat("MODEL 1 -- KEEP-IT-MAXIMAL\n")
cat("Outcome: AUC  |  Fixed: stimulus * timepoint\n")
cat(strrep("=", 70), "\n\n")

# Maximal model.
# Includes random slopes for stimulus, timepoint, and their interaction
# for each participant, plus random intercept.
# This is the maximal structure justified by the design: both stimulus
# (Target vs Distractor) and timepoint (pre vs post) vary within participants.
#
# Formula notation:
#   (1 + stimulus * timepoint | participant)
# expands to:
#   (1 + stimulus + timepoint + stimulus:timepoint | participant)
# which allows each participant to have their own intercept,
# stimulus slope, timepoint slope, and interaction slope.

m1_maximal <- tryCatch(
  lmer(auc ~ stimulus * timepoint +
         (1 + stimulus * timepoint | participant),
       data = df, REML = TRUE,
       control = lmerControl(optimizer = "bobyqa",
                             optCtrl   = list(maxfun = 2e5))),
  error = function(e) { cat("Maximal model error:", e$message, "\n"); NULL }
)

# Step-down sequence if maximal model fails or is singular.
# Order of removal follows Barr et al.: remove highest-order random terms first,
# keeping lower-order terms to maintain the hierarchy.
# We do NOT remove random intercepts -- that would assume all participants
# have identical baseline responses, which is never justified here.

fit_model1 <- function(df) {
  formulas <- list(
    maximal   = auc ~ stimulus * timepoint + (1 + stimulus * timepoint | participant),
    no_3way   = auc ~ stimulus * timepoint + (1 + stimulus + timepoint | participant),
    slopes_only = auc ~ stimulus * timepoint + (1 + stimulus | participant),
    intercept = auc ~ stimulus * timepoint + (1 | participant)
  )
  ctrl <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

  for (nm in names(formulas)) {
    m <- tryCatch(
      lmer(formulas[[nm]], data = df, REML = TRUE, control = ctrl),
      error = function(e) NULL
    )
    if (!is.null(m)) {
      sing <- isSingular(m)
      cat(sprintf("  %s: converged=%s  singular=%s\n",
                  nm, !is.null(m), sing))
      if (!sing) {
        cat(sprintf("  Selected: %s\n", nm))
        return(list(model = m, formula_name = nm))
      }
    }
  }
  cat("  All models singular -- returning intercept-only random effects\n")
  m <- lmer(formulas$intercept, data = df, REML = TRUE, control = ctrl)
  list(model = m, formula_name = "intercept")
}

cat("Fitting maximal model and step-down if needed:\n")
m1_result <- fit_model1(df)
m1        <- m1_result$model

cat("\nModel 1 summary:\n")
print(summary(m1))

# Marginal means from the selected model
emm1 <- emmeans(m1, ~ stimulus * timepoint)
cat("\nMarginal means (stimulus x timepoint):\n")
print(emm1)

sink(out("model1_summary.txt"))
cat(sprintf("Formula: %s\n\n", deparse(formula(m1))))
print(summary(m1))
cat("\nMarginal means:\n")
print(emm1)
sink()


# =============================================================================
# MODEL 2 -- MULTIPLE FEATURES, PERMUTATION INFERENCE
# Features: AUC, peak amplitude, peak latency, mean dilation
# Fixed effect of interest: stimulus (Target vs Distractor), controlling
# for timepoint and optional covariate (diagnosis)
# Random effects: (1 + timepoint | participant)
# Permutation: Freedman-Lane, REML->ML, max-T FWER
# =============================================================================

cat("\n", strrep("=", 70), "\n")
cat("MODEL 2 -- MULTIPLE FEATURES, PERMUTATION + MAX-T\n")
cat(strrep("=", 70), "\n\n")


# ---------------------------------------------------------------------------
# Helper: extract t-statistic for a fixed effect from an lmer model
# Uses ML (not REML) for permutation loop comparability
# ---------------------------------------------------------------------------

get_t_stat <- function(model, effect_name) {
  coefs <- summary(model)$coefficients
  if (!effect_name %in% rownames(coefs)) return(NA_real_)
  coefs[effect_name, "t value"]
}


# ---------------------------------------------------------------------------
# Freedman-Lane permutation for one feature
#
# Procedure:
#   1. Fit reduced model (no predictor of interest, ML)
#   2. Extract residuals from reduced model
#   3. For each permutation, permute residuals within participant
#      (respects the hierarchical structure -- within-participant exchangeability)
#   4. Refit full model on permuted outcome
#   5. Record t-statistic for predictor of interest
#   6. p = proportion of |t_perm| >= |t_obs|
#
# Why permute within participant:
#   Simple row permutation would destroy the within-participant correlation
#   structure. Permuting residuals within participant preserves the random
#   effect structure under H0 while randomizing the fixed effect of interest.
# ---------------------------------------------------------------------------

freedman_lane_lmm <- function(df_feat, outcome, predictor,
                               covariates = NULL,
                               n_perm = N_PERM) {
  df_feat <- df_feat %>% filter(!is.na(.data[[outcome]]))

  has_cov <- !is.null(covariates) &&
             all(covariates %in% names(df_feat)) &&
             all(sapply(covariates, function(cv)
               length(unique(df_feat[[cv]][!is.na(df_feat[[cv]])])) >= 2))

  ctrl <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))

  # Build formula strings
  fixed_full <- if (has_cov)
    paste0(outcome, " ~ ", predictor, " + timepoint + ",
           paste(covariates, collapse = " + "))
  else
    paste0(outcome, " ~ ", predictor, " + timepoint")

  fixed_red <- if (has_cov)
    paste0(outcome, " ~ timepoint + ", paste(covariates, collapse = " + "))
  else
    paste0(outcome, " ~ timepoint")

  # Random slopes for timepoint (varies within participant)
  rand <- "(1 + timepoint | participant)"

  # Fit full model with ML for t-statistic
  m_full_ml <- tryCatch(
    lmer(as.formula(paste(fixed_full, "+", rand)),
         data = df_feat, REML = FALSE, control = ctrl),
    error = function(e) {
      lmer(as.formula(paste0(outcome, " ~ ", predictor,
                             " + timepoint + (1 | participant)")),
           data = df_feat, REML = FALSE, control = ctrl)
    }
  )
  t_obs <- get_t_stat(m_full_ml, paste0(predictor, "Target"))
  if (is.na(t_obs)) {
    # Try alternate coefficient name
    coef_names <- rownames(summary(m_full_ml)$coefficients)
    pred_coef  <- coef_names[grepl(predictor, coef_names)][1]
    t_obs <- if (!is.na(pred_coef)) get_t_stat(m_full_ml, pred_coef) else NA
  }

  # Reduced model (ML) for residuals under H0
  m_red_ml <- tryCatch(
    lmer(as.formula(paste(fixed_red, "+", rand)),
         data = df_feat, REML = FALSE, control = ctrl),
    error = function(e)
      lmer(as.formula(paste0(outcome, " ~ timepoint + (1 | participant)")),
           data = df_feat, REML = FALSE, control = ctrl)
  )

  resid_h0  <- residuals(m_red_ml)
  fitted_h0 <- fitted(m_red_ml)

  # Permutation loop: shuffle residuals within participant
  participants <- unique(df_feat$participant)
  t_perm_vec  <- numeric(n_perm)

  pred_coef_name <- rownames(summary(m_full_ml)$coefficients)
  pred_coef_name <- pred_coef_name[grepl(predictor, pred_coef_name)][1]

  for (i in seq_len(n_perm)) {
    # Permute residuals within each participant independently
    perm_resid <- resid_h0
    for (pid in participants) {
      idx <- which(df_feat$participant == pid)
      perm_resid[idx] <- sample(resid_h0[idx])
    }
    df_feat$.y_perm <- fitted_h0 + perm_resid

    m_perm <- tryCatch(
      lmer(as.formula(sub(outcome, ".y_perm",
                          paste(fixed_full, "+", rand), fixed = TRUE)),
           data = df_feat, REML = FALSE, control = ctrl),
      error = function(e) NULL
    )
    t_perm_vec[i] <- if (!is.null(m_perm) && !is.na(pred_coef_name))
      get_t_stat(m_perm, pred_coef_name) else 0
  }

  p_perm <- max(mean(abs(t_perm_vec) >= abs(t_obs), na.rm = TRUE),
                1 / n_perm)

  # REML fit for final coefficient and BCa CI
  m_full_reml <- tryCatch(
    lmer(as.formula(paste(fixed_full, "+", rand)),
         data = df_feat, REML = TRUE, control = ctrl),
    error = function(e)
      lmer(as.formula(paste0(outcome, " ~ ", predictor,
                             " + timepoint + (1 | participant)")),
           data = df_feat, REML = TRUE, control = ctrl)
  )

  list(
    t_obs      = t_obs,
    p_perm     = p_perm,
    t_perm     = t_perm_vec,
    model_ml   = m_full_ml,
    model_reml = m_full_reml,
    coef_name  = pred_coef_name
  )
}


# ---------------------------------------------------------------------------
# BCa bootstrap for beta coefficient
# ---------------------------------------------------------------------------

bca_beta <- function(df_feat, outcome, predictor,
                     covariates = NULL, n_boot = N_BOOT) {
  ctrl <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))

  has_cov <- !is.null(covariates) && all(covariates %in% names(df_feat))
  fixed <- if (has_cov)
    paste0(outcome, " ~ ", predictor, " + timepoint + ",
           paste(covariates, collapse = " + "))
  else
    paste0(outcome, " ~ ", predictor, " + timepoint")

  rand <- "(1 + timepoint | participant)"

  boot_stat <- function(data, idx) {
    # Case bootstrap: resample participants (preserves within-participant structure)
    pids <- unique(data$participant)
    pids_boot <- sample(pids, length(pids), replace = TRUE)
    df_b <- dplyr::bind_rows(lapply(seq_along(pids_boot), function(k) {
      data %>% filter(participant == pids_boot[k]) %>%
        mutate(participant = factor(paste0("p", k)))
    }))
    m <- tryCatch(
      lmer(as.formula(paste(fixed, "+", rand)),
           data = df_b, REML = TRUE, control = ctrl),
      error = function(e) NULL
    )
    if (is.null(m)) return(NA_real_)
    coefs <- fixef(m)
    pred_c <- names(coefs)[grepl(predictor, names(coefs))][1]
    if (is.na(pred_c)) NA_real_ else coefs[pred_c]
  }

  bo <- tryCatch(
    boot(data = df_feat, statistic = boot_stat, R = n_boot),
    error = function(e) NULL
  )
  if (is.null(bo)) return(list(beta = NA, ci_lo = NA, ci_hi = NA))

  ci <- tryCatch(
    boot.ci(bo, conf = 1 - ALPHA, type = "bca"),
    error = function(e) NULL
  )
  beta_obs <- fixef(tryCatch(
    lmer(as.formula(paste(fixed, "+", rand)),
         data = df_feat, REML = TRUE, control = ctrl),
    error = function(e) NULL
  ))
  pred_c <- names(beta_obs)[grepl(predictor, names(beta_obs))][1]

  list(
    beta  = if (!is.na(pred_c)) beta_obs[pred_c] else NA,
    ci_lo = if (!is.null(ci$bca)) ci$bca[4] else NA,
    ci_hi = if (!is.null(ci$bca)) ci$bca[5] else NA
  )
}


# ---------------------------------------------------------------------------
# Westfall-Young Max-T (joint permutation across features)
# One permutation per iteration applied to ALL features simultaneously,
# preserving correlation structure between features.
# ---------------------------------------------------------------------------

max_t_features <- function(df, features, predictor,
                            covariates = NULL, n_perm = N_PERM) {
  cat("Computing Max-T null distribution (joint permutation)...\n")

  ctrl <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))

  # Observed t-statistics
  t_obs <- setNames(numeric(length(features)), features)
  red_models <- list()
  full_forms  <- list()
  red_forms   <- list()

  has_cov <- !is.null(covariates) && all(covariates %in% names(df))

  for (feat in features) {
    df_f <- df %>% filter(!is.na(.data[[feat]]))

    fixed_full <- if (has_cov)
      paste0(feat, " ~ ", predictor, " + timepoint + ",
             paste(covariates, collapse = " + "))
    else
      paste0(feat, " ~ ", predictor, " + timepoint")

    fixed_red <- if (has_cov)
      paste0(feat, " ~ timepoint + ", paste(covariates, collapse = " + "))
    else
      paste0(feat, " ~ timepoint")

    rand <- "(1 + timepoint | participant)"

    m_full <- tryCatch(
      lmer(as.formula(paste(fixed_full, "+", rand)),
           data = df_f, REML = FALSE, control = ctrl),
      error = function(e)
        lmer(as.formula(paste0(feat, " ~ ", predictor,
                               " + timepoint + (1|participant)")),
             data = df_f, REML = FALSE, control = ctrl)
    )
    m_red <- tryCatch(
      lmer(as.formula(paste(fixed_red, "+", rand)),
           data = df_f, REML = FALSE, control = ctrl),
      error = function(e)
        lmer(as.formula(paste0(feat, " ~ timepoint + (1|participant)")),
             data = df_f, REML = FALSE, control = ctrl)
    )

    coef_names <- rownames(summary(m_full)$coefficients)
    pred_c     <- coef_names[grepl(predictor, coef_names)][1]
    t_obs[feat] <- if (!is.na(pred_c)) get_t_stat(m_full, pred_c) else NA

    red_models[[feat]] <- m_red
    full_forms[[feat]]  <- fixed_full
    red_forms[[feat]]   <- fixed_red
  }

  # Joint permutation loop
  max_t_dist <- numeric(n_perm)
  participants <- unique(df$participant)

  for (i in seq_len(n_perm)) {
    # One permutation map per participant, shared across all features
    swap_map <- setNames(
      sample(c(-1L, 1L), length(participants), replace = TRUE),
      as.character(participants)
    )

    t_perm_i <- numeric(length(features))
    names(t_perm_i) <- features

    for (feat in features) {
      df_f      <- df %>% filter(!is.na(.data[[feat]]))
      m_red     <- red_models[[feat]]
      resid_h0  <- residuals(m_red)
      fitted_h0 <- fitted(m_red)

      # Apply participant-level sign based on shared swap_map
      signs <- swap_map[as.character(df_f$participant)]
      df_f$.y_perm <- fitted_h0 + signs * resid_h0

      m_perm <- tryCatch(
        lmer(as.formula(sub(feat, ".y_perm",
                            paste(full_forms[[feat]], "+ (1+timepoint|participant)"),
                            fixed = TRUE)),
             data = df_f, REML = FALSE, control = ctrl),
        error = function(e) NULL
      )
      coef_names <- if (!is.null(m_perm))
        rownames(summary(m_perm)$coefficients) else character(0)
      pred_c <- coef_names[grepl(predictor, coef_names)][1]
      t_perm_i[feat] <- if (!is.null(m_perm) && !is.na(pred_c))
        get_t_stat(m_perm, pred_c) else 0
    }
    max_t_dist[i] <- max(abs(t_perm_i), na.rm = TRUE)

    if (i %% 500 == 0) cat(sprintf("  Max-T: %d/%d\n", i, n_perm))
  }

  # Max-T p-values
  p_maxt <- setNames(numeric(length(features)), features)
  for (feat in features) {
    p_maxt[feat] <- max(mean(max_t_dist >= abs(t_obs[feat]), na.rm = TRUE),
                        1 / n_perm)
  }

  list(t_obs = t_obs, p_maxt = p_maxt, null_dist = max_t_dist)
}


# ---------------------------------------------------------------------------
# Run Model 2
# ---------------------------------------------------------------------------

cat("Running Freedman-Lane permutation for each feature...\n")

perm_results <- list()
for (feat in FEATURES) {
  cat(sprintf("\n  Feature: %s\n", feat))
  perm_results[[feat]] <- freedman_lane_lmm(
    df_feat    = df,
    outcome    = feat,
    predictor  = "stimulus",
    covariates = if (!is.null(COL_COVARIATE)) COL_COVARIATE else NULL,
    n_perm     = N_PERM
  )
  cat(sprintf("  t_obs=%.3f  p_perm=%.4f\n",
              perm_results[[feat]]$t_obs,
              perm_results[[feat]]$p_perm))
}

cat("\nRunning Max-T joint permutation...\n")
maxt_res <- max_t_features(
  df         = df,
  features   = FEATURES,
  predictor  = "stimulus",
  covariates = if (!is.null(COL_COVARIATE)) COL_COVARIATE else NULL,
  n_perm     = N_PERM
)

cat("\nRunning BCa bootstrap for each feature...\n")
bca_results <- list()
for (feat in FEATURES) {
  cat(sprintf("  Feature: %s\n", feat))
  bca_results[[feat]] <- bca_beta(
    df_feat    = df,
    outcome    = feat,
    predictor  = "stimulus",
    covariates = if (!is.null(COL_COVARIATE)) COL_COVARIATE else NULL,
    n_boot     = N_BOOT
  )
}


# ---------------------------------------------------------------------------
# Assemble results table
# ---------------------------------------------------------------------------

fmt_p <- function(p) {
  if (is.na(p))  return("-")
  if (p < 0.001) return("<.001")
  formatC(p, digits = 3, format = "f")
}

results_df <- dplyr::bind_rows(lapply(FEATURES, function(feat) {
  pr  <- perm_results[[feat]]
  bca <- bca_results[[feat]]
  data.frame(
    feature    = feat,
    beta       = round(bca$beta, 4),
    ci_lo      = round(bca$ci_lo, 4),
    ci_hi      = round(bca$ci_hi, 4),
    t_obs      = round(pr$t_obs, 3),
    p_perm     = pr$p_perm,
    p_maxt     = maxt_res$p_maxt[feat],
    sig_perm   = ifelse(pr$p_perm           < ALPHA, "*", ""),
    sig_maxt   = ifelse(maxt_res$p_maxt[feat] < ALPHA, "*", ""),
    stringsAsFactors = FALSE
  )
}))

cat("\n", strrep("=", 70), "\n")
cat("MODEL 2 RESULTS\n")
cat(strrep("=", 70), "\n")
print(results_df)
cat("\nNote: beta = Target - Distractor (positive = larger response to Target)\n")
cat("      CI from BCa bootstrap (participant-level resampling)\n")
cat("      p_perm: Freedman-Lane within-participant permutation\n")
cat("      p_maxt: Westfall-Young Max-T (joint permutation, FWER)\n")

write.csv(results_df, out("model2_results.csv"), row.names = FALSE)


# =============================================================================
# FIGURES
# =============================================================================

# ---------------------------------------------------------------------------
# Figure 1: Estimated marginal means from Model 1 (AUC, stimulus x timepoint)
# ---------------------------------------------------------------------------

emm_df <- as.data.frame(summary(emm1)) %>%
  rename(emmean_val = emmean, se_val = SE)

emm_df$stimulus  <- factor(emm_df$stimulus,  levels = c("Distractor", "Target"))
emm_df$timepoint <- factor(emm_df$timepoint, levels = c("pre", "post"))

fig_emm <- ggplot(emm_df,
                  aes(x = timepoint, y = emmean_val,
                      color = stimulus, group = stimulus)) +
  geom_line(lwd = 1.2, position = position_dodge(0.15)) +
  geom_errorbar(aes(ymin = emmean_val - se_val,
                    ymax = emmean_val + se_val),
                width = 0.12, lwd = 1.0,
                position = position_dodge(0.15)) +
  geom_point(size = 3.5, position = position_dodge(0.15)) +
  scale_color_manual(values = c(Distractor = "#4361EE", Target = "#D62828")) +
  labs(
    title = "Estimated marginal means -- AUC (Model 1)",
    x     = "Timepoint",
    y     = "AUC (estimated marginal mean +/- SE)",
    color = "Stimulus"
  ) +
  plt_theme

ggsave(out("fig_emm_auc.png"), fig_emm,
       width = 7, height = 5, dpi = 300, bg = "white")
cat("Saved: fig_emm_auc.png\n")


# ---------------------------------------------------------------------------
# Figure 2: Forest plot -- Model 2 beta coefficients with BCa CIs
# ---------------------------------------------------------------------------

results_df$sig_label <- ifelse(results_df$sig_maxt == "*",
                                sprintf("%s *", results_df$feature),
                                results_df$feature)

fig_forest <- ggplot(results_df,
                     aes(x = beta, y = reorder(feature, beta),
                         xmin = ci_lo, xmax = ci_hi)) +
  geom_vline(xintercept = 0, color = "gray40", lwd = 0.8, lty = "dashed") +
  geom_errorbarh(height = 0.25, lwd = 1.0, color = "#2E86AB") +
  geom_point(size = 4, color = "#2E86AB") +
  geom_text(
    aes(label = sprintf("beta=%.3f\np_maxt=%s", beta, fmt_p(p_maxt))),
    hjust = -0.12, size = 11 / .pt,
    family = FONT, fontface = "bold", color = "black", lineheight = 1.0
  ) +
  labs(
    title = "Model 2 -- Stimulus effect (Target - Distractor)\nBeta + 95% BCa CI",
    x     = "Beta coefficient (Target - Distractor)",
    y     = NULL
  ) +
  plt_theme

ggsave(out("fig_forest_model2.png"), fig_forest,
       width = 9, height = 5, dpi = 300, bg = "white")
cat("Saved: fig_forest_model2.png\n")


# ---------------------------------------------------------------------------
# Figure 3: Observed data -- mean +/- SE per stimulus x timepoint
# for each feature (raw data complement to EMM figure)
# ---------------------------------------------------------------------------

df_sum <- df %>%
  pivot_longer(cols = all_of(FEATURES), names_to = "feature", values_to = "value") %>%
  filter(!is.na(value)) %>%
  group_by(stimulus, timepoint, feature) %>%
  summarise(
    mean_val = mean(value, na.rm = TRUE),
    se_val   = sd(value, na.rm = TRUE) / sqrt(n()),
    .groups  = "drop"
  ) %>%
  mutate(
    stimulus  = factor(stimulus,  levels = c("Distractor", "Target")),
    timepoint = factor(timepoint, levels = c("pre", "post")),
    feature   = factor(feature,   levels = FEATURES)
  )

fig_obs <- ggplot(df_sum,
                  aes(x = timepoint, y = mean_val,
                      color = stimulus, group = stimulus)) +
  geom_line(lwd = 1.1, position = position_dodge(0.15)) +
  geom_errorbar(aes(ymin = mean_val - se_val,
                    ymax = mean_val + se_val),
                width = 0.12, lwd = 0.9,
                position = position_dodge(0.15)) +
  geom_point(size = 3, position = position_dodge(0.15)) +
  scale_color_manual(values = c(Distractor = "#4361EE", Target = "#D62828")) +
  facet_wrap(~ feature, scales = "free_y", ncol = 2) +
  labs(
    title = "Observed means +/- SE by stimulus and timepoint",
    x     = "Timepoint",
    y     = "Feature value (mean +/- SE)",
    color = "Stimulus"
  ) +
  plt_theme +
  theme(strip.text = element_text(face = "bold", size = 13, family = FONT))

ggsave(out("fig_observed_features.png"), fig_obs,
       width = 10, height = 7, dpi = 300, bg = "white")
cat("Saved: fig_observed_features.png\n")


# =============================================================================
# CONSOLE SUMMARY
# =============================================================================

cat("\n", strrep("=", 70), "\n")
cat("FINAL SUMMARY\n")
cat(strrep("=", 70), "\n\n")

cat("Model 1 (keep-it-maximal, AUC):\n")
cat(sprintf("  Formula selected: %s\n", m1_result$formula_name))
cat(sprintf("  Fixed effects (stimulus x timepoint interaction):\n"))
print(round(summary(m1)$coefficients, 4))

cat("\nModel 2 (multiple features, permutation):\n")
cat(sprintf("  Predictor: stimulus (Target vs Distractor)\n"))
cat(sprintf("  N permutations: %d  |  N bootstrap: %d\n", N_PERM, N_BOOT))
if (!is.null(COL_COVARIATE))
  cat(sprintf("  Covariate controlled: %s (Freedman-Lane)\n", COL_COVARIATE))
cat(sprintf("  FWER correction: Westfall-Young Max-T (joint permutation)\n\n"))
print(results_df %>%
        select(feature, beta, ci_lo, ci_hi, t_obs, p_perm, p_maxt,
               sig_perm, sig_maxt) %>%
        mutate(across(c(p_perm, p_maxt), fmt_p)))

cat(sprintf("\nOutputs saved to: %s\n", path.expand(output_dir)))
