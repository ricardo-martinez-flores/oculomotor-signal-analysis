# gamm_cogniaccion.R
#
# GAMM + cluster-based permutation -- Cogni-Action project
# Pupillary dynamics during reading comprehension (3 conditions: SC, MICT, C-HIIT)
#
# Input: BD_fpca.csv (output of pupil_cogniaccion.py -> fPCA normalization)
#   Columns: Participant, Condition, Time_pct, Pupil_avg_bc
#
# Two model types are implemented and compared:
#
#   A) Smooth-by-condition interaction (s + s:by)
#      gamm ~ Condition + s(Time_pct) + s(Time_pct, by=Condition) + s(Participant, re)
#      Fits a global smooth plus condition-specific deviation smooths.
#      Each condition gets its own wiggliness (k, lambda). Interpretable as
#      "how does the temporal trajectory differ from the reference condition?"
#      Reference condition (SC) smooth is absorbed into the global s(Time_pct).
#
#   B) Tensor product interaction (te or ti)
#      gamm ~ s(Participant, bs="re") + te(Time_pct, Condition_numeric)
#      OR with ti (interaction only, no main effects):
#      gamm ~ Condition + s(Time_pct) + ti(Time_pct, Condition_numeric) + s(Participant, re)
#      Models the Time x Condition interaction as a 2D smooth surface.
#      Does not assume that the interaction has the same wiggliness as the
#      main effects -- each dimension has its own smoothing parameter.
#      Better when you expect the shape of the interaction to be complex
#      and not decomposable into additive condition offsets.
#
# When to use te/ti vs s:by:
#   s(x, by=factor): condition is categorical. Each level gets its own smooth
#     deviation. Best when you have a clear reference level and want to test
#     whether each condition differs from it in shape over time.
#     Assumes each condition smooth has the same basis dimension k.
#   te(x, z): z is treated as continuous (requires numeric coding of condition).
#     Models the interaction surface directly without a reference level.
#     More flexible but harder to interpret and requires more data.
#     Use when conditions are ordered (e.g., intensity levels) or when the
#     "shape" of how conditions differ over time is the primary question.
#   ti(x, z): like te but removes the main effects, leaving only the
#     interaction. Use alongside s(x) and s(z) as main effects for a clean
#     decomposition of main effect vs interaction.
#
# This script uses approach A (s:by) as the primary model and provides
# the te/ti alternative as commented code with explanation.
#
# Inference: cluster-based permutation test on the GAMM-derived t-statistic
#   curve (pointwise difference / SE). Clusters are contiguous time regions
#   where |t| > t_crit. The null distribution of the maximum cluster mass
#   is built by permuting condition labels within participants.
#
# Effect size: Hedges g on AUC within each significant cluster window.
#   BCa bootstrap CI on the AUC-based g.
#
# Author: Ricardo Martinez-Flores
# Contact: ricardo.antonio.martinezf@gmail.com
# License: MIT

if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, mgcv, boot, ggplot2, patchwork, readxl)

select    <- dplyr::select
filter    <- dplyr::filter
mutate    <- dplyr::mutate
summarise <- dplyr::summarise
arrange   <- dplyr::arrange


# =============================================================================
# CONFIGURATION
# =============================================================================

RUTA_CSV      <- "~/Desktop/BD_fpca.csv"
OUT_DIR       <- "~/Desktop/gamm_cogniaccion_outputs"
N_PERM        <- 10000
N_BOOT        <- 5000
SEED          <- 42
ALPHA_CLUSTER <- 0.05
ALPHA_GLOBAL  <- 0.05
N_GRID_GAMM   <- 200
FONT          <- "Times New Roman"

COND_COLORS   <- c("C-HIIT" = "#E41A1C",
                   "MICT"   = "#377EB8",
                   "SC"     = "#4DAF4A")
CONDS_ORDERED <- c("SC", "MICT", "C-HIIT")

COMPARISONS <- list(
  c("SC",   "MICT"),
  c("SC",   "C-HIIT"),
  c("MICT", "C-HIIT")
)

comp_colors <- c(
  "SC vs MICT"     = unname(COND_COLORS["MICT"]),
  "SC vs C-HIIT"   = unname(COND_COLORS["C-HIIT"]),
  "MICT vs C-HIIT" = "gray40"
)

set.seed(SEED)
out_path <- path.expand(OUT_DIR)
if (!dir.exists(out_path)) dir.create(out_path, recursive = TRUE)
out <- function(f) file.path(out_path, f)


# =============================================================================
# 1. LOAD DATA
# =============================================================================

df <- readr::read_csv(RUTA_CSV, show_col_types = FALSE)
stopifnot(all(c("Participant","Condition","Time_pct","Pupil_avg_bc") %in% names(df)))

df <- df %>%
  mutate(
    Condition   = factor(Condition,   levels = CONDS_ORDERED),
    Participant = factor(Participant),
    # Numeric coding for te/ti models (see section 2B)
    Condition_num = as.numeric(Condition)
  )

cat(sprintf("N rows: %d | Participants: %d | Time range: %.2f-%.2f%%\n",
            nrow(df), n_distinct(df$Participant),
            min(df$Time_pct), max(df$Time_pct)))

# Downsample if very large (>200k rows) to keep GAMM fitting tractable
if (nrow(df) > 200000) {
  step <- ceiling(nrow(df) / 200000)
  df <- df %>%
    group_by(Participant, Condition) %>%
    dplyr::slice(seq(1, n(), by = step)) %>%
    ungroup()
  cat(sprintf("Downsampled to %d rows (step=%d)\n", nrow(df), step))
}


# =============================================================================
# 2A. GAMM -- smooth-by-condition (primary model)
#
# s(Time_pct, k=20): global smooth across all conditions
# s(Time_pct, by=Condition, k=20): condition-specific deviation smooths
#   Reference level (SC) deviation is constrained to zero, so SC is
#   represented by the global smooth alone. MICT and C-HIIT get additional
#   smooth terms that represent their deviation from SC over time.
# s(Participant, bs="re"): random intercept for participant
#   (accounts for between-participant baseline differences)
#
# Note: we do not include random slopes over time here because bam()
# with random slopes over a continuous predictor is very slow. If sample
# size permits, consider (1 + Time_pct | Participant) via gamm4::gamm4().
# =============================================================================

cat("\nFitting GAMM (s:by interaction)...\n")
df <- df %>% mutate(Condition = relevel(Condition, ref = "SC"))

gamm_fit <- mgcv::bam(
  Pupil_avg_bc ~ Condition +
    s(Time_pct, k = 20) +
    s(Time_pct, by = Condition, k = 20) +
    s(Participant, bs = "re"),
  data     = df,
  method   = "fREML",
  discrete = TRUE
)

cat("\nGAMM summary:\n")
print(summary(gamm_fit))
sink(out("gamm_summary.txt")); print(summary(gamm_fit)); sink()


# =============================================================================
# 2B. GAMM -- tensor product interaction (alternative)
#
# This block is provided as a reference for when te/ti is more appropriate.
# It is NOT run by default -- set RUN_TENSOR = TRUE to execute.
#
# te(Time_pct, Condition_num): joint smooth of time and condition (numeric).
#   Condition must be numeric. Here 1=SC, 2=MICT, 3=C-HIIT.
#   This is only valid if the ordering is meaningful (e.g., intensity gradient).
#   For purely categorical conditions, s:by is preferred.
#
# ti decomposition (recommended over raw te when main effects matter):
#   ~ Condition + s(Time_pct, k=20) + ti(Time_pct, Condition_num, k=c(10,3))
#   s(Time_pct): main effect of time (shared across conditions)
#   ti(Time_pct, Condition_num): interaction surface (deviation from additivity)
#   This allows a clean test of whether the time effect differs by condition
#   beyond a simple vertical shift (which Condition alone would capture).
#
# Key difference from s:by:
#   s:by tests per-condition deviations relative to the reference level.
#   te/ti models the interaction surface without assuming a reference level,
#   making it more symmetric but also harder to interpret for multiple groups.
# =============================================================================

RUN_TENSOR <- FALSE   # set TRUE to run the tensor product alternative

if (RUN_TENSOR) {
  cat("\nFitting GAMM with tensor product interaction (ti decomposition)...\n")

  gamm_tensor <- mgcv::bam(
    Pupil_avg_bc ~ Condition +
      s(Time_pct, k = 20) +
      ti(Time_pct, Condition_num, k = c(10, 3)) +
      s(Participant, bs = "re"),
    data     = df,
    method   = "fREML",
    discrete = TRUE
  )

  cat("\nGAMM tensor summary:\n")
  print(summary(gamm_tensor))
  sink(out("gamm_tensor_summary.txt")); print(summary(gamm_tensor)); sink()

  # AIC comparison
  cat(sprintf("\nAIC comparison:\n  s:by   = %.1f\n  ti     = %.1f\n",
              AIC(gamm_fit), AIC(gamm_tensor)))
  cat("Lower AIC = better fit. Choose accordingly.\n")
}


# =============================================================================
# 3. PREDICTIONS + POINTWISE DIFFERENCES
# =============================================================================

time_grid <- seq(min(df$Time_pct), max(df$Time_pct), length.out = N_GRID_GAMM)
ref_part  <- levels(df$Participant)[1]

pred_list <- lapply(CONDS_ORDERED, function(cond) {
  nd <- data.frame(
    Time_pct    = time_grid,
    Condition   = factor(cond, levels = CONDS_ORDERED),
    Condition_num = which(CONDS_ORDERED == cond),
    Participant = factor(ref_part, levels = levels(df$Participant))
  )
  pred <- predict(gamm_fit, newdata = nd,
                  exclude = "s(Participant)", se.fit = TRUE)
  data.frame(
    Time_pct  = time_grid, Condition = cond,
    fit       = as.numeric(pred$fit),
    se        = as.numeric(pred$se.fit),
    lwr       = as.numeric(pred$fit) - 1.96 * as.numeric(pred$se.fit),
    upr       = as.numeric(pred$fit) + 1.96 * as.numeric(pred$se.fit)
  )
})

df_pred <- dplyr::bind_rows(pred_list) %>%
  mutate(Condition = factor(Condition, levels = CONDS_ORDERED))

diff_list <- lapply(COMPARISONS, function(comp) {
  c1 <- comp[1]; c2 <- comp[2]
  p1 <- df_pred %>% filter(Condition == c1)
  p2 <- df_pred %>% filter(Condition == c2)
  se_d <- sqrt(p1$se^2 + p2$se^2)
  se_d[se_d < 1e-10] <- 1e-10
  data.frame(
    Time_pct   = time_grid,
    Comparison = paste0(c1, " vs ", c2),
    Cond1 = c1, Cond2 = c2,
    diff    = p1$fit - p2$fit,
    se_diff = se_d,
    t_stat  = (p1$fit - p2$fit) / se_d,
    lwr     = (p1$fit - p2$fit) - 1.96 * se_d,
    upr     = (p1$fit - p2$fit) + 1.96 * se_d
  )
})

df_diff <- dplyr::bind_rows(diff_list) %>%
  mutate(Comparison = factor(Comparison,
    levels = c("SC vs MICT", "SC vs C-HIIT", "MICT vs C-HIIT")))


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

find_clusters <- function(t_vec, t_crit) {
  above    <- abs(t_vec) > t_crit
  clusters <- list()
  in_c <- FALSE; start <- NA
  for (i in seq_along(t_vec)) {
    if (above[i] && !in_c) {
      in_c <- TRUE; start <- i
    } else if (!above[i] && in_c) {
      in_c <- FALSE
      idx <- start:(i - 1)
      clusters <- c(clusters, list(list(
        idx = idx, mass = sum(t_vec[idx]^2),
        t_sum = sum(t_vec[idx]),
        start_t = min(idx), end_t = max(idx),
        sign = sign(mean(t_vec[idx]))
      )))
    }
  }
  if (in_c) {
    idx <- start:length(t_vec)
    clusters <- c(clusters, list(list(
      idx = idx, mass = sum(t_vec[idx]^2),
      t_sum = sum(t_vec[idx]),
      start_t = min(idx), end_t = max(idx),
      sign = sign(mean(t_vec[idx]))
    )))
  }
  clusters
}

# Permutation function: swap condition labels within participants
# This preserves the within-participant correlation structure under H0.
perm_condition_swap <- function(df_data, cond1, cond2, time_grid) {
  df_sub <- df_data %>%
    filter(Condition %in% c(cond1, cond2)) %>%
    mutate(Condition = droplevels(Condition))

  participants <- unique(df_sub$Participant)
  swap_map <- setNames(
    sample(c(TRUE, FALSE), length(participants), replace = TRUE),
    as.character(participants)
  )

  df_perm <- df_sub %>%
    mutate(
      Cond_perm = ifelse(
        swap_map[as.character(Participant)],
        ifelse(as.character(Condition) == cond1, cond2, cond1),
        as.character(Condition)
      ),
      Cond_perm = factor(Cond_perm, levels = c(cond1, cond2))
    )

  get_curve <- function(cond) {
    dc <- df_perm %>%
      filter(Cond_perm == cond) %>%
      group_by(Time_pct) %>%
      summarise(m = mean(Pupil_avg_bc, na.rm = TRUE),
                s = sd(Pupil_avg_bc, na.rm = TRUE) / sqrt(n()),
                .groups = "drop") %>%
      arrange(Time_pct)
    list(
      m = approx(dc$Time_pct, dc$m, xout = time_grid, rule = 2)$y,
      s = approx(dc$Time_pct, dc$s, xout = time_grid, rule = 2)$y
    )
  }

  c1_c <- get_curve(cond1)
  c2_c <- get_curve(cond2)
  se_d <- sqrt(c1_c$s^2 + c2_c$s^2)
  se_d[se_d < 1e-10] <- 1e-10
  (c1_c$m - c2_c$m) / se_d
}

get_sig_clusters <- function(res, alpha = ALPHA_GLOBAL) {
  if (length(res$cluster_pvals) == 0) return(list())
  res$cluster_pvals[
    vapply(res$cluster_pvals, function(x) x$p_perm < alpha, logical(1))
  ]
}

hedges_g_indep <- function(x, y) {
  n1 <- sum(!is.na(x)); n2 <- sum(!is.na(y))
  s1 <- var(x, na.rm=TRUE); s2 <- var(y, na.rm=TRUE)
  sp <- sqrt(((n1-1)*s1 + (n2-1)*s2) / (n1+n2-2))
  if (sp == 0) return(NA_real_)
  d <- (mean(x, na.rm=TRUE) - mean(y, na.rm=TRUE)) / sp
  J <- 1 - 3 / (4*(n1+n2-2) - 1)
  d * J
}

trap_auc <- function(t, y) {
  o <- order(t); t <- t[o]; y <- y[o]
  sum(diff(t) * (head(y,-1) + tail(y,-1)) / 2)
}


# =============================================================================
# 4. CLUSTER PERMUTATION TEST
# =============================================================================

cat("\nCluster permutation test...\n")
n_part <- n_distinct(df$Participant)
t_crit <- qt(1 - ALPHA_CLUSTER / 2, df = n_part - 1)
cat(sprintf("t critical (df=%d): %.3f\n\n", n_part - 1, t_crit))

cluster_results <- list()

for (comp in COMPARISONS) {
  c1 <- comp[1]; c2 <- comp[2]
  label <- paste0(c1, " vs ", c2)
  cat(sprintf("-- %s --\n", label))

  t_obs <- df_diff %>%
    filter(Comparison == label) %>%
    arrange(Time_pct) %>%
    pull(t_stat)

  obs_clusters <- find_clusters(t_obs, t_crit)
  cat(sprintf("  Observed clusters: %d\n", length(obs_clusters)))

  df_comp       <- df %>% filter(Condition %in% c(c1, c2))
  null_max_mass <- numeric(N_PERM)

  for (pi in seq_len(N_PERM)) {
    if (pi %% 1000 == 0) cat(sprintf("  %d/%d\n", pi, N_PERM))
    t_p  <- perm_condition_swap(df_comp, c1, c2, time_grid)
    p_cl <- find_clusters(t_p, t_crit)
    null_max_mass[pi] <- if (length(p_cl) > 0)
      max(sapply(p_cl, function(x) x$mass)) else 0
  }

  cluster_pvals <- list()
  if (length(obs_clusters) > 0) {
    for (j in seq_along(obs_clusters)) {
      cl    <- obs_clusters[[j]]
      p_val <- max(mean(null_max_mass >= cl$mass), 1 / N_PERM)
      sig   <- ifelse(p_val < 0.001, "***",
                      ifelse(p_val < 0.01, "**",
                             ifelse(p_val < 0.05, "*",
                                    ifelse(p_val < 0.10, ".", "ns"))))
      cluster_pvals[[j]] <- list(
        cluster_id = j,
        start_pct  = time_grid[cl$start_t],
        end_pct    = time_grid[cl$end_t],
        mass       = cl$mass,
        direction  = ifelse(cl$sign > 0,
                            paste0(c1, ">", c2),
                            paste0(c2, ">", c1)),
        p_perm = p_val, sig = sig
      )
      cat(sprintf("  Cluster %d: %.1f%%-%.1f%%  p=%.4f %s\n",
                  j, time_grid[cl$start_t], time_grid[cl$end_t], p_val, sig))
    }
  }

  cluster_results[[label]] <- list(
    comparison    = label,
    cond1 = c1, cond2 = c2,
    t_obs         = t_obs,
    t_crit        = t_crit,
    obs_clusters  = obs_clusters,
    cluster_pvals = cluster_pvals,
    null_dist     = null_max_mass
  )
}


# =============================================================================
# 5. BCa BOOTSTRAP + HEDGES' g PER SIGNIFICANT CLUSTER
# =============================================================================

cat("\nBCa bootstrap + Hedges g per significant cluster...\n")
bca_results <- list()

for (label in names(cluster_results)) {
  res      <- cluster_results[[label]]
  c1 <- res$cond1; c2 <- res$cond2
  sig_cl   <- get_sig_clusters(res)
  if (length(sig_cl) == 0) next

  cat(sprintf("\n-- %s --\n", label))

  for (cp in sig_cl) {
    cat(sprintf("  Cluster %d: %.1f%%-%.1f%%\n",
                cp$cluster_id, cp$start_pct, cp$end_pct))

    df_win <- df %>%
      filter(Condition %in% c(c1, c2),
             Time_pct  >= cp$start_pct,
             Time_pct  <= cp$end_pct) %>%
      group_by(Participant, Condition) %>%
      summarise(Pupil_win = mean(Pupil_avg_bc, na.rm = TRUE), .groups = "drop") %>%
      pivot_wider(id_cols = Participant, names_from = Condition,
                  values_from = Pupil_win) %>%
      drop_na()

    if (!c1 %in% names(df_win) || !c2 %in% names(df_win)) {
      cat("  WARNING: insufficient data\n"); next
    }

    x <- df_win[[c1]]; y <- df_win[[c2]]
    g_obs <- hedges_g_indep(x, y)

    boot_stat <- function(data, idx) {
      hedges_g_indep(data[[c1]][idx], data[[c2]][idx])
    }
    bo <- boot::boot(data = as.data.frame(df_win), statistic = boot_stat,
                     R = N_BOOT)
    bca_ci <- tryCatch(
      boot::boot.ci(bo, type = "bca", conf = 1 - ALPHA_CLUSTER),
      error = function(e)
        boot::boot.ci(bo, type = "perc", conf = 1 - ALPHA_CLUSTER)
    )
    ci_type <- if (!is.null(bca_ci$bca)) "BCa" else "Percentile"
    ci_vals <- if (!is.null(bca_ci$bca)) bca_ci$bca[4:5] else bca_ci$percent[4:5]

    bca_results[[paste(label, cp$cluster_id)]] <- list(
      Comparison = label, Cond1 = c1, Cond2 = c2,
      Cluster    = cp$cluster_id,
      Start_pct  = cp$start_pct, End_pct = cp$end_pct,
      Hedges_g   = round(g_obs,     3),
      CI_low     = round(ci_vals[1], 3),
      CI_high    = round(ci_vals[2], 3),
      CI_type    = ci_type,
      P_perm     = cp$p_perm, Sig = cp$sig
    )
    cat(sprintf("  g=%.3f [%.3f, %.3f] %s  p=%.4f %s\n",
                g_obs, ci_vals[1], ci_vals[2], ci_type, cp$p_perm, cp$sig))
  }
}

df_bca <- if (length(bca_results) > 0)
  dplyr::bind_rows(lapply(bca_results, as.data.frame))
else data.frame()


# =============================================================================
# 6. AUC BY CLUSTER WINDOWS
# =============================================================================

cat("\nAUC by cluster windows (trapezoidal)...\n")
auc_results <- list()

hedges_g_paired <- function(x, y) {
  d <- x - y; n <- sum(!is.na(d))
  if (n < 3) return(NA_real_)
  J <- 1 - 3 / (4*(n-1) - 1)
  (mean(d, na.rm=TRUE) / sd(d, na.rm=TRUE)) * J
}

for (label in names(cluster_results)) {
  res    <- cluster_results[[label]]
  c1 <- res$cond1; c2 <- res$cond2
  sig_cl <- get_sig_clusters(res)
  if (length(sig_cl) == 0) next

  for (cp in sig_cl) {
    df_win_raw <- df %>%
      filter(Condition %in% c(c1, c2),
             Time_pct  >= cp$start_pct,
             Time_pct  <= cp$end_pct)

    df_auc <- df_win_raw %>%
      group_by(Participant, Condition) %>%
      summarise(AUC = trap_auc(Time_pct, Pupil_avg_bc), .groups = "drop") %>%
      pivot_wider(id_cols = Participant, names_from = Condition,
                  values_from = AUC) %>%
      drop_na()

    if (!c1 %in% names(df_auc) || !c2 %in% names(df_auc)) next

    x_a <- df_auc[[c1]]; y_a <- df_auc[[c2]]
    n_a <- length(x_a)

    obs_diff <- mean(x_a - y_a, na.rm = TRUE)
    null_auc <- replicate(N_PERM, {
      sw <- sample(c(-1L, 1L), n_a, replace = TRUE)
      mean(sw * (x_a - y_a), na.rm = TRUE)
    })
    p_auc <- max(mean(abs(null_auc) >= abs(obs_diff)), 1 / N_PERM)
    g_auc <- hedges_g_paired(x_a, y_a)

    bo_auc <- boot::boot(
      data      = data.frame(x = x_a, y = y_a),
      statistic = function(d, idx) hedges_g_paired(d$x[idx], d$y[idx]),
      R         = N_BOOT
    )
    bca_auc <- tryCatch(
      boot::boot.ci(bo_auc, type = "bca", conf = 1 - ALPHA_CLUSTER),
      error = function(e) boot::boot.ci(bo_auc, type = "perc",
                                         conf = 1 - ALPHA_CLUSTER)
    )
    ci_type_a <- if (!is.null(bca_auc$bca)) "BCa" else "Percentile"
    ci_auc    <- if (!is.null(bca_auc$bca)) bca_auc$bca[4:5] else bca_auc$percent[4:5]

    sig_auc <- ifelse(p_auc < 0.001, "***",
                      ifelse(p_auc < 0.01, "**",
                             ifelse(p_auc < 0.05, "*",
                                    ifelse(p_auc < 0.10, ".", "ns"))))

    auc_results[[paste(label, cp$cluster_id)]] <- data.frame(
      Comparison  = label,
      Cluster     = cp$cluster_id,
      Window      = sprintf("%.1f%%-%.1f%%", cp$start_pct, cp$end_pct),
      Cond1 = c1, Cond2 = c2,
      Mean_AUC_c1 = round(mean(x_a), 4),
      Mean_AUC_c2 = round(mean(y_a), 4),
      AUC_diff    = round(obs_diff,   4),
      P_perm      = round(p_auc,      4),
      Sig         = sig_auc,
      Hedges_g    = round(g_auc,      3),
      CI_low      = round(ci_auc[1],  3),
      CI_high     = round(ci_auc[2],  3),
      CI_type     = ci_type_a,
      stringsAsFactors = FALSE
    )
    cat(sprintf("  %s Cluster %d: AUC diff=%.4f  p=%.4f %s  g=%.3f [%.3f,%.3f]\n",
                label, cp$cluster_id, obs_diff, p_auc, sig_auc,
                g_auc, ci_auc[1], ci_auc[2]))
  }
}

df_auc_results <- if (length(auc_results) > 0)
  dplyr::bind_rows(auc_results) else data.frame()

readr::write_csv(df_auc_results, out("gamm_auc_by_cluster.csv"))


# =============================================================================
# 7. FIGURES
# =============================================================================

base_theme <- theme_classic(base_family = FONT) +
  theme(
    plot.title         = element_text(size = 16, face = "bold", family = FONT),
    axis.title         = element_text(size = 14, face = "bold", family = FONT),
    axis.text          = element_text(size = 13, face = "bold", family = FONT),
    legend.text        = element_text(size = 13, face = "bold", family = FONT),
    legend.title       = element_blank(),
    panel.grid.major.y = element_line(color = "gray92", linewidth = 0.3)
  )


# Figure 1: GAM-smoothed curves per condition with 95% CI ribbon
fig_curves <- ggplot(df_pred,
                     aes(x = Time_pct, y = fit,
                         color = Condition, fill = Condition)) +
  geom_ribbon(aes(ymin = lwr, ymax = upr), alpha = 0.15, color = NA) +
  geom_line(lwd = 1.3) +
  scale_color_manual(values = COND_COLORS) +
  scale_fill_manual(values  = COND_COLORS) +
  labs(
    title = "GAMM-estimated pupillary curves by condition",
    x     = "Reading duration (%)",
    y     = "Pupil dilation (mm, baseline-corrected)"
  ) +
  base_theme

ggsave(out("fig_gam_curves.png"), fig_curves,
       width = 9, height = 5, dpi = 300, bg = "white")
cat("Saved: fig_gam_curves.png\n")


# Figure 2: Pointwise differences + significant cluster shading + Hedges g
comp_dash <- c(
  "SC vs MICT"     = "SC \u2212 MICT",
  "SC vs C-HIIT"   = "SC \u2212 C-HIIT",
  "MICT vs C-HIIT" = "MICT \u2212 C-HIIT"
)

plots_diff <- lapply(c("SC vs MICT", "SC vs C-HIIT", "MICT vs C-HIIT"),
  function(label) {
    df_d      <- df_diff %>% filter(Comparison == label) %>% arrange(Time_pct)
    res       <- cluster_results[[label]]
    sig_cl    <- get_sig_clusters(res)
    lc        <- as.character(comp_colors[label])
    g_data    <- if (nrow(df_bca) > 0)
                   df_bca %>% filter(Comparison == label) else data.frame()

    p <- ggplot(df_d, aes(x = Time_pct)) +
      geom_ribbon(aes(ymin = lwr, ymax = upr), fill = lc, alpha = 0.15) +
      geom_hline(yintercept = 0, color = "gray40", lwd = 0.6, lty = "dashed") +
      geom_line(aes(y = diff), color = lc, lwd = 1.2)

    if (length(sig_cl) > 0) {
      for (sc in sig_cl) {
        g_row <- if (nrow(g_data) > 0)
          g_data[g_data$Cluster == sc$cluster_id, ] else data.frame()
        p_str <- if (sc$p_perm < 0.001) "p < 0.001" else
                   sprintf("p = %.3f", sc$p_perm)
        g_label <- if (nrow(g_row) > 0)
          sprintf("%s\ng = %.2f\n[%.2f, %.2f]",
                  p_str, g_row$Hedges_g, g_row$CI_low, g_row$CI_high)
        else p_str

        p <- p +
          annotate("rect",
                   xmin = sc$start_pct, xmax = sc$end_pct,
                   ymin = -Inf, ymax = Inf,
                   fill = lc, alpha = 0.20) +
          annotate("text",
                   x = (sc$start_pct + sc$end_pct) / 2,
                   y = Inf, vjust = 1.4,
                   label     = g_label,
                   size      = 12 / .pt,
                   family    = FONT, fontface = "bold",
                   color     = "black", lineheight = 1.0)
      }
    }

    p + labs(title = comp_dash[label], x = "Time (%)", y = "Difference (mm)") +
      base_theme +
      theme(plot.title = element_text(color = lc, size = 16, face = "bold"))
  }
)

fig_diff <- patchwork::wrap_plots(plots_diff, ncol = 1) +
  patchwork::plot_annotation(
    title = "Pupillary differences between conditions -- GAMM",
    theme = theme(plot.title = element_text(size = 16, face = "bold",
                                             hjust = 0.5, family = FONT))
  )

ggsave(out("fig_differences.png"), fig_diff,
       width = 10, height = 13, dpi = 300, bg = "white")
cat("Saved: fig_differences.png\n")


# Figure 3: Forest plot of AUC Hedges g per cluster
if (nrow(df_auc_results) > 0) {
  df_auc_plot <- df_auc_results %>%
    mutate(
      Label      = sprintf("%s\n(%s)", Comparison, Window),
      Color      = unname(comp_colors[as.character(Comparison)])
    )

  fig_auc <- ggplot(df_auc_plot,
                    aes(x = Hedges_g, y = Label,
                        xmin = CI_low, xmax = CI_high,
                        color = Comparison)) +
    geom_vline(xintercept = c(-0.8,-0.5,-0.2,0.2,0.5,0.8),
               color = "gray85", lty = "dotted", lwd = 0.5) +
    geom_vline(xintercept = 0, color = "gray40", lwd = 0.8, lty = "dashed") +
    geom_errorbarh(height = 0.25, lwd = 1.0) +
    geom_point(size = 4) +
    geom_text(
      aes(label = sprintf("g=%.2f %s\n[%.2f, %.2f]",
                          Hedges_g, Sig, CI_low, CI_high)),
      hjust = -0.1, size = 11 / .pt,
      family = FONT, fontface = "bold", color = "black", lineheight = 1.0
    ) +
    scale_color_manual(values = comp_colors) +
    facet_wrap(~ Comparison, ncol = 1, scales = "free_y") +
    labs(
      title = "Hedges g -- AUC within significant cluster windows",
      x     = "Hedges g (Cond1 - Cond2)",
      y     = NULL
    ) +
    base_theme +
    theme(strip.text = element_text(size = 13, face = "bold", family = FONT),
          legend.position = "none")

  ggsave(out("fig_auc_hedges_g.png"), fig_auc,
         width = 10,
         height = max(6, nrow(df_auc_plot) * 1.8),
         dpi = 300, bg = "white")
  cat("Saved: fig_auc_hedges_g.png\n")
}


# =============================================================================
# CLUSTER SUMMARY TABLE
# =============================================================================

cluster_rows <- list()
for (label in names(cluster_results)) {
  for (cp in cluster_results[[label]]$cluster_pvals) {
    g_row <- if (nrow(df_bca) > 0)
      df_bca %>% filter(Comparison == label, Cluster == cp$cluster_id)
    else data.frame()
    cluster_rows <- c(cluster_rows, list(data.frame(
      Comparison = label,
      Cluster    = cp$cluster_id,
      Start_pct  = round(cp$start_pct, 1),
      End_pct    = round(cp$end_pct,   1),
      Direction  = cp$direction,
      P_perm     = round(cp$p_perm,    4),
      Sig        = cp$sig,
      Hedges_g   = if (nrow(g_row) > 0) g_row$Hedges_g[1] else NA,
      CI_low     = if (nrow(g_row) > 0) g_row$CI_low[1]   else NA,
      CI_high    = if (nrow(g_row) > 0) g_row$CI_high[1]  else NA,
      stringsAsFactors = FALSE
    )))
  }
}

df_clusters <- if (length(cluster_rows) > 0)
  dplyr::bind_rows(cluster_rows) else data.frame()

readr::write_csv(df_clusters, out("gamm_cluster_results.csv"))

cat("\n", strrep("=", 70), "\n")
cat("FINAL SUMMARY\n")
cat(strrep("=", 70), "\n\n")
cat("GAMM: Pupil ~ Condition + s(Time_pct) +\n")
cat("      s(Time_pct, by=Condition) + s(Participant, re)\n\n")
cat("SIGNIFICANT CLUSTERS:\n")
if (nrow(df_clusters) > 0) print(df_clusters)
cat("\nAUC BY CLUSTER WINDOWS:\n")
if (nrow(df_auc_results) > 0) print(df_auc_results)
cat(sprintf("\nOutputs saved to: %s\n", out_path))
cat("  gamm_summary.txt\n")
cat("  gamm_cluster_results.csv\n")
cat("  gamm_auc_by_cluster.csv\n")
cat("  fig_gam_curves.png\n")
cat("  fig_differences.png\n")
cat("  fig_auc_hedges_g.png\n")
if (RUN_TENSOR) cat("  gamm_tensor_summary.txt\n")
