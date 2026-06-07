# permutation_analysis.R
#
# General-purpose permutation-based statistical analysis
# for pre-post and between-group designs
#
# Methods implemented:
#   - Freedman-Lane permutation (with or without covariate)
#   - Westfall-Young Max-T (FWER control for multiple comparisons)
#   - Hedges g with BCa bootstrap confidence intervals
#   - Noncentral-t CI as fallback for very small n
#   - Optional spaghetti + violin figure per outcome
#
# When to use each method:
#
#   Standard sign-flip permutation:
#     Use when there is no covariate to control for. Tests the null
#     hypothesis that the mean pre-post difference is zero by randomly
#     flipping the sign of each participant's difference.
#
#   Freedman-Lane permutation:
#     Use when a covariate (e.g., medication status, sex, age) must be
#     controlled. Residualizes the outcome on the covariate under H0,
#     then permutes the residuals. This correctly tests the covariate-
#     adjusted effect while preserving the covariate's relationship
#     with the outcome.
#
#   Westfall-Young Max-T:
#     Use when comparing multiple outcomes or multiple groups simultaneously.
#     Unlike Bonferroni or BH-FDR, Max-T uses joint permutation: one sign
#     per participant, applied simultaneously to all comparisons. This
#     preserves the correlation structure between test statistics and is
#     more powerful than Bonferroni when outcomes are correlated.
#
#   BCa bootstrap for Hedges g:
#     The bias-corrected and accelerated (BCa) bootstrap corrects for
#     both bias and skewness in the bootstrap distribution, making it
#     more accurate than percentile bootstrap, especially for small n.
#     When n < 10, the bootstrap resampling space may be too small for
#     stable BCa estimates; in that case, use the noncentral-t CI.
#
# Script structure:
#   1. Configuration (paths, outcomes, covariates)
#   2. Helper functions (permutation, bootstrap, effect size)
#   3. Analysis functions (pre-post, between-group)
#   4. Figure functions (optional)
#   5. Main pipeline
#
# To adapt for a new dataset:
#   Edit the DATA CONFIGURATION section. The analysis functions
#   do not need to be modified.
#
# Author: Ricardo Martinez-Flores
# Contact: ricardo.antonio.martinezf@gmail.com
# License: MIT

library(dplyr)
library(tidyr)
library(ggplot2)
library(patchwork)
library(boot)

# Word table output (optional -- comment out if not needed)
use_word_table <- TRUE
if (use_word_table) {
  library(officer)
  library(flextable)
}

theme_set(theme_bw(base_size = 14, base_family = "serif"))
set.seed(42)


# =============================================================================
# DATA CONFIGURATION
# Edit this section to adapt the script to a new dataset.
# =============================================================================

N_PERM <- 10000   # permutations for all tests
N_BOOT <- 10000   # bootstrap resamples for BCa CI

output_dir <- "~/Desktop/permutation_outputs"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Path to data file (CSV or XLSX)
data_path <- "~/Desktop/your_data.csv"

# Column names in the data file
col_participant <- "participant"
col_group       <- "group"        # set to NULL if no between-group comparison
col_covariate   <- "covariate"    # set to NULL if no covariate

# Outcome definitions: list of named lists with pre and post column names.
# Also specify improvement_direction ("increase" or "decrease") for labeling.
outcomes <- list(
  outcome_A = list(pre = "var_A_pre", post = "var_A_post",
                   direction = "increase"),
  outcome_B = list(pre = "var_B_pre", post = "var_B_post",
                   direction = "decrease")
)

# Group labels (only used if col_group is not NULL)
group_labels <- c("Group1", "Group2")

# Which outcomes to visualize (spaghetti + violin figure)
# Set to NULL to skip figures, or to names(outcomes) for all.
figure_outcomes <- names(outcomes)

# Which outcomes to include in Word table
# Set to NULL to skip Word table.
table_outcomes <- names(outcomes)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# Hedges g (paired, within-person)
hedges_g_paired <- function(pre, post) {
  n    <- length(pre)
  diff <- post - pre
  d    <- mean(diff) / sd(diff)
  J    <- 1 - (3 / (4 * (n - 1) - 1))
  d * J
}

# Hedges g (independent samples, for between-group comparison of deltas)
hedges_g_indep <- function(x1, x2) {
  n1 <- length(x1); n2 <- length(x2)
  if (n1 < 2 || n2 < 2) return(NA_real_)
  md <- mean(x1, na.rm = TRUE) - mean(x2, na.rm = TRUE)
  sp <- sqrt(((n1-1)*var(x1, na.rm=TRUE) + (n2-1)*var(x2, na.rm=TRUE)) / (n1+n2-2))
  if (sp == 0) return(NA_real_)
  J  <- 1 - 3 / (4*(n1+n2-2) - 1)
  (md / sp) * J
}

# Freedman-Lane permutation -- pre-post (within group)
# Tests H0: mean(delta) = 0, controlling for covariate
# When no covariate or covariate has only one level: falls back to sign-flip.
freedman_lane_prepost <- function(delta, covariate = NULL, n_perm = N_PERM) {
  n <- length(delta)

  no_cov <- is.null(covariate) ||
            all(is.na(covariate)) ||
            length(unique(covariate[!is.na(covariate)])) < 2

  if (no_cov) {
    # Standard sign-flip permutation (no covariate)
    t_obs  <- mean(delta) / (sd(delta) / sqrt(n))
    t_perm <- replicate(n_perm, {
      d_s <- sample(c(-1, 1), n, replace = TRUE) * delta
      mean(d_s) / (sd(d_s) / sqrt(n))
    })
    p_value <- mean(abs(t_perm) >= abs(t_obs))
    return(list(t_obs = t_obs, p_value = p_value, method = "sign-flip"))
  }

  # Freedman-Lane: center covariate (intercept = adjusted mean)
  cov_c  <- covariate - mean(covariate, na.rm = TRUE)
  df_mod <- data.frame(delta = delta, cov_c = cov_c)

  m_full <- lm(delta ~ cov_c, data = df_mod)
  t_obs  <- summary(m_full)$coefficients["(Intercept)", "t value"]

  # Reduced model forces no mean change (H0), keeps covariate relationship
  m_red    <- lm(delta ~ 0 + cov_c, data = df_mod)
  res_h0   <- residuals(m_red)
  fit_h0   <- fitted(m_red)

  t_perm <- replicate(n_perm, {
    signs  <- sample(c(-1L, 1L), n, replace = TRUE)
    y_star <- fit_h0 + signs * res_h0
    df_mod$ds <- y_star
    summary(lm(ds ~ cov_c, data = df_mod))$coefficients["(Intercept)", "t value"]
  })

  p_value <- mean(abs(t_perm) >= abs(t_obs))
  list(t_obs = t_obs, p_value = p_value, method = "Freedman-Lane")
}

# Freedman-Lane permutation -- between groups
# Tests H0: no group difference in delta, controlling for covariate
freedman_lane_between <- function(delta_1, delta_2,
                                  cov_1 = NULL, cov_2 = NULL,
                                  n_perm = N_PERM) {
  delta <- c(delta_1, delta_2)
  group <- c(rep(1L, length(delta_1)), rep(0L, length(delta_2)))
  n     <- length(delta)

  has_cov <- !is.null(cov_1) && !is.null(cov_2) &&
             !all(is.na(c(cov_1, cov_2))) &&
             length(unique(c(cov_1, cov_2)[!is.na(c(cov_1, cov_2))])) >= 2

  if (has_cov) {
    cov <- as.numeric(!is.na(c(cov_1, cov_2)) & c(cov_1, cov_2) == 1)
    df_mod <- data.frame(delta = delta, group = group, cov = cov)
    m_full <- lm(delta ~ group + cov, data = df_mod)
    m_red  <- lm(delta ~ cov,         data = df_mod)
  } else {
    df_mod <- data.frame(delta = delta, group = group)
    m_full <- lm(delta ~ group, data = df_mod)
    m_red  <- lm(delta ~ 1,     data = df_mod)
  }

  t_obs <- tryCatch(
    summary(m_full)$coefficients["group", "t value"],
    error = function(e) NA_real_
  )
  if (is.na(t_obs)) return(list(t_obs = NA, p_value = NA))

  res_h0 <- residuals(m_red)
  fit_h0 <- fitted(m_red)

  # Standard Freedman-Lane: permute residuals
  t_perm <- replicate(n_perm, {
    perm <- sample(n)
    df_mod$ds <- fit_h0 + res_h0[perm]
    m_b <- if (has_cov) lm(ds ~ group + cov, data = df_mod)
           else         lm(ds ~ group,         data = df_mod)
    tryCatch(summary(m_b)$coefficients["group", "t value"], error = function(e) NA)
  })

  valid <- t_perm[!is.na(t_perm)]
  list(t_obs = t_obs, p_value = mean(abs(valid) >= abs(t_obs)))
}

# Westfall-Young Max-T (FWER control for multiple outcomes)
# Joint permutation: one sign per participant, applied to all outcomes
# simultaneously. This preserves the correlation structure between tests.
max_t_prepost <- function(delta_matrix, n_perm = N_PERM) {
  # delta_matrix: participants x outcomes
  n_part <- nrow(delta_matrix)
  n_out  <- ncol(delta_matrix)

  # Observed t-statistics per outcome
  t_obs <- apply(delta_matrix, 2, function(d) {
    s <- sd(d, na.rm = TRUE)
    if (s == 0) return(0)
    mean(d, na.rm = TRUE) / (s / sqrt(sum(!is.na(d))))
  })

  # Permutation distribution of max|t|
  max_t_dist <- replicate(n_perm, {
    signs <- sample(c(-1, 1), n_part, replace = TRUE)
    t_perm_vec <- apply(delta_matrix, 2, function(d) {
      d_s <- signs * d
      s <- sd(d_s, na.rm = TRUE)
      if (s == 0) return(0)
      mean(d_s, na.rm = TRUE) / (s / sqrt(sum(!is.na(d_s))))
    })
    max(abs(t_perm_vec))
  })

  # p-value: proportion of permutations where max|t| >= |t_obs_k|
  p_maxt <- sapply(t_obs, function(t_k) {
    mean(max_t_dist >= abs(t_k))
  })

  list(t_obs = t_obs, p_maxt = p_maxt)
}

# BCa bootstrap for Hedges g (paired)
bca_hedges_paired <- function(pre, post, n_boot = N_BOOT, conf = 0.95) {
  g_obs  <- hedges_g_paired(pre, post)
  df_p   <- data.frame(pre = pre, post = post)
  g_stat <- function(data, idx) hedges_g_paired(data$pre[idx], data$post[idx])
  bo     <- boot(data = df_p, statistic = g_stat, R = n_boot)
  ci     <- tryCatch(boot.ci(bo, conf = conf, type = "bca"),
                     error = function(e) NULL)
  if (!is.null(ci) && !is.null(ci$bca)) {
    list(g = g_obs, ci_lower = ci$bca[4], ci_upper = ci$bca[5])
  } else {
    # Fallback: normal approximation
    se <- sqrt(1/length(pre) + g_obs^2 / (2*(length(pre)-1)))
    z  <- qnorm(1 - (1 - conf)/2)
    list(g = g_obs, ci_lower = g_obs - z*se, ci_upper = g_obs + z*se)
  }
}

# BCa bootstrap for Hedges g (independent samples on deltas)
bca_hedges_indep <- function(delta_1, delta_2, n_boot = N_BOOT, conf = 0.95) {
  g_obs <- hedges_g_indep(delta_1, delta_2)
  if (is.na(g_obs)) return(list(g = NA, ci_lower = NA, ci_upper = NA))

  df_c <- data.frame(
    delta = c(delta_1, delta_2),
    group = c(rep(1, length(delta_1)), rep(0, length(delta_2)))
  )
  g_stat <- function(data, idx) {
    d <- data[idx, ]
    d1 <- d$delta[d$group == 1]
    d2 <- d$delta[d$group == 0]
    if (length(d1) < 2 || length(d2) < 2) return(NA_real_)
    hedges_g_indep(d1, d2)
  }
  strata <- as.integer(factor(df_c$group))
  bo <- tryCatch(boot(data = df_c, statistic = g_stat, R = n_boot, strata = strata),
                 error = function(e) NULL)
  if (is.null(bo)) return(list(g = g_obs, ci_lower = NA, ci_upper = NA))
  ci <- tryCatch(boot.ci(bo, conf = conf, type = "bca"), error = function(e) NULL)
  if (!is.null(ci) && !is.null(ci$bca)) {
    list(g = g_obs, ci_lower = ci$bca[4], ci_upper = ci$bca[5])
  } else {
    list(g = g_obs, ci_lower = NA, ci_upper = NA)
  }
}

# Noncentral-t CI for Hedges g (use when n < 10)
# Exact CI derived from the noncentral t distribution.
# More stable than BCa when the resampling space is very small.
nct_ci_hedges <- function(pre, post, conf = 0.95) {
  n      <- length(pre)
  diff   <- post - pre
  sd_d   <- sd(diff)
  if (sd_d == 0) return(list(g = NA, ci_lower = NA, ci_upper = NA))

  d_z    <- mean(diff) / sd_d
  t_stat <- d_z * sqrt(n)
  df     <- n - 1
  J      <- 1 - 3 / (4*df - 1)
  g      <- d_z * J
  alpha  <- 1 - conf

  tryCatch({
    lo_eq <- function(ncp) pt(t_stat, df, ncp) - (1 - alpha/2)
    hi_eq <- function(ncp) pt(t_stat, df, ncp) - (alpha/2)
    lo_ncp <- uniroot(lo_eq, c(-25, 25))$root
    hi_ncp <- uniroot(hi_eq, c(-25, 25))$root
    list(g = g, ci_lower = lo_ncp/sqrt(n)*J, ci_upper = hi_ncp/sqrt(n)*J)
  }, error = function(e) {
    se <- sqrt(1/n + g^2/(2*df))
    z  <- qnorm(1 - alpha/2)
    list(g = g, ci_lower = g - z*se, ci_upper = g + z*se)
  })
}

label_effect_size <- function(g) {
  ga <- abs(g)
  if (is.na(ga)) return("NA")
  if (ga < 0.2) "negligible"
  else if (ga < 0.5) "small"
  else if (ga < 0.8) "medium"
  else "large"
}


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

# Pre-post analysis for one outcome, across subgroups
analyze_prepost <- function(data_out, outcome_name,
                            col_group = NULL,
                            col_covariate = NULL,
                            group_labels = NULL,
                            n_perm = N_PERM, n_boot = N_BOOT) {

  cat(sprintf("\n%s\n# %s\n%s\n\n",
              strrep("#", 70), outcome_name, strrep("#", 70)))

  results <- list()

  # Define subgroups to analyze
  if (!is.null(col_group) && col_group %in% names(data_out)) {
    group_vals  <- sort(unique(data_out[[col_group]]))
    subgroups <- c(list(list(label = "Overall", data = data_out)),
                   lapply(group_vals, function(gv) {
                     lbl <- if (!is.null(group_labels) &&
                                 !is.na(group_labels[gv+1]))
                              group_labels[gv+1]
                            else as.character(gv)
                     list(label = lbl, data = data_out %>% filter(.data[[col_group]] == gv))
                   }))
  } else {
    subgroups <- list(list(label = "Overall", data = data_out))
  }

  for (sg in subgroups) {
    d   <- sg$data
    lbl <- sg$label
    n   <- nrow(d)

    cat(sprintf("%s (n=%d)\n%s\n", lbl, n, strrep("-", 50)))

    if (n < 5) {
      cat("Insufficient data (n < 5) -- skipped\n\n")
      results[[lbl]] <- data.frame(
        outcome = outcome_name, group = lbl, n = n,
        mean_pre = NA, mean_post = NA, mean_diff = NA, sd_diff = NA,
        p_perm = NA, p_maxt = NA, hedges_g = NA,
        ci_lower = NA, ci_upper = NA, sig = "")
      next
    }

    mean_pre  <- mean(d$pre,  na.rm = TRUE)
    mean_post <- mean(d$post, na.rm = TRUE)
    diff      <- d$post - d$pre
    mean_diff <- mean(diff)
    sd_diff   <- sd(diff)

    cat(sprintf("Pre:  %.4f (SD=%.4f)\n", mean_pre,  sd(d$pre,  na.rm=TRUE)))
    cat(sprintf("Post: %.4f (SD=%.4f)\n", mean_post, sd(d$post, na.rm=TRUE)))
    cat(sprintf("Diff: %.4f (SD=%.4f)\n", mean_diff, sd_diff))

    cov_vec <- if (!is.null(col_covariate) && col_covariate %in% names(d))
                 d[[col_covariate]] else NULL

    perm_res <- freedman_lane_prepost(diff, cov_vec, n_perm)
    cat(sprintf("Permutation (%s): p=%.4f%s\n",
                perm_res$method, perm_res$p_value,
                ifelse(perm_res$p_value < 0.05, " *", "")))

    # BCa bootstrap (use nct if n < 10)
    g_res <- if (n >= 10) bca_hedges_paired(d$pre, d$post, n_boot)
             else          nct_ci_hedges(d$pre, d$post)

    cat(sprintf("Hedges g=%.3f [%.3f, %.3f] (%s)\n\n",
                g_res$g, g_res$ci_lower, g_res$ci_upper,
                label_effect_size(g_res$g)))

    results[[lbl]] <- data.frame(
      outcome   = outcome_name, group = lbl, n = n,
      mean_pre  = mean_pre, mean_post = mean_post,
      mean_diff = mean_diff, sd_diff = sd_diff,
      p_perm    = perm_res$p_value, p_maxt = NA,
      hedges_g  = g_res$g,
      ci_lower  = g_res$ci_lower, ci_upper = g_res$ci_upper,
      sig       = ifelse(perm_res$p_value < 0.05, "*", ""))
  }

  bind_rows(results)
}

# Between-group analysis for one outcome
analyze_between <- function(data_out, outcome_name,
                            col_group, group_vals,
                            col_covariate = NULL,
                            n_perm = N_PERM, n_boot = N_BOOT) {

  if (!col_group %in% names(data_out)) return(NULL)

  d1 <- data_out %>% filter(.data[[col_group]] == group_vals[1])
  d2 <- data_out %>% filter(.data[[col_group]] == group_vals[2])

  if (nrow(d1) < 3 || nrow(d2) < 3) return(NULL)

  delta_1 <- d1$post - d1$pre
  delta_2 <- d2$post - d2$pre

  cov_1 <- if (!is.null(col_covariate) && col_covariate %in% names(d1))
              d1[[col_covariate]] else NULL
  cov_2 <- if (!is.null(col_covariate) && col_covariate %in% names(d2))
              d2[[col_covariate]] else NULL

  fl_res  <- freedman_lane_between(delta_1, delta_2, cov_1, cov_2, n_perm)
  bca_res <- bca_hedges_indep(delta_1, delta_2, n_boot)

  cat(sprintf("  Between-group: g=%.3f [%.3f, %.3f], p=%.4f%s\n",
              bca_res$g, bca_res$ci_lower, bca_res$ci_upper,
              fl_res$p_value,
              ifelse(!is.na(fl_res$p_value) && fl_res$p_value < 0.05, " *", "")))

  data.frame(
    outcome   = outcome_name,
    between_g = bca_res$g, between_ci_lo = bca_res$ci_lower,
    between_ci_hi = bca_res$ci_upper, between_p = fl_res$p_value,
    between_sig = ifelse(!is.na(fl_res$p_value) & fl_res$p_value < 0.05, "*", "")
  )
}


# =============================================================================
# FIGURE FUNCTION (spaghetti + violin)
# =============================================================================

plot_prepost <- function(data_out, result_row, outcome_name, output_dir,
                         col_group = NULL, group_labels_map = NULL) {

  group_color_map <- c(Overall = "black",
                       Group1  = "#A23B72",
                       Group2  = "#2E86AB")

  if (!is.null(col_group) && col_group %in% names(data_out)) {
    group_vals  <- sort(unique(data_out[[col_group]]))
    subgroup_list <- c(
      list(list(label = "Overall", data = data_out)),
      lapply(group_vals, function(gv) {
        lbl <- if (!is.null(group_labels_map)) group_labels_map[as.character(gv)]
               else as.character(gv)
        list(label = lbl, data = data_out %>% filter(.data[[col_group]] == gv))
      })
    )
  } else {
    subgroup_list <- list(list(label = "Overall", data = data_out))
  }

  plot_list <- list()

  for (sg in subgroup_list) {
    d    <- sg$data %>% mutate(delta = post - pre, id = row_number())
    lbl  <- sg$label
    col  <- group_color_map[lbl]
    if (is.na(col)) col <- "steelblue"

    stat_row <- result_row %>% filter(group == lbl)
    if (nrow(stat_row) == 0) next

    d_long <- d %>%
      pivot_longer(cols = c(pre, post), names_to = "timepoint", values_to = "value") %>%
      mutate(timepoint = factor(timepoint, levels = c("pre", "post"),
                                labels = c("Pre", "Post")))

    y_range <- range(d_long$value, na.rm = TRUE)
    y_span  <- diff(y_range)
    y_lo    <- y_range[1] - y_span * 0.08
    y_hi    <- y_range[2] + y_span * 0.15

    anno_label <- sprintf("p=%.3f\ng=%.2f\n[%.2f, %.2f]",
                          stat_row$p_perm[1],
                          stat_row$hedges_g[1],
                          stat_row$ci_lower[1],
                          stat_row$ci_upper[1])

    d_vals  <- d$delta
    d_range <- range(d_vals, na.rm = TRUE)
    d_span  <- max(diff(d_range), 1e-6)
    bw      <- bw.nrd0(d_vals) * 1.2 * 3
    vl_lo   <- d_range[1] - bw
    vl_hi   <- d_range[2] + bw + d_span * 0.55
    y_anno  <- vl_hi - d_span * 0.02

    plt_theme <- theme_bw(base_size = 15, base_family = "serif") +
      theme(
        plot.title         = element_text(face = "bold", hjust = 0.5, size = 16),
        axis.title         = element_text(face = "bold", size = 14),
        axis.text          = element_text(face = "bold", size = 13),
        panel.grid.minor   = element_blank(),
        panel.grid.major.x = element_blank(),
        legend.position    = "none"
      )

    p_spag <- ggplot(d_long, aes(x = timepoint, y = value, group = id)) +
      geom_line(color = col, alpha = 0.55, linewidth = 0.7) +
      geom_point(color = col, alpha = 0.80, size = 2.5) +
      scale_y_continuous(limits = c(y_lo, y_hi), expand = c(0, 0)) +
      scale_x_discrete(expand = expansion(add = 0.35)) +
      labs(x = NULL, y = if (lbl == "Overall") outcome_name else NULL,
           title = lbl) +
      plt_theme

    p_violin <- ggplot(d, aes(x = 1, y = delta)) +
      geom_hline(yintercept = 0, linetype = "dashed", color = "gray35",
                 linewidth = 0.8) +
      geom_violin(fill = col, alpha = 0.28, color = col, linewidth = 1.1,
                  trim = FALSE, scale = "width", adjust = 1.2) +
      annotate("point", x = 1, y = mean(d_vals, na.rm = TRUE),
               size = 6, color = col, shape = 18) +
      annotate("text",  x = 1, y = y_anno, label = anno_label,
               size = 5, family = "serif", fontface = "bold",
               hjust = 0.5, vjust = 1, lineheight = 0.95) +
      scale_x_continuous(limits = c(0.55, 1.45), breaks = 1, labels = "\u0394") +
      scale_y_continuous(limits = c(vl_lo, vl_hi), expand = c(0, 0)) +
      labs(x = "Post - Pre", y = NULL) +
      plt_theme +
      theme(axis.text.y  = element_blank(),
            axis.ticks.y = element_blank())

    plot_list[[lbl]] <- p_spag + p_violin + plot_layout(widths = c(2.8, 1))
  }

  if (length(plot_list) == 0) return(invisible(NULL))

  p_final <- wrap_plots(plot_list, nrow = 1) +
    plot_annotation(
      title = outcome_name,
      theme = theme(plot.title = element_text(face = "bold", size = 18,
                                              hjust = 0.5, family = "serif"))
    )

  out_name <- file.path(output_dir,
                        sprintf("%s_figure.png",
                                gsub("[ /]", "_", outcome_name)))
  ggsave(out_name, plot = p_final,
         width = 6 * length(plot_list), height = 6,
         dpi = 300, bg = "white")
  cat(sprintf("Figure saved: %s\n", basename(out_name)))
  invisible(p_final)
}


# =============================================================================
# WORD TABLE (optional)
# =============================================================================

make_word_table <- function(table_data, output_dir) {
  fmt_mean_sd <- function(m, s) {
    mapply(function(m_, s_) sprintf("%.2f +/- %.2f", m_, s_), m, s)
  }
  fmt_g_ci <- function(g, lo, hi) {
    mapply(function(g_, lo_, hi_) {
      if (is.na(g_)) return("-")
      sprintf("%.2f [%.2f, %.2f]", g_, lo_, hi_)
    }, g, lo, hi)
  }
  fmt_p <- function(p) {
    sapply(p, function(x) {
      if (is.na(x))  return("-")
      if (x < 0.001) return("<.001")
      formatC(x, digits = 3, format = "f")
    })
  }

  overall <- table_data %>% filter(group == "Overall")
  tbl_df  <- overall %>%
    mutate(
      col1 = outcome,
      col2 = fmt_mean_sd(mean_pre, sd_diff),
      col3 = fmt_g_ci(hedges_g, ci_lower, ci_upper),
      col4 = fmt_p(p_perm)
    ) %>%
    select(col1, col2, col3, col4)

  ft <- flextable(tbl_df) %>%
    set_header_labels(col1 = "Outcome", col2 = "M +/- SD",
                      col3 = "g [95% CI]", col4 = "p") %>%
    theme_booktabs() %>%
    bold(part = "header") %>%
    align(align = "center", part = "all") %>%
    align(j = 1, align = "left", part = "all") %>%
    fontsize(size = 10, part = "all") %>%
    font(fontname = "Times New Roman", part = "all") %>%
    autofit()

  doc <- read_docx() %>%
    body_add_par("Statistical Results", style = "heading 1") %>%
    body_add_par(
      paste("M +/- SD refers to pre-intervention values.",
            "g = Hedges g (bias-corrected).",
            "95% CI from BCa bootstrap (nct if n < 10).",
            "p from Freedman-Lane permutation (Nperm=10000)."),
      style = "Normal") %>%
    body_add_flextable(ft)

  out_path <- file.path(output_dir, "results_table.docx")
  print(doc, target = out_path)
  cat(sprintf("Word table saved: %s\n", basename(out_path)))
}


# =============================================================================
# MAIN PIPELINE
# =============================================================================

cat(strrep("=", 70), "\n")
cat("PERMUTATION ANALYSIS PIPELINE\n")
cat(strrep("=", 70), "\n")

# Load data
if (grepl("\\.xlsx$", data_path)) {
  library(readxl)
  df_raw <- read_excel(data_path)
} else {
  df_raw <- read.csv(data_path)
}

# Fix decimal comma if needed
char_cols <- names(df_raw)[sapply(df_raw, is.character)]
for (col in char_cols) {
  converted <- suppressWarnings(as.numeric(gsub(",", ".", df_raw[[col]])))
  if (sum(!is.na(converted)) > sum(!is.na(df_raw[[col]]))) {
    df_raw[[col]] <- converted
  }
}

cat(sprintf("Data loaded: %d rows\n\n", nrow(df_raw)))

# Main loop
all_prepost <- data.frame()
all_between <- data.frame()

for (out_name in names(outcomes)) {
  out_def <- outcomes[[out_name]]

  data_out <- df_raw %>%
    select(
      pre  = all_of(out_def$pre),
      post = all_of(out_def$post),
      any_of(c(col_participant, col_group, col_covariate))
    ) %>%
    filter(!is.na(pre), !is.na(post))

  if (!is.null(col_covariate) && col_covariate %in% names(data_out)) {
    cov_col <- col_covariate
  } else {
    cov_col <- NULL
  }

  res_pp <- analyze_prepost(data_out, out_name,
                            col_group     = col_group,
                            col_covariate = cov_col)
  all_prepost <- bind_rows(all_prepost, res_pp)

  if (!is.null(col_group) && col_group %in% names(data_out)) {
    group_vals_present <- sort(unique(data_out[[col_group]]))
    if (length(group_vals_present) >= 2) {
      cat(sprintf("  Between-group: %s\n", out_name))
      res_bt <- analyze_between(data_out, out_name,
                                col_group     = col_group,
                                group_vals    = group_vals_present,
                                col_covariate = cov_col)
      if (!is.null(res_bt)) all_between <- bind_rows(all_between, res_bt)
    }
  }

  if (out_name %in% figure_outcomes) {
    group_labels_map <- if (!is.null(col_group) && !is.null(group_labels)) {
      gv <- sort(unique(data_out[[col_group]]))
      setNames(group_labels, as.character(gv))
    } else NULL

    plot_prepost(data_out, res_pp, out_name, output_dir,
                 col_group = col_group,
                 group_labels_map = group_labels_map)
  }
}

# Max-T across all outcomes (for FWER control)
if (nrow(all_prepost) > 0 && length(outcomes) > 1) {
  cat(strrep("=", 70), "\n")
  cat("MAX-T CORRECTION (all outcomes, overall group)\n")
  cat(strrep("=", 70), "\n")

  overall_rows <- all_prepost %>% filter(group == "Overall")
  out_names_ok <- overall_rows$outcome

  delta_matrix <- sapply(out_names_ok, function(on) {
    out_def  <- outcomes[[on]]
    data_out <- df_raw %>%
      select(pre  = all_of(out_def$pre),
             post = all_of(out_def$post)) %>%
      filter(!is.na(pre), !is.na(post))
    data_out$post - data_out$pre
  })

  if (!is.null(dim(delta_matrix)) && nrow(delta_matrix) >= 5) {
    maxt_res <- max_t_prepost(delta_matrix, N_PERM)
    all_prepost$p_maxt <- NA
    for (i in seq_along(out_names_ok)) {
      all_prepost$p_maxt[all_prepost$outcome == out_names_ok[i] &
                         all_prepost$group == "Overall"] <- maxt_res$p_maxt[i]
    }
    cat("Max-T p-values (overall):\n")
    for (i in seq_along(out_names_ok)) {
      cat(sprintf("  %s: p_MaxT=%.4f\n", out_names_ok[i], maxt_res$p_maxt[i]))
    }
  }
}

# Save results
write.csv(all_prepost, file.path(output_dir, "prepost_results.csv"), row.names = FALSE)
if (nrow(all_between) > 0) {
  write.csv(all_between, file.path(output_dir, "between_results.csv"), row.names = FALSE)
}

# Word table
if (use_word_table && !is.null(table_outcomes) && nrow(all_prepost) > 0) {
  table_data <- all_prepost %>% filter(outcome %in% table_outcomes)
  if (nrow(table_data) > 0) make_word_table(table_data, output_dir)
}

# Console summary
cat(strrep("=", 70), "\n")
cat("SUMMARY\n")
cat(strrep("=", 70), "\n")
print(all_prepost %>%
        filter(group == "Overall") %>%
        select(outcome, n, mean_diff, p_perm, p_maxt, hedges_g, ci_lower, ci_upper) %>%
        mutate(across(where(is.numeric), round, 4)))

cat(sprintf("\nOutputs saved to: %s\n", output_dir))
