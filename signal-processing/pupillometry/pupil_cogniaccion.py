# pupil_cogniaccion.py
#
# Pupillometry preprocessing pipeline -- Cogni-Action project
# Pupillary dynamics during reading comprehension in adolescents
#
# Eye-tracker: Tobii (300 Hz)
# Signal: pupil diameter in pixels, converted via calibration to mm
# Validity: Tobii validity codes (0 = valid, >0 = invalid)
#
# Paradigm: 3 reading texts counterbalanced across participants.
# Texts are matched for syntactic complexity and word count.
# The analysis focuses on the temporal trajectory of pupil dilation
# across the reading epoch, not a single summary metric like AUC.
# This reflects the hypothesis that reading difficulty modulates
# the dynamics of cognitive load over time, not just its peak.
#
# Key difference from SAPIENS:
# There is no externally defined trial structure. The reading epoch
# runs from text onset to button-press (participant indicates completion).
# Trial duration varies across participants and texts, so normalization
# to percentage time is essential for cross-participant alignment.
# Within-participant averaging across 3 texts is done before group averaging
# (hierarchical averaging) to avoid giving more weight to participants
# with more valid samples.
#
# Pipeline overview:
#   1. Validity flag (Tobii codes) -> NaN
#   2. Physiological range filter [2-8 mm] + derivative filter
#   3. Expand invalid gaps by +/- EXPAND_MS around each gap
#   4. Binocular averaging
#   5. Gaussian smoothing (sigma = SIGMA_MS)
#   6. Baseline correction (median of pre-reading fixation window)
#   7. Normalize to percentage of reading duration
#   8. Hierarchical averaging: trial -> participant -> group
#
# Why expand gaps before interpolation:
# Tobii validity codes do not capture the full contamination window
# around a blink. The signal typically begins recovering for some time
# after the validity flag returns to 0. Expanding the invalid mask
# removes that recovery artifact from the interpolation source.
#
# Author: Ricardo Martinez-Flores
# Contact: ricardo.antonio.martinezf@gmail.com
# License: MIT

import pandas as pd
import numpy as np
import os
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Global parameters
# ---------------------------------------------------------------------------

FS           = 300        # Tobii sampling rate (Hz)
DT_MS        = 1000.0 / FS

# Gap expansion: blink contamination extends beyond the validity flag window.
# The post-blink recovery period varies by tracker and participant.
# Inspect your own data to determine an appropriate value.
EXPAND_MS    = None   # set by user (ms)

# Gaussian smoothing: sigma in ms, converted to samples.
# Choose based on the temporal resolution you need and the noise level.
# Larger sigma removes more noise but blurs temporal features.
SIGMA_MS     = None   # set by user (ms)

# Derived values -- computed automatically once you set EXPAND_MS and SIGMA_MS above.
# Do not edit these directly.
EXPAND_SAMP  = round(EXPAND_MS / 1000 * FS) if EXPAND_MS is not None else None
SIGMA_SAMP   = SIGMA_MS / 1000 * FS         if SIGMA_MS  is not None else None

# Baseline window: pre-reading fixation cross duration (ms).
# Adjust to match your experimental timing.
BASELINE_MS  = 200

# Physiological pupil range (mm). Values outside are artifacts.
PUPIL_MIN_MM = 2.0
PUPIL_MAX_MM = 8.0

# Maximum rate of change between samples (mm/sample).
# Genuine pupil responses are slower than 1 mm per 3.3 ms.
MAX_DERIV_MM = 1.0

# Maximum allowed interpolation before discarding an eye (%).
MAX_INTERP_PCT = 40.0

# IQR multiplier for post-hoc curve-level outlier detection.
# Curves whose mean falls outside Q1 - k*IQR or Q3 + k*IQR are excluded.
# See Leys et al. (2013) for guidance on choosing k.
IQR_THRESH = None   # set by user

# Common grid for cross-participant alignment
N_GRID = 1000

# Column names (Tobii export format)
TS_COL = 'Recording timestamp'
PUP_L  = 'Pupil diameter left'
PUP_R  = 'Pupil diameter right'
VAL_L  = 'Validity left'
VAL_R  = 'Validity right'

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14


# ---------------------------------------------------------------------------
# Signal quality functions
# ---------------------------------------------------------------------------

def get_invalid_mask(validity_series):
    return validity_series.astype(str).str.strip().str.lower() == 'invalid'


def expand_mask(bool_array, n):
    # Expand a boolean mask by n samples in each direction.
    # Uses convolution rather than dilation to avoid scipy dependency on ndimage.
    kernel   = np.ones(2 * n + 1)
    expanded = np.convolve(bool_array.astype(float), kernel, mode='same') > 0
    return expanded


def apply_physiological_qc(p):
    p = p.copy()
    # Range filter
    p[(p < PUPIL_MIN_MM) | (p > PUPIL_MAX_MM)] = np.nan
    # Derivative filter: only within valid segments
    valid = ~np.isnan(p)
    deriv = np.abs(np.concatenate([[0.0], np.diff(p)]))
    p[valid & (deriv > MAX_DERIV_MM)] = np.nan
    return p


def interpolate_linear(t, p):
    valid_idx = np.where(~np.isnan(p))[0]
    if len(valid_idx) == 0:
        return np.full(len(p), np.nan)

    t_v = t[valid_idx]
    p_v = p[valid_idx]
    _, uid = np.unique(t_v, return_index=True)
    t_v, p_v = t_v[uid], p_v[uid]

    if len(t_v) < 2:
        return np.full(len(p), p_v[0] if len(p_v) == 1 else np.nan)

    return np.interp(t, t_v, p_v)


# ---------------------------------------------------------------------------
# Trial-level preprocessing
# ---------------------------------------------------------------------------

def preprocess_trial(trial_df, participant='', text_id='', verbose=False):
    t = trial_df[TS_COL].values.copy()
    t = t - t[0]

    results = {}

    for eye, pup_col, val_col in [('left', PUP_L, VAL_L), ('right', PUP_R, VAL_R)]:
        p   = trial_df[pup_col].values.copy().astype(float)
        inv = get_invalid_mask(trial_df[val_col]).values

        p[inv] = np.nan
        p = apply_physiological_qc(p)

        valid_idx = np.where(~np.isnan(p))[0]
        if len(valid_idx) == 0:
            results[eye]          = None
            results[f'pct_{eye}'] = 100.0
            results[f't_{eye}']   = t
            continue

        # Remove leading invalid samples before computing % interpolated.
        # Participant may not have been tracked at epoch start.
        first_valid = valid_idx[0]
        if first_valid > 0:
            p     = p[first_valid:]
            t_eye = t[first_valid:]
        else:
            t_eye = t.copy()

        pct_interp = np.sum(np.isnan(p)) / len(p) * 100

        if pct_interp > MAX_INTERP_PCT:
            if verbose:
                print(f"    [{eye}] discarded: {pct_interp:.1f}% missing")
            results[eye]          = None
            results[f'pct_{eye}'] = pct_interp
            results[f't_{eye}']   = t_eye
            continue

        # Expand gaps, then interpolate
        nan_mask_exp = expand_mask(np.isnan(p), EXPAND_SAMP)
        p[nan_mask_exp] = np.nan
        p_interp = interpolate_linear(t_eye, p)

        results[eye]          = p_interp
        results[f'pct_{eye}'] = pct_interp
        results[f't_{eye}']   = t_eye

    # Binocular average
    left   = results.get('left',  None)
    right  = results.get('right', None)
    t_left = results.get('t_left',  t)
    t_right= results.get('t_right', t)

    l_ok = left  is not None and not np.all(np.isnan(left))
    r_ok = right is not None and not np.all(np.isnan(right))

    if not l_ok and not r_ok:
        return None

    if not l_ok:
        pupil_avg, t_out = right.copy(), t_right
    elif not r_ok:
        pupil_avg, t_out = left.copy(), t_left
    else:
        if len(t_left) >= len(t_right):
            t_out = t_left
            r_al  = np.interp(t_out, t_right, right)
            pupil_avg = np.nanmean(np.stack([left, r_al], axis=1), axis=1)
        else:
            t_out = t_right
            l_al  = np.interp(t_out, t_left, left)
            pupil_avg = np.nanmean(np.stack([l_al, right], axis=1), axis=1)

    # Fill residual NaN before smoothing
    pupil_clean = (
        pd.Series(pupil_avg)
        .interpolate(method='linear', limit_direction='both')
        .ffill().bfill().fillna(0)
        .values
    )

    # Gaussian smoothing
    pupil_filt = gaussian_filter1d(pupil_clean, sigma=SIGMA_SAMP, truncate=3.0)

    # Baseline correction: median of pre-reading window
    baseline_mask = t_out <= BASELINE_MS
    n_baseline    = baseline_mask.sum()
    if n_baseline > 10:
        baseline_val = np.nanmedian(pupil_filt[baseline_mask])
    else:
        baseline_val = np.nanmedian(pupil_filt[:min(100, len(pupil_filt))])

    pupil_corr = pupil_filt - baseline_val
    pupil_corr = pd.Series(pupil_corr).ffill().bfill().fillna(0).values

    return {
        'time_ms':   t_out,
        'pupil':     pupil_corr,
        'pct_left':  results.get('pct_left',  np.nan),
        'pct_right': results.get('pct_right', np.nan),
    }


# ---------------------------------------------------------------------------
# Outlier exclusion (curve level)
# ---------------------------------------------------------------------------

def remove_outlier_curves(curves_by_cond):
    filtered = {}
    excluded = []

    for cond, curves in curves_by_cond.items():
        if len(curves) < 3:
            filtered[cond] = curves
            continue

        means = np.array([np.nanmean(c['pupil_grid']) for c in curves])
        q1, q3 = np.nanpercentile(means, 25), np.nanpercentile(means, 75)
        iqr    = q3 - q1
        lo = q1 - IQR_THRESH * iqr
        hi = q3 + IQR_THRESH * iqr

        kept = []
        for c, m in zip(curves, means):
            if lo <= m <= hi:
                kept.append(c)
            else:
                excluded.append((c['participant'], c['text'], cond, round(float(m), 4)))

        filtered[cond] = kept

    if excluded:
        print(f"\nCurves excluded by IQR ({IQR_THRESH}x):")
        for e in excluded:
            print(f"  {e[0]} / text {e[1]} / {e[2]}  mean={e[3]:.4f}")

    return filtered


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_analysis(data_path, output_dir, text_col='text_id', condition_col='condition'):
    print("Cogni-Action pupillometry pipeline")
    print("Focus: temporal dynamics of pupil dilation during reading")

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path, low_memory=False)
    print(f"Loaded: {df.shape}")

    # Fix decimal comma (Tobii sometimes exports with comma as decimal separator)
    for col in [TS_COL, PUP_L, PUP_R]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                       .str.replace(',', '.', regex=False)
                       .pipe(pd.to_numeric, errors='coerce')
            )

    # Timestamp units: Tobii can export in ms or microseconds
    sample_range = (
        df.groupby(['Participant name', TS_COL])
          .apply(lambda x: x[TS_COL].max() - x[TS_COL].min())
          .median()
        if 'Participant name' in df.columns else
        df[TS_COL].max() - df[TS_COL].min()
    )
    if sample_range > 1_000_000:
        df[TS_COL] = df[TS_COL] / 1000
        print("Timestamps converted from microseconds to milliseconds")

    group_cols = ['Participant name', text_col]
    if condition_col in df.columns:
        group_cols.append(condition_col)

    df_sorted = df.sort_values(group_cols + [TS_COL]).reset_index(drop=True)
    grouped   = df_sorted.groupby(group_cols, sort=False)

    print(f"Processing {grouped.ngroups} text epochs")

    grid = np.linspace(1, 99, N_GRID)
    curves_by_text = {}
    interp_log     = []
    skipped        = []

    for keys, trial_df in grouped:
        participant = keys[0]
        text_id     = keys[1]
        condition   = keys[2] if len(keys) > 2 else 'reading'

        res = preprocess_trial(trial_df, participant, text_id)

        if res is None:
            skipped.append((participant, text_id, 'both eyes invalid'))
            continue

        t_ms  = res['time_ms']
        pupil = res['pupil']

        if len(t_ms) < 4 or t_ms[-1] <= 0:
            skipped.append((participant, text_id, 'insufficient data'))
            continue

        t_pct = t_ms / t_ms[-1] * 100
        _, uid = np.unique(t_pct, return_index=True)
        t_u = t_pct[uid]
        p_u = pupil[uid]

        finite = np.isfinite(t_u) & np.isfinite(p_u)
        if finite.sum() < 4:
            skipped.append((participant, text_id, 'non-finite signal'))
            continue

        pupil_grid = np.interp(grid, t_u[finite], p_u[finite])

        key = str(text_id)
        curves_by_text.setdefault(key, []).append({
            'participant': participant,
            'text':        text_id,
            'pupil_grid':  pupil_grid
        })

        interp_log.append({
            'participant':  participant,
            'text':         text_id,
            'condition':    condition,
            'pct_left':     round(float(res['pct_left']),  2),
            'pct_right':    round(float(res['pct_right']), 2),
        })

    # IQR outlier removal
    curves_clean = remove_outlier_curves(curves_by_text)

    # Save interpolation report
    pd.DataFrame(interp_log).to_csv(
        os.path.join(output_dir, 'interpolation_report.csv'), index=False)

    if skipped:
        print(f"\nSkipped epochs ({len(skipped)}):")
        for s in skipped:
            print(f"  {s[0]} / text {s[1]}: {s[2]}")

    # Figures
    plot_individual_curves(curves_clean, grid, output_dir)
    plot_group_mean(curves_clean, grid, output_dir)

    return curves_clean


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_individual_curves(curves_by_text, grid, output_dir):
    texts = sorted(curves_by_text.keys())
    n     = len(texts)
    if n == 0:
        return

    colors = ['#E63946', '#457B9D', '#2A9D8F']
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, text_id, color in zip(axes, texts, colors):
        curves = curves_by_text.get(text_id, [])
        for c in curves:
            ax.plot(grid, c['pupil_grid'], color=color, lw=0.8, alpha=0.4)

        if curves:
            matrix = np.vstack([c['pupil_grid'] for c in curves])
            mean_c = np.nanmean(matrix, axis=0)
            ax.plot(grid, mean_c, color='black', lw=2.0, label='Mean')

        ax.axhline(0, color='gray', lw=0.8, ls='--', alpha=0.5)
        ax.set_title(f"Text {text_id}", fontsize=13, fontweight='bold')
        ax.set_xlabel('Reading duration (%)', fontsize=12)
        ax.spines[['top', 'right']].set_visible(False)

    axes[0].set_ylabel('Pupil dilation (mm)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'individual_curves.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: individual_curves.png")


def plot_group_mean(curves_by_text, grid, output_dir):
    texts  = sorted(curves_by_text.keys())
    colors = ['#E63946', '#457B9D', '#2A9D8F']

    fig, ax = plt.subplots(figsize=(10, 5))

    for text_id, color in zip(texts, colors):
        curves = curves_by_text.get(text_id, [])
        if not curves:
            continue
        matrix = np.vstack([c['pupil_grid'] for c in curves])
        mean_c = np.nanmean(matrix, axis=0)
        sem_c  = np.nanstd(matrix,  axis=0, ddof=1) / np.sqrt(matrix.shape[0])

        ax.plot(grid, mean_c, color=color, lw=2.0,
                label=f"Text {text_id} (n={len(curves)})")
        ax.fill_between(grid, mean_c - sem_c, mean_c + sem_c,
                         color=color, alpha=0.2)

    ax.axhline(0, color='gray', lw=0.8, ls='--', alpha=0.5)
    ax.set_xlim(1, 99)
    ax.set_xlabel('Reading duration (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Pupil dilation (mm)',  fontsize=13, fontweight='bold')
    ax.legend(frameon=False, fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'group_mean_curves.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: group_mean_curves.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Expected CSV format: one row per sample, with columns:
    #   Participant name, text_id, condition,
    #   Recording timestamp, Pupil diameter left, Pupil diameter right,
    #   Validity left, Validity right
    data_path  = os.path.expanduser("~/Desktop/BD_cogniaccion.csv")
    output_dir = os.path.expanduser("~/Desktop/BD_cogniaccion_outputs")

    curves = run_analysis(data_path, output_dir)
    if curves:
        for text_id, cs in curves.items():
            print(f"Text {text_id}: {len(cs)} valid participant-epochs")
