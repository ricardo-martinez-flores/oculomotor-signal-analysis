# pupil_sapiens.py
#
# Pupillometry preprocessing pipeline -- SAPIENS project
# Cognitive load across 9 conditions in healthy young adults
#
# Eye-tracker: Pupil Labs (Pupil Invisible / Neon)
# Sampling rate: 60 Hz (estimated from timestamps)
# Signal: 3D pupil diameter in mm
# Confidence: model_confidence per sample (0-1)
#
# Paradigm: participants viewed 9 conditions of varying cognitive demand
# (Basal, Scroll, Reality, Music, Reading, Podcast, Tetris, Documentary, N-back)
# in a single continuous session. Conditions are stored as separate CSV files
# named {participant_id}_{condition_code}.csv
#
# Pipeline overview:
#   1. Confidence gate: samples below threshold set to NaN
#   2. Confidence-based interpolation with edge padding
#   3. Gaussian smoothing
#   4. Baseline correction (median of first 20 s from basal condition)
#   5. Normalization to percentage of condition duration
#   6. AUC and peak extraction per condition
#
# Why this order matters:
#   Smoothing after interpolation ensures no NaN propagation through the
#   Gaussian kernel. Baseline correction after smoothing uses the same
#   smoothed signal that will be analyzed, avoiding inconsistency between
#   the reference and the data.
#
# Author: Ricardo Martinez-Flores
# Contact: ricardo.antonio.martinezf@gmail.com
# License: MIT

import pandas as pd
import numpy as np
import os
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import trapezoid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Global parameters
# ---------------------------------------------------------------------------

# Confidence threshold: samples below this value are treated as invalid.
# Pupil Labs model_confidence reflects tracking reliability. Values below 0.6
# generally indicate blinks, partial occlusions, or tracking loss.
CONFIDENCE_THRESHOLD = 0.6

# Edge padding around invalid regions (ms).
# Transitions into and out of blinks typically corrupt 50-100 ms of signal
# on either side. Padding prevents interpolating over artifact-contaminated edges.
PAD_MS = 80

# Gaussian smoothing sigma (samples).
# Choose based on the timescale of the effect you expect to detect.
# A sigma of N samples corresponds to a low-pass cutoff of ~1/(2*pi*N/fs) Hz.
# Larger sigma = more smoothing = lower cutoff frequency.
# Too large a sigma will attenuate genuine fast pupil responses.
GAUSSIAN_SIGMA = None   # set by user (samples)

# Baseline window (s): first 20 seconds of the basal condition.
BASELINE_WINDOW_S = 20

# Outlier detection threshold (z-score on per-participant mean AUC).
# Participants whose mean AUC exceeds this z-score are excluded from group figures.
# Typical values in the literature: 2.5 - 3.5. More conservative = fewer exclusions.
Z_OUTLIER = None   # set by user

# Number of time bins for group-level curve construction.
BIN_WIDTH_PCT = 1.8  # percent of trial duration

# Font for figures
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16


# ---------------------------------------------------------------------------
# Condition mapping
# ---------------------------------------------------------------------------

CONDITIONS = {
    '01': 'Basal',
    '02': 'Documentary',
    '03': 'Reading',
    '04': 'Music',
    '05': 'Podcast',
    '06': 'Reality',
    '07': 'Scroll',
    '08': 'Tetris',
    '09': 'N-back'
}

CONDITION_COLORS = {
    'Basal':       '#2E8B57',
    'Documentary': '#4169E1',
    'Reading':     '#FF6347',
    'Music':       '#9370DB',
    'Podcast':     '#20B2AA',
    'Reality':     '#FF69B4',
    'Scroll':      '#FFA500',
    'Tetris':      '#DC143C',
    'N-back':      '#000000'
}

# Ordered display for figures (from low to high expected cognitive load)
DISPLAY_ORDER = ['Basal', 'Scroll', 'Reality', 'Music',
                 'Reading', 'Podcast', 'Tetris', 'Documentary', 'N-back']

# N-back trial timing per participant (start_s, end_s).
# N-back has a fixed internal structure (2-back and 3-back blocks).
# Only one trial per participant is used for analysis, selected based on
# signal quality. Times were verified against raw event logs.
NBACK_TIMES = {
    '01': {
        'nback2_1': (5*60+21, 6*60+2),  'nback2_2': (6*60+48, 7*60+28),
        'nback2_3': (8*60+18, 8*60+58), 'nback3_1': (9*60+55, 10*60+36),
        'nback3_2': (11*60+20, 12*60+0),'nback3_3': (12*60+40, 13*60+21)
    },
    '02': {
        'nback2_1': (5*60+5,  5*60+46), 'nback2_2': (6*60+20, 7*60+1),
        'nback2_3': (7*60+35, 8*60+15), 'nback3_1': (9*60+30, 10*60+13),
        'nback3_2': (10*60+47,11*60+30),'nback3_3': (12*60+2,  12*60+44)
    },
    '03': {
        'nback2_1': (5*60+13, 5*60+54), 'nback2_2': (6*60+30, 7*60+11),
        'nback2_3': (7*60+40, 8*60+21), 'nback3_1': (9*60+32, 10*60+13),
        'nback3_2': (10*60+48,11*60+30),'nback3_3': (12*60+4,  12*60+44)
    },
    '05': {
        'nback2_1': (4*60+9,  4*60+50), 'nback2_2': (5*60+18, 5*60+58),
        'nback2_3': (6*60+28, 7*60+9),  'nback3_1': (7*60+45, 8*60+26),
        'nback3_2': (8*60+47, 9*60+28), 'nback3_3': (9*60+58, 10*60+39)
    },
    '06': {
        'nback2_1': (4*60+55, 5*60+37), 'nback2_2': (6*60+3,  6*60+43),
        'nback2_3': (7*60+12, 7*60+52), 'nback3_1': (8*60+22, 9*60+2),
        'nback3_2': (9*60+19, 9*60+59), 'nback3_3': (10*60+10,10*60+50)
    },
    '07': {
        'nback2_1': (6*60+33, 7*60+13), 'nback2_2': (7*60+34, 8*60+16),
        'nback2_3': (8*60+44, 9*60+24), 'nback3_1': (10*60+8, 10*60+50),
        'nback3_2': (11*60+5, 11*60+46),'nback3_3': (11*60+58,12*60+39)
    },
    '08': {
        'nback2_1': (5*60+13, 5*60+53), 'nback2_2': (6*60+52, 7*60+32),
        'nback2_3': (8*60+26, 9*60+6),  'nback3_1': (10*60+1, 10*60+42),
        'nback3_2': (11*60+21,12*60+1), 'nback3_3': (12*60+30,13*60+10)
    },
    '09': {
        'nback2_1': (4*60+10, 4*60+51), 'nback2_2': (5*60+11, 5*60+52),
        'nback2_3': (6*60+15, 6*60+55), 'nback3_1': (8*60+18, 9*60+0),
        'nback3_2': (9*60+59, 10*60+40),'nback3_3': (11*60+1, 11*60+42)
    }
}

# Best trial per participant for N-back analysis.
# Selection was based on signal quality (lowest interpolation %, cleanest
# baseline) verified in a diagnostic step not shown here.
ALLOWED_NBACK_TRIALS = {
    '01': 'nback2_1', '02': 'nback3_3', '03': 'nback3_2',
    '05': 'nback3_2', '06': 'nback3_2', '07': 'nback3_2',
    '08': 'nback2_1', '09': 'nback2_1'
}

# Per-participant, per-condition eye exclusions.
# These were identified during individual signal inspection when one eye
# showed sustained tracking failure independent of blinks.
EYE_EXCLUSIONS = {
    '02': {'04': 'exclude_left',  '05': 'exclude_left',  '08': 'exclude_right'},
    '05': {'01': 'exclude_left',  '03': 'exclude_left',  '04': 'exclude_left'},
    '03': {'01': 'exclude_left',  '03': 'exclude_left',  '04': 'exclude_right'},
    '08': {'01': 'exclude_left',  '03': 'exclude_right', '05': 'exclude_right'},
    '06': {'08': 'exclude_left'}
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def compute_fs(time_s):
    t = np.asarray(time_s)
    if len(t) < 3:
        return 60.0
    dt = np.median(np.diff(t))
    return float(np.round(1.0 / dt, 3)) if dt > 0 else 60.0


def interpolate_with_padding(signal, time_s, confidence, fs,
                              confidence_threshold=CONFIDENCE_THRESHOLD,
                              pad_ms=PAD_MS,
                              long_gap_threshold_s=0.5):
    x    = np.array(signal,     dtype=float)
    t    = np.array(time_s,     dtype=float)
    conf = np.array(confidence, dtype=float)

    invalid = (conf < confidence_threshold) | np.isnan(x)

    # Identify long gaps for reporting
    diff_flag = np.diff(np.concatenate(([0], invalid.astype(int), [0])))
    starts = np.where(diff_flag ==  1)[0]
    ends   = np.where(diff_flag == -1)[0]
    long_gaps = []
    for s, e in zip(starts, ends):
        t_start = t[s]   if s   < len(t) else t[-1]
        t_end   = t[e-1] if e-1 < len(t) else t[-1]
        dur = t_end - t_start
        if dur >= long_gap_threshold_s:
            long_gaps.append((t_start, t_end, dur))

    # Pad invalid regions
    pad_samples = int(round(pad_ms / 1000.0 * fs))
    padded = invalid.copy()
    for s, e in zip(starts, ends):
        a = max(0, s - pad_samples)
        b = min(len(x), e + pad_samples)
        padded[a:b] = True

    src = x.copy()
    src[padded] = np.nan

    valid_idx = np.where(~np.isnan(src))[0]
    out = src.copy()
    if len(valid_idx) >= 2:
        nan_idx = np.where(np.isnan(src))[0]
        if len(nan_idx) > 0:
            out[nan_idx] = np.interp(t[nan_idx], t[valid_idx], src[valid_idx])

    if np.sum(~np.isnan(out)) < 2:
        fill = np.nanmedian(x) if np.isfinite(np.nanmedian(x)) else 0.0
        out[:] = fill

    n_invalid_orig = int(np.sum(invalid))
    n_still_nan    = int(np.sum(np.isnan(out)))
    pct_interp = 100.0 * (n_invalid_orig - n_still_nan) / len(x) if len(x) > 0 else 0.0

    return out, pct_interp, long_gaps


def process_eye(df_eye, participant, condition_code):
    fs = compute_fs(df_eye['time_s'].values)
    conf = df_eye['model_confidence'].fillna(0).values
    raw  = df_eye['diameter_3d'].values

    interp_sig, pct, long_gaps = interpolate_with_padding(
        raw, df_eye['time_s'].values, conf, fs)

    smoothed = gaussian_filter1d(interp_sig.astype(float),
                                 sigma=GAUSSIAN_SIGMA, mode='nearest')
    return smoothed, df_eye['time_s'].values, pct, long_gaps


def load_condition_file(path, participant, condition_code):
    df = None
    for sep in [';', ',', '\t']:
        try:
            tmp = pd.read_csv(path, sep=sep, nrows=3)
            if len(tmp.columns) >= 4:
                df = pd.read_csv(path, sep=sep)
                break
        except Exception:
            continue

    if df is None:
        return None

    expected = ['pupil_timestamp', 'eye_id', 'diameter_3d', 'model_confidence']
    if len(df.columns) == 4 and not all(c in df.columns for c in expected):
        df.columns = expected

    for col in expected:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=expected).copy()
    if len(df) == 0:
        return None

    df = df.sort_values('pupil_timestamp').copy()
    df['time_s'] = df['pupil_timestamp'] - df['pupil_timestamp'].iloc[0]
    return df


def compute_binocular(left_sig, left_t, right_sig, right_t, exclusion_rule):
    l_ok = left_sig  is not None and np.sum(~np.isnan(left_sig))  > 0
    r_ok = right_sig is not None and np.sum(~np.isnan(right_sig)) > 0

    if exclusion_rule == 'exclude_left':
        if not r_ok:
            return None, None
        return right_sig, right_t

    if exclusion_rule == 'exclude_right':
        if not l_ok:
            return None, None
        return left_sig, left_t

    if not l_ok and not r_ok:
        return None, None
    if not l_ok:
        return right_sig, right_t
    if not r_ok:
        return left_sig, left_t

    # Align to longer eye timeline, then average
    if len(left_t) >= len(right_t):
        t_ref = left_t
        r_aligned = np.interp(t_ref, right_t, right_sig)
        combined = np.nanmean(np.stack([left_sig, r_aligned], axis=1), axis=1)
    else:
        t_ref = right_t
        l_aligned = np.interp(t_ref, left_t, left_sig)
        combined = np.nanmean(np.stack([l_aligned, right_sig], axis=1), axis=1)

    return combined, t_ref


def normalize_to_grid(time_s, signal, n_points=500):
    t_pct = (time_s - time_s[0]) / (time_s[-1] - time_s[0]) * 100
    _, uid = np.unique(t_pct, return_index=True)
    t_u = t_pct[uid]
    s_u = signal[uid]

    finite = np.isfinite(t_u) & np.isfinite(s_u)
    if finite.sum() < 4:
        return None, None

    grid = np.linspace(1, 99, n_points)
    interp = np.interp(grid, t_u[finite], s_u[finite])
    return grid, interp


def compute_metrics(grid, signal_grid, baseline_val):
    corrected = signal_grid - baseline_val
    auc = float(trapezoid(np.maximum(corrected, 0), grid))
    peak_val  = float(np.max(corrected))
    peak_time = float(grid[np.argmax(corrected)])
    return {'auc': auc, 'peak_value': peak_val, 'peak_time_pct': peak_time,
            'baseline': baseline_val}


def detect_outliers(metrics_by_cond, z_thresh=Z_OUTLIER):
    outliers = {}
    for cond, entries in metrics_by_cond.items():
        if len(entries) < 3:
            continue
        aucs = np.array([e['auc'] for e in entries])
        z = np.abs(stats.zscore(aucs))
        bad = [entries[i]['participant'] for i in np.where(z > z_thresh)[0]]
        if bad:
            outliers[cond] = bad
    return outliers


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_analysis(data_dir, output_dir):
    print("SAPIENS pupillometry pipeline")
    print("Filtering: confidence gate -> interpolation -> Gaussian (sigma=30)")

    os.makedirs(output_dir, exist_ok=True)

    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    participant_files = {}
    for fn in csv_files:
        base = fn.replace('.csv', '')
        if '_' in base:
            parts = base.split('_')
            if len(parts) == 2:
                pid, code = parts
                participant_files.setdefault(pid, {})[code] = \
                    os.path.join(data_dir, fn)

    print(f"Participants found: {sorted(participant_files.keys())}")

    interp_log  = []
    metrics_all = []
    curves_by_cond = {c: [] for c in CONDITIONS}

    for pid in sorted(participant_files.keys()):
        files = participant_files[pid]
        print(f"\nParticipant {pid}")

        if '01' not in files:
            print("  No basal file -- skipped")
            continue

        df_basal = load_condition_file(files['01'], pid, '01')
        if df_basal is None:
            print("  Could not load basal -- skipped")
            continue

        # Baseline from first 20 s of basal condition (after smoothing)
        left_df  = df_basal[df_basal['eye_id'] == 0].copy()
        right_df = df_basal[df_basal['eye_id'] == 1].copy()

        l_sig, l_t, _, _ = process_eye(left_df,  pid, '01') if len(left_df)  > 0 else (None, None, None, None)
        r_sig, r_t, _, _ = process_eye(right_df, pid, '01') if len(right_df) > 0 else (None, None, None, None)

        excl_basal = EYE_EXCLUSIONS.get(pid, {}).get('01', None)
        basal_combined, basal_t = compute_binocular(l_sig, l_t, r_sig, r_t, excl_basal)

        if basal_combined is None:
            print("  Basal signal failed -- skipped")
            continue

        baseline_mask = basal_t <= BASELINE_WINDOW_S
        if baseline_mask.sum() < 10:
            baseline_val = np.nanmedian(basal_combined[:min(100, len(basal_combined))])
        else:
            baseline_val = np.nanmedian(basal_combined[baseline_mask])

        print(f"  Baseline = {baseline_val:.4f} mm")

        for code in sorted(CONDITIONS.keys()):
            if code not in files:
                continue

            df_cond = load_condition_file(files[code], pid, code) if code != '01' \
                      else df_basal

            if df_cond is None:
                continue

            # N-back: extract only the selected trial
            if code == '09':
                trial_name = ALLOWED_NBACK_TRIALS.get(pid)
                if trial_name is None or pid not in NBACK_TIMES:
                    continue
                if trial_name not in NBACK_TIMES[pid]:
                    continue
                t_start, t_end = NBACK_TIMES[pid][trial_name]
                mask = (df_cond['time_s'] >= t_start - 2) & \
                       (df_cond['time_s'] <= t_end + 2)
                df_cond = df_cond[mask].copy()
                if len(df_cond) < 10:
                    continue

            left_df  = df_cond[df_cond['eye_id'] == 0].copy()
            right_df = df_cond[df_cond['eye_id'] == 1].copy()

            l_sig, l_t, l_pct, l_gaps = process_eye(left_df,  pid, code) \
                if len(left_df)  > 0 else (None, None, np.nan, [])
            r_sig, r_t, r_pct, r_gaps = process_eye(right_df, pid, code) \
                if len(right_df) > 0 else (None, None, np.nan, [])

            excl = EYE_EXCLUSIONS.get(pid, {}).get(code, None)
            combined, t_out = compute_binocular(l_sig, l_t, r_sig, r_t, excl)

            if combined is None:
                continue

            grid, grid_sig = normalize_to_grid(t_out, combined)
            if grid is None:
                continue

            m = compute_metrics(grid, grid_sig, baseline_val)
            m['participant'] = pid
            m['condition']   = CONDITIONS[code]

            metrics_all.append(m)
            curves_by_cond[code].append({
                'participant': pid,
                'grid':        grid,
                'signal':      grid_sig - baseline_val
            })

            interp_log.append({
                'participant': pid,
                'condition':   code,
                'pct_left':    round(float(l_pct), 2) if not np.isnan(l_pct) else np.nan,
                'pct_right':   round(float(r_pct), 2) if not np.isnan(r_pct) else np.nan,
                'n_gaps_left': len(l_gaps),
                'n_gaps_right': len(r_gaps)
            })

            print(f"  {CONDITIONS[code]:15s} AUC={m['auc']:.2f}  "
                  f"peak={m['peak_value']:.3f} mm  "
                  f"interp L={l_pct:.1f}% R={r_pct:.1f}%")

    # Outlier detection
    metrics_by_cond = {}
    for m in metrics_all:
        code = [k for k, v in CONDITIONS.items() if v == m['condition']]
        if code:
            metrics_by_cond.setdefault(code[0], []).append(m)

    outliers = detect_outliers(metrics_by_cond)
    if outliers:
        print(f"\nOutliers detected (z > {Z_OUTLIER}):")
        for cond, pids in outliers.items():
            print(f"  {cond}: {pids}")

    # Save outputs
    df_metrics = pd.DataFrame(metrics_all)
    df_metrics.to_csv(os.path.join(output_dir, 'metrics_long.csv'), index=False)

    df_interp = pd.DataFrame(interp_log)
    df_interp.to_csv(os.path.join(output_dir, 'interpolation_report.csv'), index=False)

    # Figures
    plot_group_curves(curves_by_cond, outliers, output_dir)
    plot_auc_bars(df_metrics, outliers, output_dir)

    print(f"\nOutputs saved to: {output_dir}")
    return df_metrics


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_group_curves(curves_by_cond, outliers, output_dir):
    groups = {
        'Basic':   ['01', '09'],
        'Passive': ['07', '06', '04'],
        'Active':  ['03', '05', '08', '02']
    }

    bins = np.arange(0, 100.2, BIN_WIDTH_PCT)

    fig, axes = plt.subplots(2, 2, figsize=(24, 16))
    axes = axes.flatten()

    def bin_curves(code):
        bad_pids = outliers.get(code, [])
        curves = [c for c in curves_by_cond.get(code, [])
                  if c['participant'] not in bad_pids]
        if len(curves) == 0:
            return None

        bin_means, bin_sems, bin_centers = [], [], []
        for j in range(len(bins) - 1):
            per_part = []
            for c in curves:
                mask = (c['grid'] >= bins[j]) & (c['grid'] < bins[j+1])
                if mask.sum() > 0:
                    per_part.append(np.mean(c['signal'][mask]))
            if len(per_part) >= 2:
                arr = np.array(per_part)
                bin_centers.append(bins[j] + BIN_WIDTH_PCT / 2)
                bin_means.append(np.mean(arr))
                bin_sems.append(np.std(arr, ddof=1) / np.sqrt(len(arr)))

        if len(bin_centers) < 5:
            return None

        bm = np.array(bin_means)
        bs = np.array(bin_sems)
        bc = np.array(bin_centers)

        # Mild group-level smoothing (sigma=2 bins, ~3.6% of trial)
        from scipy.ndimage import gaussian_filter1d as gf1d
        bm = gf1d(bm.astype(float), sigma=2, mode='nearest')
        bs = gf1d(bs.astype(float), sigma=2, mode='nearest')

        return bc, bm, bs, len(curves)

    cond_data = {}
    for code in CONDITIONS:
        res = bin_curves(code)
        if res is not None:
            cond_data[code] = res

    for g_idx, (g_name, codes) in enumerate(groups.items()):
        ax = axes[g_idx]
        for code in codes:
            if code not in cond_data:
                continue
            bc, bm, bs, n = cond_data[code]
            cname = CONDITIONS[code]
            color = CONDITION_COLORS[cname]
            ax.plot(bc, bm, color=color, lw=2.5,
                    label=f"{cname} (n={n})", alpha=0.9)
            ax.fill_between(bc, bm - bs, bm + bs, color=color, alpha=0.2)

        ax.axhline(0, color='gray', lw=0.8, ls='--', alpha=0.5)
        ax.set_xlabel('Task duration (%)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Pupil dilation (mm)', fontsize=14, fontweight='bold')
        ax.set_title(g_name, fontsize=16, fontweight='bold')
        ax.legend(loc='lower left', fontsize=11)
        ax.set_xlim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

    # All conditions panel
    ax_all = axes[3]
    for code in CONDITIONS:
        if code not in cond_data:
            continue
        bc, bm, bs, n = cond_data[code]
        cname = CONDITIONS[code]
        ax_all.plot(bc, bm, color=CONDITION_COLORS[cname], lw=2.5,
                    label=f"{cname} (n={n})", alpha=0.9)

    ax_all.axhline(0, color='gray', lw=0.8, ls='--', alpha=0.5)
    ax_all.set_xlabel('Task duration (%)', fontsize=14, fontweight='bold')
    ax_all.set_ylabel('Pupil dilation (mm)', fontsize=14, fontweight='bold')
    ax_all.set_title('All conditions', fontsize=16, fontweight='bold')
    ax_all.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax_all.set_xlim(0, 100)
    ax_all.grid(True, alpha=0.3)
    ax_all.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'group_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: group_curves.png")


def plot_auc_bars(df_metrics, outliers, output_dir):
    bad_pids = set()
    for pids in outliers.values():
        bad_pids.update(pids)

    df_clean = df_metrics[~df_metrics['participant'].isin(bad_pids)].copy()

    means = df_clean.groupby('condition')['auc'].mean()
    sems  = df_clean.groupby('condition')['auc'].sem()
    counts = df_clean.groupby('condition')['auc'].count()

    ordered = [c for c in DISPLAY_ORDER if c in means.index]

    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(ordered))
    colors = [CONDITION_COLORS[c] for c in ordered]
    bars = ax.bar(x, [means[c] for c in ordered], color=colors, alpha=0.8)

    for bar, cname in zip(bars, ordered):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + means.max() * 0.02,
                f"{means[cname]:.1f}",
                ha='center', va='bottom', fontweight='bold', fontsize=13)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{c}\n(n={counts.get(c, 0)})" for c in ordered],
        fontsize=13
    )
    ax.set_ylabel('AUC (area under curve)', fontsize=14, fontweight='bold')
    ax.set_title('Pupillary AUC by condition', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'auc_bars.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: auc_bars.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Data directory: one CSV per participant-condition pair
    # Named as {participant_id}_{condition_code}.csv
    # e.g., 01_01.csv (participant 01, basal condition)
    data_dir   = os.path.expanduser("~/Desktop/Sapiens")
    output_dir = os.path.expanduser("~/Desktop/Sapiens/outputs")

    df_metrics = run_analysis(data_dir, output_dir)

    if df_metrics is not None:
        print("\nPer-condition AUC summary:")
        print(df_metrics.groupby('condition')['auc']
              .agg(['mean', 'std', 'count']).round(3))
