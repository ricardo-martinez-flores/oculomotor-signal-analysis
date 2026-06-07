# pupil_eyeAD.py
#
# Pupillometry preprocessing pipeline -- Eye-AD project
# Pupillary response to visual oddball (target / distractor words) in MCI
#
# Eye-tracker: Tobii (33 Hz)
# Signal: pupil diameter (pixels, averaged across eyes after validity filtering)
# Paradigm: visual oddball with word stimuli. Each trial = one word presentation
#   (2000 ms epoch from stimulus onset). Targets are infrequent stimuli
#   requiring a motor response; distractors are frequent stimuli to be ignored.
#
# Population: older adults (MCI and cognitively healthy controls).
# Clinical classification based on CSF biomarkers (Abeta42, p-Tau, t-Tau).
#
# Key difference from Cogni-Action and SAPIENS:
# Trials are discrete and time-locked to stimulus onset (event-based paradigm).
# Analysis uses absolute time (ms from stimulus onset), not percentage time,
# because trial durations are fixed and comparison of temporal dynamics
# between conditions requires a common millisecond timeline.
#
# Vergence (binocular angle) is computed in the full pipeline but is not
# included here. The vergence computation is based on proprietary processing
# developed within the research group.
#
# Pipeline overview:
#   1. Extract show_word epochs from continuous recording
#   2. Validity and out-of-range exclusion
#   3. Velocity-based saccade detection and NaN marking
#   4. Interpolation to common time grid (0-2000 ms, 33 Hz)
#   5. Binocular averaging
#   6. Baseline correction (mean of pre-stimulus samples)
#   7. Gaussian smoothing (sigma set by user, in samples at 33 Hz)
#   8. Trial-level QC: interpolation ratio, baseline SD, MAD-based outlier
#   9. Hierarchical averaging: trial -> participant -> group
#
# Why MAD-based outlier removal instead of fixed thresholds:
# The MAD threshold adapts to the distribution of responses within a session.
# A fixed absolute threshold would be too liberal for participants with small
# baseline pupils and too conservative for others.
# MAD scales with the actual variance in the session.
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

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12


# ---------------------------------------------------------------------------
# Monitor and tracker configuration
# ---------------------------------------------------------------------------

MONITOR = {
    "Width":          1024,
    "Height":         768,
    "PhysicalWidth":  340.0,   # mm
    "PhysicalHeight": 190.0,   # mm
    "TrackerHz":      33,
}

# Trial timing
TMAX_MS  = 2000.0
FS       = MONITOR["TrackerHz"]
DT_MS    = 1000.0 / FS
N_SAMP   = int((TMAX_MS / 1000) * FS) + 1
TS_GRID  = np.arange(0, N_SAMP) * DT_MS

# Baseline: first 8 samples (~240 ms pre-stimulus)
IT0 = 8


# ---------------------------------------------------------------------------
# Quality control thresholds
# ---------------------------------------------------------------------------

# Gaussian smoothing sigma (samples at 33 Hz).
# At 33 Hz, 1 sample = ~30 ms. Choose sigma based on the temporal scale
# of the pupil response you expect to capture.
# Example: sigma = 2 samples ~ 60 ms half-width; sigma = 4 ~ 120 ms.
# Too large a sigma will blur the onset and peak of event-locked responses.
GAUSSIAN_SIGMA = None   # set by user (samples at 33 Hz)

QC = {
    # Maximum proportion of samples replaced by interpolation.
    # Trials above this threshold are excluded before group averaging.
    # Typical range in the literature: 0.20 - 0.40.
    'max_interp_ratio': None,   # set by user

    # Maximum SD of the pre-stimulus baseline window.
    # High baseline SD indicates unstable fixation before stimulus onset.
    'max_baseline_sd': None,    # set by user

    # Minimum number of valid (non-interpolated) samples per trial.
    'min_valid_pts': None,      # set by user

    # Velocity threshold for saccade detection (screen units per ms).
    # Samples where eye velocity exceeds this are set to NaN.
    'velocity_thresh': None,    # set by user

    # Multiplier applied to the MAD for amplitude-based outlier removal.
    # Trials whose peak absolute change exceeds median + k*MAD are excluded.
    # See: Leys et al. (2013) J Exp Social Psychol for guidance on k.
    'mad_mult': None,           # set by user
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def myinterp(t, x, t_new):
    valid = np.where(~np.isnan(x))[0]
    if len(valid) > 0:
        return np.interp(t_new, t[valid], x[valid])
    return np.zeros(len(t_new))


def calc_mad(data):
    clean = data[~np.isnan(data)]
    if len(clean) == 0:
        return np.nan
    return np.median(np.abs(clean - np.median(clean)))


def is_valid_duration(trial_df, min_ms=3000, max_ms=6000):
    if len(trial_df) < 2:
        return False
    t = trial_df['timestamp'].values
    dur = (t[-1] - t[0]) / 1000
    return min_ms <= dur <= max_ms


# ---------------------------------------------------------------------------
# Trial-level QC evaluation
# ---------------------------------------------------------------------------

def evaluate_trial_qc(signal, baseline_idx=None):
    if baseline_idx is None:
        baseline_idx = list(range(0, IT0))

    qc = {}

    valid_n    = np.sum(~np.isnan(signal))
    qc['n_valid']       = valid_n
    qc['interp_ratio']  = 1.0 - valid_n / len(signal)

    bl = signal[baseline_idx]
    bl = bl[~np.isnan(bl)]
    qc['baseline_mean'] = np.mean(bl)     if len(bl) >= 3 else np.nan
    qc['baseline_sd']   = np.std(bl)      if len(bl) >= 3 else np.nan

    clean = signal[~np.isnan(signal)]
    qc['max_abs_change'] = max(abs(clean.min()), abs(clean.max())) \
                           if len(clean) > 0 else np.nan

    qc['pass_interp']    = qc['interp_ratio'] <= QC['max_interp_ratio']
    qc['pass_baseline']  = qc['baseline_sd']  <= QC['max_baseline_sd'] \
                           if not np.isnan(qc['baseline_sd']) else False
    qc['pass_n']         = qc['n_valid']      >= QC['min_valid_pts']

    return qc


def apply_mad_filter(trial_list):
    valid_changes = [
        t['qc']['max_abs_change']
        for t in trial_list
        if t['qc']['pass_interp'] and t['qc']['pass_baseline']
           and t['qc']['pass_n']
           and not np.isnan(t['qc']['max_abs_change'])
    ]
    if len(valid_changes) < 3:
        return [False] * len(trial_list), np.nan, np.nan

    arr    = np.array(valid_changes)
    median = np.median(arr)
    mad    = calc_mad(arr)

    if np.isnan(mad) or mad == 0:
        return [False] * len(trial_list), median, mad

    threshold = median + QC['mad_mult'] * mad

    passes = []
    for t in trial_list:
        base_ok = (t['qc']['pass_interp'] and t['qc']['pass_baseline']
                   and t['qc']['pass_n'])
        if base_ok and not np.isnan(t['qc']['max_abs_change']):
            passes.append(t['qc']['max_abs_change'] <= threshold)
        else:
            passes.append(False)

    return passes, median, mad


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_participant_trials(df):
    print("Processing trials (QC: interpolation ratio, baseline SD, MAD)")

    df = df.copy()

    numeric_cols = [
        'timestamp', 'leftEyeRawX', 'leftEyeRawY', 'rightEyeRawX', 'rightEyeRawY',
        'leftEyePupilSize', 'rightEyePupilSize',
        'leftEyeValidity', 'rightEyeValidity',
        'leftEye3dX', 'leftEye3dY', 'leftEye3dZ',
        'rightEye3dX', 'rightEye3dY', 'rightEye3dZ',
        'validityScore', 'currentTrial', 'strColourCode',
        'resultResponse', 'reactionTime'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    screen_w = MONITOR["PhysicalWidth"]
    screen_h = MONITOR["PhysicalHeight"]

    trial_list = []

    for participant in df['participant_id'].unique():
        df_p = df[df['participant_id'] == participant].copy()

        for condition in ['pre', 'post']:
            df_c = df_p[df_p['condition'] == condition].copy()
            if len(df_c) == 0:
                continue

            for trial in df_c['currentTrial'].unique():
                dfX = df_c[df_c['currentTrial'] == trial].copy()

                if len(dfX) < 10 or not is_valid_duration(dfX):
                    continue

                # Stimulus identity (1 = distractor, 2 = target)
                try:
                    code_vals = dfX['strColourCode'].dropna()
                    stim_code = float(code_vals.iloc[0]) if len(code_vals) > 0 else np.nan
                except Exception:
                    stim_code = np.nan

                stimulus = 'Target' if stim_code == 2 else 'Distractor'

                # Reaction time and response
                try:
                    rt = float(dfX['reactionTime'].dropna().iloc[0])
                except Exception:
                    rt = np.nan
                try:
                    resp = float(dfX['resultResponse'].dropna().iloc[0])
                except Exception:
                    resp = np.nan

                t_raw = dfX['timestamp'].values / 1e3
                t_raw = t_raw - t_raw[0]

                # Identify show_word epoch
                Its = np.where(
                    (dfX['currentTrial'].values == trial) &
                    (dfX['strState'].values == 'show_word')
                )[0]

                if len(Its) < 10:
                    continue

                itso, itf = Its[0], Its[-1]
                tso = t_raw[itso]
                tw  = t_raw[itso:itf+1] - tso

                lx01 = dfX['leftEyeRawX'].values[itso:itf+1]
                ly01 = dfX['leftEyeRawY'].values[itso:itf+1]
                rx01 = dfX['rightEyeRawX'].values[itso:itf+1]
                ry01 = dfX['rightEyeRawY'].values[itso:itf+1]
                lp   = dfX['leftEyePupilSize'].values[itso:itf+1]
                rp   = dfX['rightEyePupilSize'].values[itso:itf+1]
                lv   = dfX['leftEyeValidity'].values[itso:itf+1]
                rv   = dfX['rightEyeValidity'].values[itso:itf+1]

                n = min(len(lx01), len(ly01), len(rx01), len(ry01),
                        len(lp), len(rp), len(lv), len(rv))
                lx01, ly01, rx01, ry01 = lx01[:n], ly01[:n], rx01[:n], ry01[:n]
                lp, rp, lv, rv = lp[:n], rp[:n], lv[:n], rv[:n]
                tw = tw[:n]

                # Convert gaze coordinates to physical screen space
                lx = (lx01 - 0.5) * screen_w
                ly = (0.5 - ly01) * screen_h
                rx = (rx01 - 0.5) * screen_w
                ry = (0.5 - ry01) * screen_h

                # Invalid samples: out of range or non-zero validity
                Iexcl = np.where(
                    (lx01 < 0) | (ly01 < 0) | (rx01 < 0) | (ry01 < 0) |
                    (lv > 0) | (rv > 0)
                )[0]

                for arr in [lx, ly, rx, ry, lp, rp]:
                    arr[Iexcl] = np.nan

                # Velocity-based saccade rejection
                vlx = np.concatenate([[0], np.diff(lx - lx)]) / DT_MS
                vly = np.concatenate([[0], np.diff(ly - ly)]) / DT_MS
                vrx = np.concatenate([[0], np.diff(rx - rx)]) / DT_MS
                vry = np.concatenate([[0], np.diff(ry - ry)]) / DT_MS

                fast = np.where(
                    (vlx**2 + vly**2 >= QC['velocity_thresh']**2) |
                    (vrx**2 + vry**2 >= QC['velocity_thresh']**2)
                )[0]

                for arr in [lp, rp]:
                    if len(fast) > 0:
                        arr[fast] = np.nan

                try:
                    lp_i = myinterp(tw, lp, TS_GRID)
                    rp_i = myinterp(tw, rp, TS_GRID)
                    p    = (lp_i + rp_i) / 2.0
                except Exception:
                    continue

                # Baseline correction and Gaussian smoothing
                it0_adj = min(IT0, len(p) - 1)
                p0 = np.nanmean(p[:it0_adj+1])
                if np.isnan(p0):
                    continue

                p_bc     = p - p0
                p_smooth = gaussian_filter1d(p_bc.astype(float),
                                             sigma=GAUSSIAN_SIGMA,
                                             mode='nearest')

                qc = evaluate_trial_qc(p_smooth)

                trial_list.append({
                    'participant': participant,
                    'condition':   condition,
                    'trial':       trial,
                    'stimulus':    stimulus,
                    'stim_code':   stim_code,
                    'rt':          rt,
                    'response':    resp,
                    'pupil':       p_smooth.copy(),
                    'qc':          qc
                })

    print(f"Trials extracted: {len(trial_list)}")

    passes_mad, _, _ = apply_mad_filter(trial_list)
    n_mad = sum(passes_mad)
    print(f"After MAD filter: {n_mad}/{len(trial_list)}")

    return trial_list, passes_mad


# ---------------------------------------------------------------------------
# Hierarchical averaging
# ---------------------------------------------------------------------------

def hierarchical_average(trial_list, passes, value_key='pupil',
                          n_bins=15):
    df_records = []
    for i, t in enumerate(trial_list):
        if passes[i]:
            for j, val in enumerate(t[value_key]):
                df_records.append({
                    'participant': t['participant'],
                    'condition':   t['condition'],
                    'stimulus':    t['stimulus'],
                    'timestamp':   TS_GRID[j],
                    'value':       val
                })

    df = pd.DataFrame(df_records)
    if len(df) == 0:
        return []

    bins    = np.linspace(0, TMAX_MS, n_bins + 1)
    results = []

    for stimulus in ['Target', 'Distractor']:
        for condition in ['pre', 'post']:
            sub = df[(df['stimulus'] == stimulus) & (df['condition'] == condition)]
            if len(sub) == 0:
                continue

            part_curves = []
            for pid in sub['participant'].unique():
                p_data = sub[sub['participant'] == pid]

                binned = []
                bin_c  = []
                for j in range(len(bins) - 1):
                    mask = (p_data['timestamp'] >= bins[j]) & \
                           (p_data['timestamp'] <  bins[j+1])
                    if mask.sum() >= 3:
                        binned.append(p_data[mask]['value'].mean())
                        bin_c.append((bins[j] + bins[j+1]) / 2)

                if len(binned) >= 5:
                    binned = np.array(binned) - binned[0]
                    part_curves.append({'times': np.array(bin_c), 'values': binned})

            if len(part_curves) < 3:
                continue

            all_t = np.unique(np.concatenate([c['times'] for c in part_curves]))
            matrix = np.array([
                np.interp(all_t, c['times'], c['values'])
                for c in part_curves
            ])
            mean_v = np.nanmean(matrix, axis=0)
            sem_v  = np.nanstd(matrix,  axis=0, ddof=1) / np.sqrt(len(part_curves))

            results.append({
                'stimulus':       stimulus,
                'condition':      condition,
                'times':          all_t,
                'mean':           mean_v,
                'sem':            sem_v,
                'n_participants': len(part_curves)
            })

    return results


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_pupil(results, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))

    colors  = {'Target': '#D62828', 'Distractor': '#4361EE'}
    lstyles = {'pre': '-', 'post': '--'}

    for r in results:
        color  = colors[r['stimulus']]
        ls     = lstyles[r['condition']]
        label  = f"{r['stimulus']} {r['condition'].upper()} (n={r['n_participants']})"
        ax.plot(r['times'], r['mean'], color=color, ls=ls, lw=2.5, label=label)
        ax.fill_between(r['times'],
                        r['mean'] - r['sem'],
                        r['mean'] + r['sem'],
                        color=color, alpha=0.2)

    ax.axhline(0, color='gray', lw=0.8, ls=':', alpha=0.5)
    ax.set_xlim(0, TMAX_MS)
    ax.set_xlabel('Time from stimulus onset (ms)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Pupil dilation (mm)',            fontsize=13, fontweight='bold')
    ax.set_title('Pupillary response -- oddball paradigm', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    out = os.path.join(output_dir, 'pupil_oddball.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: pupil_oddball.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_analysis(data_path, output_dir):
    print("Eye-AD pupillometry pipeline")
    print("Paradigm: visual oddball (target/distractor words)")
    print("Note: vergence computation is not included in this script.")

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    print(f"Loaded: {df.shape}")
    print(f"Participants: {df['participant_id'].nunique()}")

    trial_list, passes = process_participant_trials(df)

    if not any(passes):
        print("No valid trials after QC.")
        return None

    # Save trial-level data
    records = []
    for i, t in enumerate(trial_list):
        if passes[i]:
            for j, val in enumerate(t['pupil']):
                records.append({
                    'participant': t['participant'],
                    'condition':   t['condition'],
                    'trial':       t['trial'],
                    'stimulus':    t['stimulus'],
                    'stim_code':   t['stim_code'],
                    'rt':          t['rt'],
                    'response':    t['response'],
                    'timestamp_ms': TS_GRID[j],
                    'pupil':       val
                })

    df_out = pd.DataFrame(records)
    df_out.to_csv(os.path.join(output_dir, 'pupil_trials_clean.csv'),
                  index=False, sep=';', decimal=',')

    # Hierarchical averaging and figure
    results = hierarchical_average(trial_list, passes)
    if results:
        plot_pupil(results, output_dir)

    # QC summary
    print("\nQC summary (accepted trials):")
    for stimulus in ['Target', 'Distractor']:
        for cond in ['pre', 'post']:
            n = sum(1 for i, t in enumerate(trial_list)
                    if passes[i] and t['stimulus'] == stimulus
                    and t['condition'] == cond)
            print(f"  {stimulus} {cond}: {n} trials")

    print(f"\nOutputs saved to: {output_dir}")
    return df_out


if __name__ == "__main__":
    # Expected data: single CSV with columns:
    #   participant_id, condition (pre/post), currentTrial, strState,
    #   timestamp, leftEyeRawX/Y, rightEyeRawX/Y,
    #   leftEyePupilSize, rightEyePupilSize,
    #   leftEyeValidity, rightEyeValidity,
    #   leftEye3dX/Y/Z, rightEye3dX/Y/Z,
    #   strColourCode, resultResponse, reactionTime
    data_path  = os.path.expanduser("~/Desktop/Palabras/experimental1.csv")
    output_dir = os.path.expanduser("~/Desktop/Palabras/outputs")

    df_results = run_analysis(data_path, output_dir)
