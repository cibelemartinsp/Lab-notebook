#%%
###################################################################################
###### LEADERBOARD + INTERACTIVE ROI SELECTOR -> DECONVOLVED TRACES FIGURE ########
###### FLEXIBLE VERSION: Works with Adult and Young (by age groups) data        ########
###################################################################################

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

sys.path.append('../physion/src')
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.dataviz.raw import plot as plot_raw
from physion.imaging.dcnv import oasis
from physion.utils import plot_tools as pt

import physion
print('physion imported from:', physion.__file__)

# keep text editable in the SVG (Illustrator / Inkscape)
plt.rcParams['svg.fonttype'] = 'none'

# NOTE: the add_VisualStim hack is GONE. physion draws the grating icons by itself
# as long as Data(..., with_visual_stim=True) and init_stim(data) are used.

# ------------------------------------------------------------
# 1) CONFIG - FLEXIBLE DATASET SELECTION
# ------------------------------------------------------------

# ============================================================
# DATASET CONFIGURATION
# ============================================================

# Choose which dataset to analyze:
# - Set to one of the folder names from Dataset_Organization.py
#DATASET_NAME = "SST-cells_cond-GluN1-KO_Adult_V1"  # Adult example
#DATASET_NAME = "PV-cells_WT_Adult_V1"          # Adult example
#DATASET_NAME = "PV-cells_cond-GluN1-KO_Adult_V1"  # Adult example
#DATASET_NAME = "SST-cells_WT_Young_V1"          # Young example (will split by age)
DATASET_NAME = "PYR-SynGCaMP_WT_Adult_V1"      # Adult pyramidal
#DATASET_NAME = "PYR-PV-SynGCaMP_WT_Young_V1"   # Young pyramidal

# Age intervals for young animals (from Dataset_Organization.py)
AGE_INTERVALS = [(15, 19), (20, 23), (24, 27)]

# Automatically determine if this is a young dataset
IS_YOUNG = 'Young' in DATASET_NAME

# Base path (from Dataset_Organization.py)
BASE_PATH = os.path.expanduser('~/CURATED/Cibele/')
NWB_FOLDER = os.path.join(BASE_PATH, DATASET_NAME, 'NWBs')

print(f"\nDataset: {DATASET_NAME}")
print(f"NWB Folder: {NWB_FOLDER}")
print(f"Type: {'YOUNG (by age groups)' if IS_YOUNG else 'ADULT'}")

# ============================================================
# VISUALIZATION CONFIGURATION
# ============================================================

TARGET_PROTOCOL = "ff-gratings-8orientation-2contrasts-15repeats"
#TARGET_PROTOCOL = "ff-gratings-2orientations-8contrasts-15repeats"
# TARGET_PROTOCOL = ""  # <- empty string = no protocol filtering

COLOR = 'tab:orange'      # deconvolved fill color ('tab:purple' for KO, 'tab:red' for PV)
LINE_COLOR = 'tab:green'  # dF/F line color
ICONS = 'own'             # 'own' = draw grating icons here, 'physion' = try physion's renderer
N_STIM = 16               # number of stimuli shown
PAD = 2                   # seconds before first / after last stimulus
N_ROIS = 5                # ROIs shown by default (image uses 5 per block)
TAU = 1.5                 # deconvolution decay (s)
SMOOTHING = 1.5           # display smoothing (samples)
SCALE = 1                 # dF/F amplitude of the little black scale bars
SCORE_MODE = 'evoked'     # 'evoked' (deconvolved stim-modulation) or 'snr' (dF/F SNR)
ALPHA = 1.0               # fill alpha/transparency (1.0 = opaque/exact color, 0.5 = semi-transparent)

dFoF_parameters = dict(
    roi_to_neuropil_fluo_inclusion_factor=1.15,
    neuropil_correction_factor=0.7,
    method_for_F0='sliding_percentile',
    percentile=5.,        # percent
    sliding_window=5*60,  # seconds
)

# ------------------------------------------------------------
# 2) HELPERS
# ------------------------------------------------------------

def protocol_strings(data):
    """All strings in which the protocol name could live."""
    out = []
    protos = getattr(data, 'protocols', None)
    if protos is not None:
        if isinstance(protos, (list, tuple, np.ndarray)):
            out += [str(p) for p in protos]
        else:
            out.append(str(protos))
    out.append(str(getattr(data, 'metadata', '')))
    return out


def matches_protocol(data):
    if not TARGET_PROTOCOL:
        return True
    return any(TARGET_PROTOCOL.lower() in s.lower() for s in protocol_strings(data))


def prepare(fpath):
    data = Data(fpath, with_visual_stim=True, verbose=False)
    init_stim(data)
    data.build_dFoF(verbose=False, **dFoF_parameters)
    return data


def init_stim(data):
    """Works with physion versions that have / don't have init_visual_stim."""
    # In new physion (v2+), init_visual_stim was renamed to build_visual_stim
    if hasattr(data, 'build_visual_stim'):
        data.build_visual_stim(verbose=False)
    elif hasattr(data, 'init_visual_stim'):
        data.init_visual_stim()


def get_stim(data):
    """dict-like with time_start, time_stop, contrast, time_duration (arrays)."""
    vs = getattr(data, 'visual_stim', None)
    exp = getattr(vs, 'experiment', None)
    if exp is not None and 'time_start' in exp:
        return exp
    # fallback: read the stimulus table straight from the NWB file
    stim = data.nwbfile.stimulus
    def grab(*keys):
        for k in keys:
            if k in stim:
                return np.array(stim[k].data[:])
        return None
    ts = grab('time_start_realigned', 'time_start')
    te = grab('time_stop_realigned', 'time_stop')
    c = grab('contrast')
    if ts is None:
        raise RuntimeError('no stimulus times found; NWB stimulus keys: %s' % list(stim.keys()))
    if te is None:
        te = ts + 2.
    out = dict(time_start=ts, time_stop=te,
               contrast=c if c is not None else np.ones_like(ts),
               time_duration=te-ts)
    ang = grab('angle', 'orientation')
    if ang is not None:
        out['angle'] = ang
    return out


def get_dff(data):
    """dF/F as (nROIs, time)."""
    dff = np.asarray(data.dFoF)
    nR = getattr(data, 'nROIs', None)
    if nR is not None and dff.shape[0] != nR and dff.shape[1] == nR:
        dff = dff.T
    return dff


def get_dcnv(data, tau=TAU):
    dff = get_dff(data)
    return oasis(dff, dff.shape[0], tau, 1./data.CaImaging_dt)


def stim_window(data, n_stim=N_STIM, pad=PAD):
    exp = get_stim(data)
    ts = np.array(exp['time_start'])
    dur = exp['time_duration'][0]
    n = min(n_stim, len(ts))
    return [float(ts[0]-pad), float(ts[n-1]+dur+pad)]


def build_contrast_trace(data):
    contrast = 0*data.t_dFoF
    exp = get_stim(data)
    for tstart, tstop, c in zip(exp['time_start'], exp['time_stop'], exp['contrast']):
        contrast[(data.t_dFoF > tstart) & (data.t_dFoF < tstop)] = c
    return contrast


def rank_rois(data, tlim):
    t = data.t_dFoF
    win = (t > tlim[0]) & (t < tlim[1])
    dff = get_dff(data)[:, win]

    if SCORE_MODE == 'snr':
        smooth = gaussian_filter1d(dff, 2, axis=1)
        score = smooth.std(axis=1)/((dff-smooth).std(axis=1)+0.05)
    else:  # 'evoked': deconvolved activity during stimuli vs. outside
        dcnv = get_dcnv(data)[:, win]
        exp = get_stim(data)
        in_stim = np.zeros(t.shape, dtype=bool)
        for ts, te in zip(exp['time_start'], exp['time_stop']):
            in_stim |= (t > ts) & (t < te)
        in_stim = in_stim[win]
        score = (dcnv[:, in_stim].mean(axis=1) - dcnv[:, ~in_stim].mean(axis=1)) \
                / (dcnv.std(axis=1) + 1e-6)

    score = np.nan_to_num(score, nan=0.0)
    return np.argsort(score)[::-1], score


# ------------------------------------------------------------
# 3) THE FIGURE (deconvolved fill + dF/F line, stim bands, contrast bar, icons)
# ------------------------------------------------------------

def draw_stim_icons(data, tlim, ax_s, fig):
    """Grating icons above the traces (stripes oriented by 'angle', shaded by contrast)."""
    ax_s.set_xlim(tlim); ax_s.set_ylim(0, 1); ax_s.axis('off')
    if ICONS == 'physion':
        try:
            plot_raw(data, tlim=tlim,
                     settings={'VisualStim': {'fig_fraction': 0.5, 'color': 'black',
                                              'with_screen_inset': True}},
                     ax=ax_s)
            return
        except Exception as e:
            print('[icons] physion renderer failed (%r) -> drawing icons myself' % (e,))
            ax_s.cla(); ax_s.set_xlim(tlim); ax_s.set_ylim(0, 1); ax_s.axis('off')

    exp = get_stim(data)
    ts, dur = np.array(exp['time_start']), np.array(exp['time_duration'])
    ctr = np.array(exp['contrast']) if 'contrast' in exp else np.ones_like(ts)
    ang = None
    for k in ('angle', 'orientation'):
        if k in exp:
            ang = np.array(exp[k]); break

    pos = ax_s.get_position()
    W_in = fig.get_size_inches()[0]*pos.width
    H_in = fig.get_size_inches()[1]*pos.height
    dt = tlim[1]-tlim[0]
    for i, (t0, d) in enumerate(zip(ts, dur)):
        if not (tlim[0] <= t0 <= tlim[1]):
            continue
        s_in = min(0.92*H_in, 0.9*d/dt*W_in)          # square icon, in inches
        w, h = s_in/W_in*dt, s_in/H_in                # ... converted to data units
        xc, yc = t0+d/2., 0.5
        rect = Rectangle((xc-w/2, yc-h/2), w, h, transform=ax_s.transData,
                         fc='white', ec='k', lw=0.3)
        ax_s.add_patch(rect)
        if ang is None:
            continue
        th = np.deg2rad(float(ang[i])+90.)            # stripes perpendicular to drift direction
        dx, dy = np.cos(th), np.sin(th)
        gray = str(float(np.clip(1-ctr[i], 0, 1)))
        for o in np.linspace(-1, 1, 6):
            ux, uy = -dy*o, dx*o
            line, = ax_s.plot([xc+(ux-2*dx)*w/2, xc+(ux+2*dx)*w/2],
                              [yc+(uy-2*dy)*h/2, yc+(uy+2*dy)*h/2],
                              color=gray, lw=0.8, solid_capstyle='butt')
            line.set_clip_path(rect)
    ax_s.set_xlim(tlim); ax_s.set_ylim(0, 1)


def plot_traces(groups, tlim, figsize=(7, 4), savepath=None):
    """
    groups: list of dict(data=Data, rois=[...], color='tab:orange', scale=1, dcnv=None)
    Several groups are stacked (like PV / SST in panel A).
    """
    fig, (ax_s, ax) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                   gridspec_kw=dict(height_ratios=[1, 8], hspace=0.02))
    first, row = None, 0

    for g in groups:
        data = g['data']
        first = first or data
        dcnv_all = g.get('dcnv')
        if dcnv_all is None:
            dcnv_all = get_dcnv(data)
        dff_all = get_dff(data)
        cond = (data.t_dFoF > tlim[0]) & (data.t_dFoF < tlim[1])
        t = data.t_dFoF[cond]
        row0 = row

        for roi in list(g['rois'])[::-1]:   # first ROI listed = top row
            # deconvolved: filled, colored
            d = gaussian_filter1d(dcnv_all[roi, cond], SMOOTHING)
            d = d/(np.max(d)+1e-12)
            ax.fill_between(t, row+0*d, row+d, lw=0.1, alpha=g.get('alpha', ALPHA), color=g['color'])
            # dF/F: green line
            f = gaussian_filter1d(dff_all[roi, cond], SMOOTHING)
            amp = (f.max()-f.min())+1e-12
            f = (f-f.min())/amp
            ax.plot(t, row+f, color=g.get('line_color', LINE_COLOR), lw=0.5)
            # per-ROI scale bar
            ax.plot([tlim[0]-1]*2, [row, row+g.get('scale', SCALE)/amp], 'k-', lw=0.5)
            row += 1

        ax.text(tlim[0]-3, (row0+row)/2., '%s\u0394F/F' % g.get('scale', SCALE),
                rotation=90, ha='center', va='center', fontsize=6)
        row += 1  # gap between groups

    # contrast bar + grey stimulus bands (from the first session)
    contrast = build_contrast_trace(first)
    cond = (first.t_dFoF > tlim[0]) & (first.t_dFoF < tlim[1])
    ax.fill_between(first.t_dFoF[cond], -2+0*contrast[cond], -2+contrast[cond],
                    color='tab:grey', lw=0.1)
    exp = get_stim(first)
    dur = exp['time_duration'][0]
    for ts in np.array(exp['time_start']):
        if tlim[0] < ts < tlim[1]:
            ax.fill_between([ts, ts+dur], [-2, -2], [row, row], color='k', alpha=0.1, lw=0)

    pt.draw_bar_scales(ax, Xbar=5, Xbar_label='5s', Ybar=1e-12, fontsize=6, lw=0.2)
    pt.set_plot(ax, [])

    # grating icons strip on top (physion's own VisualStim renderer)
    draw_stim_icons(first, tlim, ax_s, fig)
    ax.set_xlim(tlim)

    if savepath:
        fig.savefig(savepath, format=os.path.splitext(savepath)[1][1:] or 'svg',
                    dpi=300, bbox_inches='tight', facecolor='white')
    return fig


# ------------------------------------------------------------
# 4) LEADERBOARD FOR SINGLE FOLDER WITH AGE FILTERING
# ------------------------------------------------------------

def build_leaderboard_from_scan(folder_path, age_interval=None, group_name=""):
    """
    Build leaderboard by scanning folder and optionally filtering by age.
    
    Args:
        folder_path: Path to NWB folder
        age_interval: Tuple of (min_age, max_age) or None for all ages
        group_name: Label for this group (e.g., 'P15-19')
    """
    # Get all NWB files in the folder
    nwb_files = sorted(glob.glob(os.path.join(folder_path, "*.nwb")))
    
    if not nwb_files:
        print(f"  [WARNING] No NWB files found in {folder_path}")
        return [], 0
    
    print(f"  Found {len(nwb_files)} NWB files in {folder_path}")
    
    # If age filtering is requested, we need to load metadata to get ages
    if age_interval is not None:
        min_age, max_age = age_interval
        print(f"  Filtering by age range: {age_interval}")
        filtered_files = []
        for fpath in nwb_files:
            try:
                data = Data(fpath, metadata_only=True, verbose=False)
                age = getattr(data, 'age', -1)
                if min_age <= age <= max_age and age >= 0:
                    filtered_files.append(fpath)
                    print(f"    {os.path.basename(fpath)}: age={age} ✓")
                else:
                    print(f"    {os.path.basename(fpath)}: age={age} ✗")
            except Exception as e:
                print(f"    {os.path.basename(fpath)}: error reading age - {e}")
        nwb_files = filtered_files
        print(f"  Filtered to {len(nwb_files)} files in age range {age_interval}")
    
    print(f"  Screening for protocol: '{TARGET_PROTOCOL}' (score = {SCORE_MODE})")

    leaderboard, skipped_protocol, diag = [], 0, None

    for fpath in nwb_files:
        fname = os.path.basename(fpath)
        try:
            # Load only metadata first to check protocol
            data = Data(fpath, metadata_only=True, verbose=False)
            if diag is None:
                diag = (fname, protocol_strings(data))
            
            # Check if protocol matches
            if not matches_protocol(data):
                skipped_protocol += 1
                continue
            
            # Now load full data for ranking
            data = Data(fpath, with_visual_stim=True, verbose=False)
            init_stim(data)
            data.build_dFoF(verbose=False, **dFoF_parameters)
            tlim = stim_window(data)
            ranked, scores = rank_rois(data, tlim)
            
            age = getattr(data, 'age', -1)
            leaderboard.append(dict(
                file_path=fpath, 
                file_name=fname,
                top_score=float(scores[ranked[0]]),
                sorted_rois=ranked, 
                tlim_bounds=tlim,
                group=group_name,
                age=age
            ))
        except Exception as e:
            print(f"  !! {fname}: {e!r}")

    leaderboard.sort(key=lambda x: x['top_score'], reverse=True)
    print(f"  -> ranked {len(leaderboard)} files, {skipped_protocol} skipped by protocol filter.")
    
    if not leaderboard and diag is not None:
        print("\n  [DIAGNOSTIC] Nothing matched. What physion sees in the first file:")
        print("     file     :", diag[0])
        for s in diag[1]:
            print("     protocol :", s[:400])
        print("     -> adapt TARGET_PROTOCOL to one of the names above (or set it to '').")
    
    return leaderboard, skipped_protocol


# ------------------------------------------------------------
# 5) MAIN: FLEXIBLE ANALYSIS FOR ADULT OR YOUNG (BY AGE GROUP)
# ------------------------------------------------------------

def to_color(name):
    name = name.strip()
    if 'tab:'+name in mcolors.TABLEAU_COLORS:
        return 'tab:'+name
    if mcolors.is_color_like(name):
        return name
    raise ValueError(name)


def run_interactive_selector(leaderboard, dataset_label=""):
    """Interactive selector for a single leaderboard (one folder/age group)."""
    if not leaderboard:
        print(f"\n[WARNING] No valid files for dataset: '{dataset_label}'")
        return

    while True:
        plt.close('all')
        print("\n" + "="*60)
        label_text = f" {dataset_label}" if dataset_label else ""
        print("AVAILABLE SESSIONS LEADERBOARD" + label_text)
        print("="*60)
        for idx, item in enumerate(leaderboard[:10]):
            age_str = f" (Age: {item.get('age', 'N/A')})" if 'age' in item else ""
            print(f"  [{idx + 1}] {item['file_name']}{age_str} (Score: {item['top_score']:.2f})")
        print("  [Q] Quit")
        print("="*60)

        file_choice = input("\n>> Session index to load (1-10) [default 1]: ").strip()
        if file_choice.lower() == 'q':
            print("Exiting.")
            return
        try:
            champion = leaderboard[int(file_choice)-1 if file_choice else 0]
        except (ValueError, IndexError):
            print("[Error] Invalid option.")
            continue

        print(f"\nLoading {champion['file_name']} ...")
        data = prepare(champion['file_path'])
        dcnv = get_dcnv(data)
        tlim = champion['tlim_bounds']
        ranked = [int(r) for r in champion['sorted_rois']]
        color, line_color, alpha, n_rois = COLOR, LINE_COLOR, ALPHA, N_ROIS
        current_rois = ranked[:n_rois]

        while True:
            print(f"\nRanking (best first): {ranked[:15]} ...")
            print(f"Window {np.round(tlim, 1)} | {len(current_rois)} traces | ROIs (top row first): {current_rois}")
            print(f"fill color: {color} | dF/F line color: {line_color} | alpha: {alpha}")
            plt.close('all')
            fig = plot_traces([dict(data=data, rois=current_rois, color=color,
                                    line_color=line_color, alpha=alpha, scale=SCALE, dcnv=dcnv)], tlim)
            plt.show(block=False)
            plt.pause(0.5)

            print("\nOptions:")
            print("  [ENTER]     save this figure")
            print("  n 8         show the 8 best-ranked ROIs (any number)")
            print("  4,12,0,8    show these specific ROIs (first = top row)")
            print("  c purple    deconvolved fill color (name, tab:xxx or #hex)")
            print("  l black     dF/F line color")
            print("  a 0.8       fill alpha/transparency (0.0-1.0, default=1.0)")
            print("  b           back to the session list")
            user_choice = input(">> Your choice: ").strip()
            cmd = user_choice.lower()

            try:
                if cmd == 'b':
                    break
                elif not cmd:
                    out_dir = os.path.join(os.path.expanduser('~'), 'Desktop', 'Final-Figures-PhD')
                    os.makedirs(out_dir, exist_ok=True)
                    # Include age group in filename if present
                    label = champion.get('group', '')
                    age = champion.get('age', '')
                    age_str = f"_Age{age}" if age >= 0 else ""
                    suffix = f"_{label}{age_str}" if label or age >= 0 else ""
                    out = os.path.join(out_dir, f"TRACES_DECONV_{DATASET_NAME}{suffix}_{os.path.splitext(champion['file_name'])[0]}.svg")
                    fig.savefig(out, format='svg', dpi=300, bbox_inches='tight', facecolor='white')
                    print(f"\nSaved: {out}")
                    plt.close('all')
                    return
                elif cmd.startswith('n '):
                    n_rois = int(cmd[2:])
                    current_rois = ranked[:n_rois]
                elif cmd.startswith('c '):
                    color = to_color(user_choice[2:])
                elif cmd.startswith('l '):
                    line_color = to_color(user_choice[2:])
                elif cmd.startswith('a '):
                    alpha = float(user_choice[2:])
                    alpha = max(0.0, min(1.0, alpha))  # Clamp between 0 and 1
                    print(f"  Alpha set to: {alpha}")
                else:
                    current_rois = [int(x.strip()) for x in user_choice.split(',')]
            except ValueError as e:
                print(f"[Error] could not parse '{user_choice}' ({e}). Examples: 'n 8' | 'c purple' | '1,5,2,9'")


# ============================================================
# MAIN EXECUTION
# ============================================================

print("\n" + "="*70)
print(" TRACES PLOT - FLEXIBLE DATASET ANALYSIS")
print("="*70)
print(f"Dataset: {DATASET_NAME}")
print(f"NWB Folder: {NWB_FOLDER}")
print(f"Type: {'YOUNG (by age groups)' if IS_YOUNG else 'ADULT'}")
print(f"Protocol: {TARGET_PROTOCOL}")
print("="*70 + "\n")

if not os.path.isdir(NWB_FOLDER):
    print(f"[ERROR] NWB folder does not exist: {NWB_FOLDER}")
    print("Available folders in CURATED/Cibele:")
    for item in os.listdir(BASE_PATH):
        print(f"  - {item}")
    sys.exit(1)

if IS_YOUNG:
    # ============================================================
    # YOUNG DATASET: Split by age groups and process each separately
    # ============================================================
    print("[YOUNG MODE] Will split by age groups and process each separately\n")
    
    # Build leaderboards for each age group
    age_group_leaderboards = {}
    
    for interval in AGE_INTERVALS:
        group_name = f"P{interval[0]}-{interval[1]}"
        print(f"\n{'='*70}")
        print(f"Processing age group: {group_name}")
        print(f"{'='*70}")
        leaderboard, _ = build_leaderboard_from_scan(NWB_FOLDER, interval, group_name)
        age_group_leaderboards[group_name] = leaderboard
    
    # Run interactive selector for each age group
    for group_name in sorted(age_group_leaderboards.keys()):
        leaderboard = age_group_leaderboards[group_name]
        if not leaderboard:
            print(f"\n[WARNING] No valid files for age group: {group_name}. Skipping.")
            continue
        
        print(f"\n{'='*70}")
        print(f"ANALYZING AGE GROUP: {group_name}")
        print(f"{'='*70}")
        run_interactive_selector(leaderboard, f"({group_name})")
        
        # Ask if user wants to continue to next age group
        if len(age_group_leaderboards) > 1:
            cont = input(f"\nContinue to next age group? (Y/n): ").strip().lower()
            if cont == 'n':
                break
    
    print("\n[YOUNG MODE] Analysis complete for all age groups.")

else:
    # ============================================================
    # ADULT DATASET: Process single folder
    # ============================================================
    print("[ADULT MODE] Processing single folder...\n")
    
    leaderboard, _ = build_leaderboard_from_scan(NWB_FOLDER, None, DATASET_NAME)
    run_interactive_selector(leaderboard, f"({DATASET_NAME})")

print("\nDone!")









#%%
######################################################################
# Combined selected NWBs, ROIs, and colors for plotting
# Format: list of dicts with nwb_file, rois, color for each group
# Protocol: 2orientations-8contrasts-15repeats
######################################################################

selected_combinations = [
    {
        "group": "SST-cre P20-23",
        "nwb_file": "2025_07_18-16-09-54",
        "rois": [2, 3],
        "color": "#99b993ff"
    },
    {
        "group": "SST-cre adult",
        "nwb_file": "2024_12_26-15-44-29",
        "rois": [15, 20],
        "color": "#12522eff"
    },
    {
        "group": "PV-cre P20-23",
        "nwb_file": "2024_12_11-14-08-03",
        "rois": [25, 21],
        "color": "#ff9ab5ff"
    },
    {
        "group": "PV-cre adult",
        "nwb_file": "2024_04_24-15-46-52",
        "rois": [11, 13],
        "color": "#b30a7bff"
    },
    {
        "group": "Pyramidal P20-23",
        "nwb_file": "2026_04_30-11-41-31",
        "rois": [79, 28],
        "color": "#6caaccff"
    },
    {
        "group": "Pyramidal adult",
        "nwb_file": "2025_10_29-14-39-31",
        "rois": [9, 13],
        "color": "#005d8aff"
    }
]
# Plot traces for selected combinations
# Search for NWB files across all relevant dataset folders and plot them together
# Using protocol: ff-gratings-2orientations-8contrasts-15repeats

# Define dataset folders to search in for the selected NWB files
# Note: P20-23 files are in Young datasets, adults in Adult datasets
dataset_folders = [
    "SST-cells_WT_Young_V1",         # SST-cre P20-23
    "SST-cells_WT_Adult_V1",        # SST-cre adult
    "PV-cells_WT_Young_V1",         # PV-cre P20-23
    "PV-cells_WT_Adult_V1",         # PV-cre adult
    "PYR-SynGCaMP_WT_Young_V1",      # Pyramidal P20-23
    "PYR-SynGCaMP_WT_Adult_V1",      # Pyramidal adult
    "PYR-PV-SynGCaMP_WT_Young_V1",   # Pyramidal P20-23 (alternative folder)
]

# Find the NWB files
groups = []
base_path = os.path.expanduser('~/CURATED/Cibele/')

for combo in selected_combinations:
    nwb_file = combo["nwb_file"] + ".nwb"
    found_path = None
    
    for dataset in dataset_folders:
        folder = os.path.join(base_path, dataset, 'NWBs')
        candidate = os.path.join(folder, nwb_file)
        if os.path.exists(candidate):
            found_path = candidate
            break
    
    # If not found in the standard folders, search all dataset folders
    if found_path is None:
        all_folders = glob.glob(os.path.join(base_path, '*', 'NWBs'))
        for folder in all_folders:
            candidate = os.path.join(folder, nwb_file)
            if os.path.exists(candidate):
                found_path = candidate
                break
    
    # Debug: if still not found, list all NWB files that exist
    if found_path is None:
        print(f"\n[DEBUG] Could not find {nwb_file}")
        print(f"[DEBUG] Searched folders:")
        for dataset in dataset_folders:
            folder = os.path.join(base_path, dataset, 'NWBs')
            print(f"  - {folder}")
        all_folders = glob.glob(os.path.join(base_path, '*', 'NWBs'))
        for folder in all_folders:
            print(f"  - {folder}")
        print(f"[DEBUG] All NWB files found in CURATED/Cibele:")
        for folder in all_folders:
            files = glob.glob(os.path.join(folder, '*.nwb'))
            for f in files:
                print(f"  - {f}")
    
    if found_path:
        print(f"Found {nwb_file} in {found_path}")
        data = prepare(found_path)
        dcnv = get_dcnv(data)
        tlim = stim_window(data)
        
        groups.append({
            'data': data,
            'rois': combo["rois"],
            'color': combo["color"],
            'line_color': LINE_COLOR,
            'alpha': ALPHA,
            'scale': SCALE,
            'dcnv': dcnv,
            'label': combo["group"]
        })
    else:
        print(f"WARNING: Could not find {nwb_file} in any dataset folder")

if groups:
    # Use the tlim from the first group as reference
    first_tlim = stim_window(groups[0]['data'])
    
    # Save path for SVG
    out_dir = os.path.join(os.path.expanduser('~'), 'Desktop', 'Final-Figures-PhD')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'TRACES_COMBINED_selected_combinations.svg')
    
    # Reverse groups to plot Pyramidal at top, PV-cre middle, PV-cre NR1 KO at bottom
    # (matplotlib plots with y=0 at bottom, so first group goes to bottom by default)
    groups = groups[::-1]
    
    # Plot and save
    fig = plot_traces(groups, first_tlim, figsize=(10, 8), savepath=out_path)
    plt.show()
    print(f"\nFigure saved as: {out_path}")
else:
    print("No valid groups to plot")











#%%
######################################################################
# REST vs RUN (1/3): CONFIG + HELPERS -> run this cell first
# trial-averaged dF/F per stimulus, rest and run traces overlaid
# - episodes classified by the mean running speed during the stimulus
# - NWBs taken from the dataset folder (same scheme as the cells above)
# - then: cell 2/3 = one ROI example, cell 3/3 = all sessions
# self-contained: does not need the cells above
######################################################################

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.append('../physion/src')
from physion.analysis.read_NWB import Data
from physion.analysis.episodes.build import EpisodeData

plt.rcParams['svg.fonttype'] = 'none'

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

#DATASET_NAME = "PYR-SynGCaMP_WT_Adult_V1"
#DATASET_NAME = "PV-cells_WT_Adult_V1"
DATASET_NAME = "SST-cells_WT_Adult_V1"

BASE_PATH = os.path.expanduser('~/CURATED/Cibele/')
NWB_FOLDER = os.path.join(BASE_PATH, DATASET_NAME, 'NWBs')

TARGET_PROTOCOL = "ff-gratings-8orientation-2contrasts-15repeats"
CONTRASTS = None             # None = all contrasts of the protocol (one panel per contrast x angle)

SESSIONS = []                # e.g. ["2025_10_29-14-39-31"]; empty = all sessions of the folder
ROIS = None                  # None = all ROIs of the session, or a list of ROI indices

# figure modes:
#  'best-roi' : single ROI with the largest pref-orth response (like a single-cell example), absolute angles
#  'aligned'  : mean over ROIs after aligning each ROI to its preferred orientation (center panel = preferred)
#  'mean'     : plain mean over ROIs, absolute angles (tuning washes out: ROIs prefer different angles)
BEST_ROI_RANK = 0            # 'best-roi' mode: 0 = most selective ROI, 1 = second, ...

RUN_SPEED_THRESHOLD = 0.1    # cm/s, mean speed during the stimulus (0.1 = physion's behavior default)
MIN_EPISODES = 2             # min episodes per state and angle to draw a trace
BASELINE_SUBTRACT = False    # True = subtract the pre-stimulus mean of each episode

REST_COLOR = 'k'
RUN_COLOR = 'tab:orange'
STIM_COLOR = '#f2e4ea'       # pink band during the stimulus
SCALE = 0.5                  # dF/F of the vertical scale bar
WITH_GRAND_AVERAGE = True    # extra figure: mean of the session means

OUT_DIR = os.path.join(os.path.expanduser('~'), 'Desktop', 'Final-Figures-PhD', 'REST-vs-RUN')

if DATASET_NAME[:4] == 'PYR-':
    roi_to_neuropil_factor = 0.   # pyramidal cells (see analysis/Preprocessing_Settings.py)
else:
    roi_to_neuropil_factor = 1.15 # interneurons

dFoF_parameters = dict(
    roi_to_neuropil_fluo_inclusion_factor=roi_to_neuropil_factor,
    neuropil_correction_factor=0.7,
    method_for_F0='sliding_percentile',
    percentile=5.,
    sliding_window=5*60,
)

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def find_session_files(folder, sessions=SESSIONS):
    files = sorted(glob.glob(os.path.join(folder, '*.nwb')))
    if sessions:
        names = [s.replace('.nwb', '') for s in sessions]
        files = [f for f in files if os.path.basename(f).replace('.nwb', '') in names]
    out = []
    for f in files:
        try:
            data = Data(f, metadata_only=True, verbose=False)
            if any(TARGET_PROTOCOL in p for p in data.protocols):
                out.append(f)
        except Exception as e:
            print(f"  !! {os.path.basename(f)}: {e!r}")
    print(f"{len(out)} sessions with protocol '{TARGET_PROTOCOL}' in {folder}")
    return out


def episode_speeds(data, Episodes):
    """Mean running speed of each episode during the stimulus window."""
    data.build_running()
    speeds = np.zeros(len(Episodes.time_start_realigned))
    for i, (t0, d) in enumerate(zip(Episodes.time_start_realigned, Episodes.time_duration)):
        cond = (data.t_running >= t0) & (data.t_running < t0+d)
        speeds[i] = data.running[cond].mean() if np.any(cond) else np.nan
    return speeds


def orientation_tuning(Episodes, dff, angles):
    """
    evoked response (stim mean - prestim mean) per ROI and angle, all episodes
    at the highest contrast -> preferred angle index, OSI and selectivity score of each ROI
    score = r_pref - r_orth (dF/F): strong AND selective (OSI alone favors weak noisy ROIs)
    """
    stim = (Episodes.t > 0) & (Episodes.t < Episodes.time_duration[0])
    evoked = dff[:, :, stim].mean(axis=2) - dff[:, :, Episodes.t < 0].mean(axis=2)  # (episodes, rois)
    c_max = np.isclose(Episodes.contrast, Episodes.contrast.max())
    R = np.array([evoked[c_max & np.isclose(Episodes.angle, a)].mean(axis=0) for a in angles]).T
    n = len(angles)
    pref = R.argmax(axis=1)
    r_pref = np.clip(R[np.arange(len(R)), pref], 0, None)
    r_orth = np.clip(R[np.arange(len(R)), (pref+n//2) % n], 0, None)   # +90 deg
    osi = (r_pref-r_orth)/(r_pref+r_orth+1e-12)
    osi[r_pref <= 0] = 0                                               # non-responsive ROIs
    return pref, osi, r_pref-r_orth


def load_session(fpath):
    """Loads one NWB once: episodes dF/F, running state and ROI tuning (reused by all modes)."""
    data = Data(fpath, verbose=False)
    if not data.has_running():
        print(f"  [X] {os.path.basename(fpath)}: no running speed in this NWB")
        return None
    data.build_dFoF(verbose=False, **dFoF_parameters)
    protocol = [p for p in data.protocols if TARGET_PROTOCOL in p][0]
    Episodes = EpisodeData(data, quantities=['dFoF'], protocol_name=protocol, verbose=False)

    rois = np.arange(data.nROIs) if ROIS is None else np.array(ROIS)
    dff = Episodes.dFoF[:, rois, :]                          # (episodes, rois, time)
    if BASELINE_SUBTRACT:
        dff = dff - dff[:, :, Episodes.t < 0].mean(axis=2, keepdims=True)

    speeds = episode_speeds(data, Episodes)
    running = speeds > RUN_SPEED_THRESHOLD
    angles = np.unique(Episodes.angle)
    pref, osi, score = orientation_tuning(Episodes, dff, angles)
    return dict(name=os.path.basename(fpath).replace('.nwb', ''),
                Episodes=Episodes, dff=dff, rois=rois, running=running,
                angles=angles, pref=pref, osi=osi, score=score)


def session_responses(sess, mode, roi=None):
    """
    returns dict(t, conds, rest=[...], run=[...], n_rest, n_run, ...)
    conds: list of (contrast, angle), contrast first then angle
           (in 'aligned' mode, angle = orientation relative to the preferred one)
    rest/run: per condition, array (n_samples, time) or None
              samples = episodes ('best-roi', 'mean') or ROIs ('aligned')
    roi: ROI index for 'best-roi' mode (None = best ROI of rank BEST_ROI_RANK)
    """
    Episodes, dff, rois, running = sess['Episodes'], sess['dff'], sess['rois'], sess['running']
    angles, pref, osi = sess['angles'], sess['pref'], sess['osi']
    n = len(angles)
    contrasts = np.unique(Episodes.contrast) if CONTRASTS is None else np.array(CONTRASTS)

    out = dict(name=sess['name'], mode=mode, t=Episodes.t, nROIs=len(rois),
               duration=float(Episodes.time_duration[0]),
               rest=[], run=[], n_rest=int(np.sum(~running)), n_run=int(np.sum(running)))

    if mode == 'aligned':
        # relative orientations, preferred one in the center panel
        shifts = np.arange(n) - n//2
        out['conds'] = [(c, s*(angles[1]-angles[0])) for c in contrasts for s in shifts]
        for c in contrasts:
            for s in shifts:
                for state, s_cond in [('rest', ~running), ('run', running)]:
                    traces = []
                    for j in range(len(rois)):
                        cond = s_cond & np.isclose(Episodes.contrast, c) & \
                                np.isclose(Episodes.angle, angles[(pref[j]+s) % n])
                        if np.sum(cond) >= MIN_EPISODES:
                            traces.append(dff[cond, j, :].mean(axis=0))
                    out[state].append(np.array(traces) if traces else None)
        out['label'] = f"{len(rois)} ROIs aligned to preferred orientation"
    else:
        if mode == 'best-roi':
            if roi is None:
                j = np.argsort(sess['score'])[::-1][BEST_ROI_RANK]
            else:
                j = int(np.where(rois == roi)[0][0])
            resp = dff[:, j, :]
            out['label'] = f"ROI {rois[j]}: OSI {osi[j]:.2f}"
        else:
            resp = dff.mean(axis=1)
            out['label'] = f"mean of {len(rois)} ROIs"
        out['conds'] = [(c, a) for c in contrasts for a in angles]
        for c, a in out['conds']:
            cond = np.isclose(Episodes.contrast, c) & np.isclose(Episodes.angle, a)
            out['rest'].append(resp[cond & ~running] if np.sum(cond & ~running) >= MIN_EPISODES else None)
            out['run'].append(resp[cond & running] if np.sum(cond & running) >= MIN_EPISODES else None)

    print(f"  [v] {out['name']}: {out['label']} | rest n={out['n_rest']} | run n={out['n_run']} episodes")
    return out


def best_rois_table(sess, n=10):
    """Prints the ROIs ranked by selectivity score (to choose an example ROI)."""
    print(f"\n{sess['name']}: best ROIs (score = pref-orth dF/F, at highest contrast)")
    for j in np.argsort(sess['score'])[::-1][:n]:
        print(f"  ROI {sess['rois'][j]:4d} | pref {sess['angles'][sess['pref'][j]]:6.1f} deg"
              f" | OSI {sess['osi'][j]:.2f} | score {sess['score'][j]:.3f}")


def draw_grating_icon(ax, angle, contrast=1.0):
    """Small static-grating square, stripes darker for higher contrast."""
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect('equal'); ax.axis('off')
    box = Rectangle((-1, -1), 2, 2, fc='white', ec='0.5', lw=0.3)
    ax.add_patch(box)
    th = np.deg2rad(angle)
    dx, dy = np.cos(th), np.sin(th)
    gray = str(float(np.clip(1-contrast, 0, 0.9)))
    for o in np.linspace(-1.4, 1.4, 8):          # stripes perpendicular to the angle direction
        line, = ax.plot([o*dx-1.5*dy, o*dx+1.5*dy], [o*dy+1.5*dx, o*dy-1.5*dx],
                        color=gray, lw=0.6, solid_capstyle='butt')
        line.set_clip_path(box)


def mean_sem(x):
    return x.mean(axis=0), x.std(axis=0)/np.sqrt(len(x))


def plot_rest_vs_run(resp, title, savepath=None):
    t, conds = resp['t'], resp['conds']
    n = len(conds)
    fig, AX = plt.subplots(2, n+1, figsize=(0.8*(n+1), 2.),
                           gridspec_kw=dict(height_ratios=[4, 1], wspace=0.05, hspace=0.05))

    # common y range across panels
    lo, hi = np.inf, -np.inf
    for state in ['rest', 'run']:
        for x in resp[state]:
            if x is not None:
                m, s = mean_sem(x)
                lo, hi = min(lo, (m-s).min()), max(hi, (m+s).max())
    if not np.isfinite(lo):
        print(f"  [X] {title}: not enough episodes to plot")
        plt.close(fig)
        return None
    pad = 0.05*(hi-lo)
    ylim = (lo-pad, hi+pad)

    for i, (c, a) in enumerate(conds):
        ax = AX[0, i]
        ax.axvspan(0, resp['duration'], color=STIM_COLOR, lw=0)
        for state, color in [('rest', REST_COLOR), ('run', RUN_COLOR)]:
            x = resp[state][i]
            if x is None:
                continue
            m, s = mean_sem(x)
            ax.fill_between(t, m-s, m+s, color=color, alpha=0.3, lw=0)
            ax.plot(t, m, color=color, lw=0.8)
        ax.set_ylim(ylim); ax.set_xlim(t[0], t[-1]); ax.axis('off')
        draw_grating_icon(AX[1, i], a, c)
        if i == 0 or c != conds[i-1][0]:
            AX[1, i].set_title('c=%g' % c, fontsize=6, loc='left', pad=1)
        if resp.get('mode') == 'aligned':
            AX[1, i].text(0, -1.3, '%+g' % a if a else 'pref.', ha='center', va='top', fontsize=5)

    # scale bars in the last column
    ax = AX[0, n]
    ax.set_ylim(ylim); ax.set_xlim(t[0], t[-1]); ax.axis('off')
    y0 = ylim[0]+0.1*(ylim[1]-ylim[0])
    ax.plot([t[0]]*2, [y0, y0+SCALE], 'k-', lw=1)
    ax.plot([t[0], t[0]+resp['duration']], [y0]*2, 'k-', lw=1)
    ax.text(t[0]+0.05*(t[-1]-t[0]), y0+SCALE/2, '%s ΔF/F' % SCALE,
            rotation=90, ha='left', va='center', fontsize=7)
    ax.text(t[0]+resp['duration']/2, y0, '%gs' % resp['duration'], ha='center', va='top', fontsize=7)
    AX[1, n].axis('off')

    AX[0, 0].set_title(title, loc='left', fontsize=8)
    AX[0, n].plot([], [], color=REST_COLOR, label='rest (n=%i)' % resp['n_rest'])
    AX[0, n].plot([], [], color=RUN_COLOR, label='run (n=%i)' % resp['n_run'])
    AX[0, n].legend(frameon=False, fontsize=6, loc='upper left', bbox_to_anchor=(0, 1.15))

    if savepath:
        fig.savefig(savepath, format='svg', dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  saved: {savepath}")
    return fig


def grand_average(responses):
    """Mean across sessions of the session means (sessions treated as samples)."""
    ok = [r for r in responses if r is not None]
    conds = ok[0]['conds']
    out = dict(mode=ok[0]['mode'], t=ok[0]['t'], conds=conds, duration=ok[0]['duration'],
               rest=[], run=[], n_rest=0, n_run=0)
    for state in ['rest', 'run']:
        for i in range(len(conds)):
            means = [r[state][i].mean(axis=0) for r in ok
                     if np.allclose(r['conds'], conds) and r[state][i] is not None]
            out[state].append(np.array(means) if len(means) >= MIN_EPISODES else None)
        out['n_%s' % state] = max([0]+[len(x) for x in out[state] if x is not None])
    return out

os.makedirs(OUT_DIR, exist_ok=True)

#%%
######################################################################
# REST vs RUN (2/3): ONE ROI EXAMPLE (single session, single ROI)
# - EXAMPLE_ROI = None -> best ROI of the session; the ranking is printed to choose another one
######################################################################

EXAMPLE_SESSION = "2024_08_27-11-45-06"   # NWB filename (without .nwb) inside NWB_FOLDER
EXAMPLE_ROI = None                        # ROI index, or None = best ROI

example_file = os.path.join(NWB_FOLDER, EXAMPLE_SESSION + '.nwb')
example = load_session(example_file) if os.path.exists(example_file) else None
if not os.path.exists(example_file):
    print(f"[ERROR] {example_file} not found (check DATASET_NAME in cell 1/3)")
if example is not None:
    best_rois_table(example)
    resp = session_responses(example, 'best-roi', roi=EXAMPLE_ROI)
    out = os.path.join(OUT_DIR, f"REST-vs-RUN_{DATASET_NAME}_example_{resp['name']}_{resp['label'].split(':')[0].replace(' ', '')}.svg")
    plot_rest_vs_run(resp, f"{resp['name']} - {resp['label']}", savepath=out)
    plt.show()

#%%
######################################################################
# REST vs RUN (3/3): ALL SESSIONS OF THE FOLDER
# - for each session: one figure per mode in MODES
# - plus the grand average (mean of session means) per mode
######################################################################

MODES = ['best-roi', 'aligned']   # add 'mean' for the plain mean over ROIs

print(f"\nREST vs RUN | {DATASET_NAME} | {TARGET_PROTOCOL} | "
      f"run threshold={RUN_SPEED_THRESHOLD} cm/s | modes={MODES}")

responses = {mode: [] for mode in MODES}
for fpath in find_session_files(NWB_FOLDER):
    try:
        sess = load_session(fpath)
        if sess is None:
            continue
        for mode in MODES:
            resp = session_responses(sess, mode)
            responses[mode].append(resp)
            out = os.path.join(OUT_DIR, f"REST-vs-RUN_{DATASET_NAME}_{mode}_{resp['name']}.svg")
            plot_rest_vs_run(resp, f"{resp['name']} - {resp['label']}", savepath=out)
            plt.close('all')
    except Exception as e:
        print(f"  !! {os.path.basename(fpath)}: {e!r}")

if WITH_GRAND_AVERAGE:
    for mode in MODES:
        if not responses[mode]:
            continue
        ga = grand_average(responses[mode])
        out = os.path.join(OUT_DIR, f"REST-vs-RUN_{DATASET_NAME}_{mode}_grand-average.svg")
        fig = plot_rest_vs_run(ga, f"{DATASET_NAME} - {mode} - mean of sessions", savepath=out)
        if fig is not None:
            # in the grand average, n = number of sessions
            for txt in fig.axes[len(ga['conds'])].get_legend().get_texts():
                txt.set_text(txt.get_text().replace('n=', 'sessions='))
            fig.savefig(out, format='svg', dpi=300, bbox_inches='tight', facecolor='white')

plt.show()

#%%
