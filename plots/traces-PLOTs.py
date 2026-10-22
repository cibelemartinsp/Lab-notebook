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
