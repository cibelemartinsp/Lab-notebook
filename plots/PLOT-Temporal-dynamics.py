# %% [markdown]
# # Analyze Temporal Dynamics

# %%
import sys, os, tempfile
import numpy as np
from scipy import stats

sys.path.append('../physion/src') # add src code directory for physion

import physion.utils.plot_tools as pt
pt.set_style('dark')

summary_folder = os.path.expanduser('~/CURATED/Cibele/summary')




# %%
################################################################
###### Temporal dynamics multiple comparations #################
################## With statistics #############################
################################################################
# YOU NEED TO CHANGE THE LABLS AND COLORS TO MATCH YOUR DATASETS

import numpy as np
import os
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import physion.utils.plot_tools as pt

def compute_multi_group_stats_and_plot(
        keys,
        path,
        colors,
        group_labels=None,
        zoom_window=(0, 0.3),
        late_window=(0.8, 2.0),
        baseline_window=(-1, 0),
        peak_window=(0, 1),
        delay_window=(-0.01, 0.3),
        norm="min-max",
        figsize=(9, 4.5),
        savepath=None):

    if group_labels is None:
        group_labels = [f"Group {i+1}" for i in range(len(keys))]

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.2, 1], wspace=0.25)
    
    ax_main = fig.add_subplot(gs[0, 0])
    ax_zoom = fig.add_subplot(gs[0, 1])

    all_late_values = []
    all_delay_values = []

    # ====================================================
    # DATA PROCESSING & METRIC EXTRACTION
    # ====================================================
    for i, key in enumerate(keys):
        filepath = os.path.join(path, f"{key}.npy")
        if not os.path.exists(filepath):
            print(f"[warning] file not found: {filepath}")
            continue

        Responses = np.load(filepath, allow_pickle=True)
        if len(Responses) == 0:
            continue

        t = Responses[0]['t']

        # Gather significant ROIs per response
        Deconvolved_list = []
        for R in Responses:
            sig = R.get('significant', np.ones(R["Deconvolved"].shape[0], dtype=bool))
            d = R["Deconvolved"][sig] if sig.sum() > 0 else R["Deconvolved"]
            Deconvolved_list.append(d)
        
        Deconvolved_sessions = [np.mean(d, axis=0) for d in Deconvolved_list if len(d) > 0]
        Deconvolved_mat = np.vstack(Deconvolved_sessions)

        # Baseline subtraction & peak normalization per session trace
        bmask = (t >= baseline_window[0]) & (t < baseline_window[1])
        pmask = (t >= peak_window[0]) & (t <= peak_window[1])

        processed_traces = []
        for d in Deconvolved_sessions:
            d_base = d - d[bmask].mean()
            p_max = d_base[pmask].max()
            d_norm = d_base / p_max if p_max != 0 else d_base
            processed_traces.append(d_norm)
        processed_traces = np.array(processed_traces)

        # 1. Late-Window Mean Response per session/ROI
        l_mask = (t > late_window[0]) & (t < late_window[1])
        late_vals = np.array([np.mean(d[l_mask]) for d in processed_traces])
        all_late_values.append(late_vals)

        # 2. Delay to Peak per session/ROI
        d_mask = (t > delay_window[0]) & (t < delay_window[1])
        delay_vals = []
        for d in processed_traces:
            peak_idx = np.argmax(d[d_mask])
            delay_vals.append(t[d_mask][peak_idx])
        all_delay_values.append(np.array(delay_vals))

        # Plot traces
        mean_trace = Deconvolved_mat.mean(axis=0)
        sem_trace = stats.sem(Deconvolved_mat, axis=0)
        if norm == "min-max":
            base = mean_trace[bmask].mean()
            mean_trace = mean_trace - base
            denom = mean_trace.max() if mean_trace.max() != 0 else 1
            mean_trace = mean_trace / denom
            sem_trace = sem_trace / denom

        pt.plot(t, mean_trace, sy=sem_trace, color=colors[i], ax=ax_main, ms=2)
        
        z_mask = (t >= zoom_window[0]) & (t <= zoom_window[1])
        pt.plot(t[z_mask], mean_trace[z_mask], sy=sem_trace[z_mask], color=colors[i], ax=ax_zoom, ms=2)

    # ====================================================
    # MULTI-GROUP STATISTICAL TESTING (Kruskal-Wallis & Pairwise Mann-Whitney)
    # ====================================================
    print("\n" + "="*50)
    print("DETAILED MULTI-GROUP STATISTICAL RESULTS")
    print("="*50)

    p_val_l, p_val_d = 1.0, 1.0

    # Late Window Stats
    if len(all_late_values) > 1:
        print(f"\n--- Late-Window Response ({late_window[0]}s - {late_window[1]}s) ---")
        for idx, lbl in enumerate(group_labels[:len(all_late_values)]):
            print(f"  {lbl} Mean ± SEM: {np.mean(all_late_values[idx]):.3f} ± {stats.sem(all_late_values[idx]):.3f}")
        
        h_stat_l, p_val_l = stats.kruskal(*all_late_values)
        print(f"  Kruskal-Wallis H: {h_stat_l:.2f}, p-value: {p_val_l:.5f}")
        
        # Pairwise comparisons (Mann-Whitney U with Bonferroni correction) if more than 2 groups
        if len(all_late_values) > 2:
            print("  Pairwise Mann-Whitney U tests (Bonferroni corrected):")
            n_comparisons = len(all_late_values) * (len(all_late_values) - 1) / 2
            for i in range(len(all_late_values)):
                for j in range(i + 1, len(all_late_values)):
                    _, p_pair = stats.mannwhitneyu(all_late_values[i], all_late_values[j], alternative='two-sided')
                    p_corr = min(p_pair * n_comparisons, 1.0)
                    print(f"    {group_labels[i]} vs {group_labels[j]}: p = {p_corr:.5f} (raw p = {p_pair:.5f})")

    # Delay to Peak Stats
    if len(all_delay_values) > 1:
        print(f"\n--- Delay to Peak Response ({delay_window[0]}s - {delay_window[1]}s) ---")
        for idx, lbl in enumerate(group_labels[:len(all_delay_values)]):
            print(f"  {lbl} Mean ± SEM: {np.mean(all_delay_values[idx]):.3f} ± {stats.sem(all_delay_values[idx]):.3f}")
        
        h_stat_d, p_val_d = stats.kruskal(*all_delay_values)
        print(f"  Kruskal-Wallis H: {h_stat_d:.2f}, p-value: {p_val_d:.5f}")
        
        if len(all_delay_values) > 2:
            print("  Pairwise Mann-Whitney U tests (Bonferroni corrected):")
            n_comparisons = len(all_delay_values) * (len(all_delay_values) - 1) / 2
            for i in range(len(all_delay_values)):
                for j in range(i + 1, len(all_delay_values)):
                    _, p_pair = stats.mannwhitneyu(all_delay_values[i], all_delay_values[j], alternative='two-sided')
                    p_corr = min(p_pair * n_comparisons, 1.0)
                    print(f"    {group_labels[i]} vs {group_labels[j]}: p = {p_corr:.5f} (raw p = {p_pair:.5f})")
            
    print("="*50 + "\n")

    # Helper function for stars
    def p_to_stars(p):
        if p < 0.001: return "***"
        elif p < 0.01: return "**"
        elif p < 0.05: return "*"
        else: return "ns"

    global_late_star = p_to_stars(p_val_l)
    global_delay_star = p_to_stars(p_val_d)

    # ====================================================
    # STYLING & DYNAMIC ANNOTATIONS ON PLOT
    # ====================================================
    ax_main.set_xlabel("time from stim. (s)")
    ax_main.set_ylabel("deconv. resp.\n(peak norm.)")
    ax_main.set_xlim(-0.6, 2.8)
    ax_main.set_ylim(-0.1, 1.25)
    
    # Highlight late window & add global significance label
    ax_main.axvspan(late_window[0], late_window[1], color='gray', alpha=0.1)
    if global_late_star != "ns":
        ax_main.hlines(y=0.05, xmin=late_window[0], xmax=late_window[1], color='black', linewidth=1.5)
        ax_main.text(np.mean(late_window), 0.08, global_late_star, ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')
    else:
        ax_main.text(np.mean(late_window), 0.08, "ns", ha='center', va='bottom', fontsize=10, fontstyle='italic', color='black')

    # Zoom plot settings
    ax_zoom.set_xlim(zoom_window[0], zoom_window[1])
    ax_zoom.set_ylim(-0.1, 1.25)
    ax_zoom.set_xlabel("0   0.3")
    ax_zoom.yaxis.set_ticklabels([])

    if global_delay_star != "ns":
        ax_zoom.hlines(y=1.12, xmin=0.1, xmax=0.18, color='black', linewidth=1.5)
        ax_zoom.text(0.14, 1.15, global_delay_star, ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')
    else:
        ax_zoom.text(0.14, 1.15, "ns", ha='center', va='bottom', fontsize=10, fontstyle='italic', color='black')

    ax_main.axvspan(zoom_window[0], zoom_window[1], color='gray', alpha=0.1)

    for axis in [ax_main, ax_zoom]:
        axis.set_facecolor("white")
        for spine in ['top', 'right']:
            axis.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            axis.spines[spine].set_color("black")
            axis.spines[spine].set_linewidth(1.0)
        axis.tick_params(color="black", labelcolor="black")

    plt.subplots_adjust(left=0.12, right=0.95, top=0.85, bottom=0.18)

    if savepath is not None:
        fig.savefig(savepath, dpi=300, facecolor="white", bbox_inches="tight")

    return fig, (ax_main, ax_zoom)

# ====================================================
# EXECUTAR O SCRIPT COM OS 3 GRUPOS
# ====================================================
DATASETS = [
    #'Deconvolved_SST-cells_WT_P15-P19_V1_contrast-1.0',
    #'Deconvolved_SST-cells_WT_P20-P23_V1_contrast-1.0',
    'Deconvolved_SST-cells_WT_P24-P27_V1_contrast-1.0',
    'Deconvolved_SST-cells_WT_Adult_V1_contrast-1.0',
    'Deconvolved_PV-cells_WT_Adult_V1_contrast-1.0',
    'Deconvolved_PYR-SynGCaMP_WT_Adult_V1_contrast-1.0'
]
LABELS = ['SST-WT24-27', 'SST-WT', 'PV-WT', 'PYR-WT']
COLORS = ['k', '#12522eff', '#b30a7bff', '#005d8aff']
summary_folder = os.path.expanduser("~/CURATED/Cibele/summary")

fig, axes = compute_multi_group_stats_and_plot(
    keys=DATASETS,
    path=summary_folder,
    colors=COLORS,
    group_labels=LABELS,
    zoom_window=(0, 0.3),
    late_window=(0.8, 2.0),
    norm="min-max",
    savepath=os.path.expanduser("~/Desktop/Final-Figures-PhD/Neuron_style_panel_D_3groups.svg")
)




