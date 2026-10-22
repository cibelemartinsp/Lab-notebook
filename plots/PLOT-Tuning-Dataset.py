# %%
import sys, os
sys.path += [os.path.join(os.path.expanduser('~'),\
                          'lab-notebook', 'cibele', 'physion', 'src')]

import physion.utils.plot_tools as pt
from physion.analysis.protocols.orientation_tuning\
        import plot_orientation_tuning_curve, plot_selectivity,\
                    plot_responsiveness, plot_selectivity_distrib

summary_folder =\
        os.path.expanduser('~/CURATED/Cibele/summary')
folders = [
    #"PV-cells_WT_Adult_V1", 
    #"PV-cells_WT_Young_V1",
    #"PV-cells_cond-GluN1-KO_Adult_V1", 
    #"PYR-PV-SynGCaMP_WT_Young_V1",
    "SST-cells_cond-GluN1-KO_Young_V1",
    #"SST-cells_WT_Adult_V1",
    #"SST-cells_WT_Young_V1",
    #"SST-cells_cond-GluN1-KO_Adult_V1",
    #"SST-cells_WT_Adult_V1_Taddy"
]
for i, folder in enumerate(folders):

    fig, ax = plot_orientation_tuning_curve(\
                            ['%s_contrast-1.0' % folder, 
                                '%s_contrast-0.5' % folder],
                             average_by='ROIs',
                            # average_by='subjects',
                            colors = [pt.tab10(i), 'lightgrey'],
                            path=summary_folder)
    fig, ax = plot_selectivity(\
                            ['%s_contrast-1.0' % folder, 
                                '%s_contrast-0.5' % folder],
                            # average_by='ROIs',
                            # average_by='subjects',
                            #  using='fit',
                            colors = [pt.tab10(i), 'lightgrey'],
                            path=summary_folder)
    print(folder, fig)







#%%
#########################################################
#### ORIENTATION TUNING CURVES WITH NUMBER ANIMALS ######
#########################################################

def count_animals_from_npy(summary_folder, dataset_name):
    import os
    import numpy as np

    prefix = f"Tunings_{dataset_name}"
    
    # find matching file
    files = [f for f in os.listdir(summary_folder)
             if f.startswith(prefix) and f.endswith(".npy")]
    
    if len(files) == 0:
        raise FileNotFoundError(f"No .npy files starting with {prefix}")
    
    file_path = os.path.join(summary_folder, files[0])

    # load array of dicts
    arr = np.load(file_path, allow_pickle=True)

    # extract subjects from each entry
    subjects = [entry["subject"] for entry in arr]

    # count unique animals
    unique_subjects = sorted(set(subjects))

    return len(unique_subjects), unique_subjects

summary_folder = "/Users/cibelemartinspinho/CURATED/Cibele/summary/"

datasets = [
    #'PYR-PV-SynGCaMP_WT_P15-P19_V1_contrast-0.5',
    #'PYR-PV-SynGCaMP_WT_P20-P23_V1_contrast-0.5',
    #'PYR-PV-SynGCaMP_WT_P24-P27_V1_contrast-0.5',
    'PYR-SynGCaMP_WT_Adult_V1_contrast-0.5',
    'PYR-SynGCaMP_WT_Adult_V1_contrast-1.0'
    #'SST-cells_cond-GluN1-KO_P15-P19_V1_contrast-1.0',
    #'SST-cells_cond-GluN1-KO_P20-P23_V1_contrast-1.0',
    #'SST-cells_cond-GluN1-KO_P24-P27_V1_contrast-1.0',
    #'SST-cells_cond-GluN1-KO_Adult_V1_contrast-1.0',
    #'SST-cells_WT_P15-P19_V1_contrast-0.5',
    #'SST-cells_WT_P20-P23_V1_contrast-0.5',
    #'SST-cells_WT_P24-P27_V1_contrast-0.5',
    #'PV-cells_cond-GluN1-KO_Adult_V1_contrast-0.5',
    #'PV-cells_cond-GluN1-KO_Adult_V1_contrast-0.5'
]

fig, ax = plot_orientation_tuning_curve(
    datasets,
    average_by='sessions', #default is 'subjects'
    #colors=['#c6e7bbff', '#99b993ff','#428e58ff', '#12522eff'],
    colors=['darkgray', '#005d8aff' ],
    path=summary_folder
)

#Count animals and print
print("\nAnimal counts:")
for d in datasets:
    n, animals = count_animals_from_npy(summary_folder, d)
    print(f"{d}: {n} animals → {animals}")


# --- REFINED AESTHETIC ADJUSTMENTS (from clean publication style) ---

# 1. Background and Figure Size
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# 2. Remove the unwanted text/legend block on the right
for txt in list(ax.texts):
    txt.remove()

# 3. Minimalist L-Shape Spines (Keep Left only, hide Top, Right, Bottom)
spine_thickness = 0.5 

ax.spines['left'].set_color('black')
ax.spines['left'].set_linewidth(spine_thickness)

for side in ['top', 'right', 'bottom']:
    ax.spines[side].set_visible(False)

# 4. Axis Label Styling
label_font = {
    'family': 'Arial',
    'color':  'black',
    'weight': 'normal',
    'size': 7,
}
ax.set_xlabel(ax.get_xlabel(), fontdict=label_font)
ax.set_ylabel(ax.get_ylabel(), fontdict=label_font)

# 5. Tick Parameters
ax.tick_params(
    axis='both', 
    colors='black', 
    labelsize=6,        
    width=spine_thickness, 
    length=3
)

#---SAVE FIGURE
fig.savefig(os.path.expanduser('~/Desktop/Final-Figures-PhD/Orientation_Tuning_Curves_SESSION.svg'))










# %%
#########################################################
########## SELECTIVITY INDEX - WITH STATISTICS ##########
#########################################################

import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu
import itertools
import numpy as np

# --- Data loading function for session-level statistics ---
def load_session_SI(summary_folder, dataset):
    """Loads selectivity index and returns ONE mean SI value per session."""
    fpath = os.path.join(summary_folder, f"Tunings_{dataset}.npy")
    data = np.load(fpath, allow_pickle=True)
    session_si = []
    for entry in data:
        si = np.array(entry["selectivities"])
        session_si.append(si.mean())
    return np.array(session_si)

##SELECTIVITY INDEX
# Define datasets for the plot (active, non-commented entries)
datasets = [
    'SST-cells_cond-GluN1-KO_P15-P19_V1_contrast-1.0',
    'SST-cells_cond-GluN1-KO_P20-P23_V1_contrast-1.0',
    'SST-cells_cond-GluN1-KO_P24-P27_V1_contrast-1.0',
    'SST-cells_cond-GluN1-KO_Adult_V1_contrast-1.0',
]

fig, ax = plot_selectivity(
    datasets,
    #average_by='ROIs',
    #average_by='subjects',
    colors=['#dbc0f2', '#b38fd6', '#9740d2', '#5d1490'],  # Solid colors, no transparency
    path=summary_folder,
    fig_args=dict(ax_scale=(0.8,1))
)


# ====================================================
# SESSION-LEVEL STATISTICS
# ====================================================

# Load session-level data
session_data = [load_session_SI(summary_folder, d) for d in datasets]

# Get short dataset names for display
dataset_names = [d.split('_contrast')[0] for d in datasets]

print("="*70)
print("SELECTIVITY INDEX - SESSION-LEVEL STATISTICS")
print("="*70)

# Print N values
print("\nDataset info (sessions):")
for d, data in zip(dataset_names, session_data):
    print(f"  {d}: N = {len(data)} sessions")

# Run Kruskal-Wallis H-test
H_sess, p_sess = kruskal(*session_data)
print(f"\nKruskal-Wallis H-test (Global): H = {H_sess:.4f}, p = {p_sess:.4g}")

# Post-hoc pairwise Mann-Whitney U tests with Bonferroni
sig_pairs = []
if p_sess < 0.05:
    print("\nPost-hoc Pairwise Comparisons (Bonferroni Corrected):")
    pairs = list(itertools.combinations(range(len(datasets)), 2))
    num_comparisons = len(pairs)
    
    for i, j in pairs:
        _, p_raw = mannwhitneyu(session_data[i], session_data[j], alternative="two-sided")
        p_adjusted = min(p_raw * num_comparisons, 1.0)
        
        name1 = dataset_names[i]
        name2 = dataset_names[j]
        sig_marker = "*" if p_adjusted < 0.05 else "NS"
        
        print(f"  {name1} vs {name2}: p = {p_adjusted:.4g} [{sig_marker}]")
        
        if p_adjusted < 0.05:
            sig_pairs.append((i, j))
else:
    print("\nResult is not significant. No post-hoc tests performed.")

print("="*70)


# --- ADD SIGNIFICANCE MARKERS TO PLOT ---
if sig_pairs:
    fig.canvas.draw()  # Ensure figure is rendered
    ymax = ax.get_ylim()[1]
    
    # Track y-positions per x-position to stack stars vertically
    y_positions = {}
    base_y = ymax * 1.05  # Start 5% above top of violins
    y_increment = ymax * 0.05  # 5% spacing between stars
    
    for i, j in sig_pairs:
        x_pos = (i + j) / 2.0
        
        # Get current y-position for this x-position
        if x_pos not in y_positions:
            y_positions[x_pos] = base_y
        else:
            y_positions[x_pos] += y_increment
        
        # Add star symbol
        ax.text(x_pos, y_positions[x_pos], '*', ha='center', va='bottom', 
                fontsize=12, color='black')


# --- REFINED AESTHETIC ADJUSTMENTS ---

# 1. Background and Spine (Axis Line) Thinning
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

spine_thickness = 0.5 

for side in ['top', 'bottom', 'left', 'right']:
    ax.spines[side].set_color('black')
    ax.spines[side].set_linewidth(spine_thickness)

# 2. Fix violin plots: remove gray outline, use solid colors, black mean/error bars
for child in ax.get_children():
    # Violin body collections - remove outline and make solid
    if isinstance(child, plt.matplotlib.collections.PolyCollection):
        child.set_edgecolor('none')
        child.set_alpha(1.0)
    # Mean points and error bars - make black
    elif isinstance(child, plt.Line2D):
        child.set_color('black')
        child.set_markeredgecolor('black')
        child.set_alpha(1.0)
    # Scatter points/error bars - make black
    elif isinstance(child, plt.matplotlib.collections.PathCollection):
        child.set_edgecolor('black')
        child.set_alpha(1.0)

# 3. Legend & Label Customization
# We target ax.texts directly because ax.get_legend() often returns None in these plots
for txt in ax.texts:
    txt.set_fontsize(6)          # Small font for the legend text
    txt.set_fontname('Arial')

# 4. Axis Label Styling (The "Selectivity Index" and "Count" titles)
label_font = {
    'family': 'Arial',
    'color':  'black',
    'weight': 'normal',
    'size': 7,
}
ax.set_xlabel(ax.get_xlabel(), fontdict=label_font)
ax.set_ylabel(ax.get_ylabel(), fontdict=label_font)

# 5. Tick Parameters (The numbers on the X and Y axes)
ax.tick_params(
    axis='both', 
    colors='black', 
    labelsize=6,        # Smaller numbers on axes
    width=spine_thickness, 
    length=3
)

#---SAVE FIGURE
fig.savefig(os.path.expanduser('~/Desktop/Final-Figures-PhD/Select.Index.svg'))




#%%
###################################################################
## SELECTIVITY INDEX WITH ACCUMULATIVE DISTRIBUTION - For ROIs ####
###################################################################

import numpy as np
import os

# --- Style parameters ---
spine_thickness = 0.5

label_font = {
    'family': 'Arial',   # or 'DejaVu Sans' for portability
    'color': 'black',
    'weight': 'normal',
    'size': 9,
}

for contrast in [0.5, 1.0]:

    fig, ax = plot_selectivity_distrib(
        [
            "SST-cells_cond-GluN1-KO_P15-P19_V1_contrast-%.1f" % contrast,
            "SST-cells_cond-GluN1-KO_P20-P23_V1_contrast-%.1f" % contrast,
            "SST-cells_cond-GluN1-KO_P24-P27_V1_contrast-%.1f" % contrast,
            "SST-cells_cond-GluN1-KO_Adult_V1_contrast-%.1f" % contrast,
            #"SST-cells_WT_P15-P19_V1_contrast-%.1f" % contrast,
            #"SST-cells_WT_P20-P23_V1_contrast-%.1f" % contrast,
            #"SST-cells_WT_P24-P27_V1_contrast-%.1f" % contrast,
            #"PYR-PV-SynGCaMP_WT_P20-P23_V1_contrast-%.1f" % contrast,
            #"PYR-PV-SynGCaMP_WT_P24-P27_V1_contrast-%.1f" % contrast,
            #"PYR-SynGCaMP_WT_Adult_V1_contrast-%.1f" % contrast,
        ],
        plot='cum. frac.',
        bins=np.linspace(0, .99, 40),
        colors=['#dbc0f2ff', '#b38fd6ff', '#9740d2ff', '#5d1490ff', '#c6e7bbff', '#99b993ff', '#428e58ff'],
        #colors=['#b38fd6ff', '#9740d2ff','#5d1490ff'],
        path=summary_folder
    )

    # ---------------------------------------------------
    # REFINED AESTHETIC ADJUSTMENTS
    # ---------------------------------------------------

    # Background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Spines
    for side in ['top', 'bottom', 'left', 'right']:
        ax.spines[side].set_color('black')
        ax.spines[side].set_linewidth(spine_thickness)

    # Title
    ax.set_title(
        f'contrast = {contrast:.1f}',
        fontsize=9,
        fontname='Arial',
        color='black'
    )

    # Axis labels (ONLY if your plotting function already sets labels)
    ax.set_xlabel(ax.get_xlabel(), fontdict=label_font)
    ax.set_ylabel(ax.get_ylabel(), fontdict=label_font)

    # Tick parameters
    ax.tick_params(
        axis='both',
        colors='black',
        labelsize=7,
        width=spine_thickness
    )

    # Legend formatting
    leg = ax.get_legend()
    if leg:
        for text in leg.get_texts():
            text.set_fontsize(5)
            text.set_fontname('Arial')

        # Optional legend frame cleanup
        leg.get_frame().set_linewidth(0.5)
        leg.get_frame().set_edgecolor('black')

    # Optional: remove top/right ticks
    ax.tick_params(top=False, right=False)

    # Better spacing
    fig.tight_layout()

    # Save
    fig.savefig(
        os.path.expanduser(
            f'~/Desktop/Final-Figures-PhD/accumulative-distribution_{contrast}.svg'
        ),
        dpi=300,
        bbox_inches='tight'
    )












# %%
#########################################################
########## RESPONSIVENESS - PIE CHART ###################
#########################################################

import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu
import numpy as np
import os
import itertools
from collections import defaultdict

plt.close('all')  # <-- IMPORTANT

# ------------------------------------------------
# Load responsiveness data per SESSION
# ------------------------------------------------
def load_responsiveness_data(summary_folder, dataset):
    """
    Loads responsiveness data and returns counts for:
    - non_responsive
    - positive  
    - negative
    
    Looks for fields: nROIs_positive, nROIs_negative, nROIs_responsive, nROIs_final
    or arrays: positive, negative, responsive
    """
    fname = f"Tunings_{dataset}.npy"
    data = np.load(os.path.join(summary_folder, fname), allow_pickle=True)

    # Initialize lists
    non_responsive_counts = []
    positive_counts = []
    negative_counts = []
    total_counts = []
    
    for entry in data:
        n_total = entry.get("nROIs_final", entry.get("nROIs", 0))
        n_resp = entry.get("nROIs_responsive", 0)
        
        # Try to get positive and negative counts from various possible fields
        n_pos = 0
        n_neg = 0
        
        # Method 1: Direct counts
        if "nROIs_positive" in entry:
            n_pos = entry["nROIs_positive"]
        elif "nROIs_pos_response" in entry:
            n_pos = entry["nROIs_pos_response"]
        
        if "nROIs_negative" in entry:
            n_neg = entry["nROIs_negative"]
        elif "nROIs_neg_response" in entry:
            n_neg = entry["nROIs_neg_response"]
        
        # Method 2: Check if there are arrays
        if n_pos == 0 and n_neg == 0:
            if "positive" in entry:
                pos_arr = entry["positive"]
                if isinstance(pos_arr, (list, np.ndarray)):
                    n_pos = np.sum(pos_arr) if len(pos_arr) > 0 else 0
            if "negative" in entry:
                neg_arr = entry["negative"]
                if isinstance(neg_arr, (list, np.ndarray)):
                    n_neg = np.sum(neg_arr) if len(neg_arr) > 0 else 0
            
            # Method 3: If we have responsive array but no pos/neg, 
            # check if there's a sign field or calculate from tuning curves
            if n_pos == 0 and n_neg == 0 and "responsive" in entry:
                responsive = entry.get("responsive", [])
                if isinstance(responsive, (list, np.ndarray)) and len(responsive) > 0:
                    # If responsive is a list of indices or booleans
                    if len(responsive) == n_resp:
                        # Assume all are positive if we can't determine
                        n_pos = n_resp
                        n_neg = 0
        
        # Fallback: if still no pos/neg, use responsive as positive
        if n_pos == 0 and n_neg == 0 and n_resp > 0:
            n_pos = n_resp
            n_neg = 0
        
        # Ensure counts don't exceed responsive
        if n_pos + n_neg > n_resp:
            n_pos = n_resp
            n_neg = 0
        
        non_responsive = n_total - n_resp
        
        non_responsive_counts.append(non_responsive)
        positive_counts.append(n_pos)
        negative_counts.append(n_neg)
        total_counts.append(n_total)
    
    return {
        'non_responsive': np.array(non_responsive_counts),
        'positive': np.array(positive_counts),
        'negative': np.array(negative_counts),
        'total': np.array(total_counts)
    }

# ------------------------------------------------
# DATASETS
# ------------------------------------------------
datasets = [
    'SST-cells_WT_Adult_V1_contrast-1.0',
    'SST-cells_WT_Adult_Ventricule_V1_contrast-1.0',
    'SST-cells_cond-GluN1-KO_Adult_V1_contrast-1.0',
]

# ------------------------------------------------
# Load data
# ------------------------------------------------
all_data = {d: load_responsiveness_data(summary_folder, d) for d in datasets}

# ====================================================
# SESSION-LEVEL STATISTICS
# ====================================================

print("="*70)
print("RESPONSIVENESS - SESSION-LEVEL STATISTICS")
print("="*70)

# Calculate session-level responsiveness (for statistics)
session_data = []
for d in datasets:
    data = all_data[d]
    resp_per_session = data['positive'] + data['negative']
    session_resp = resp_per_session / data['total']
    session_data.append(session_resp)

dataset_names = [d.split('_contrast')[0] for d in datasets]

# Print N values and data info
print("\nDataset info (sessions):")
for d, data_dict, name in zip(datasets, all_data.values(), dataset_names):
    n_sessions = len(data_dict['total'])
    total_rois = data_dict['total'].sum()
    responsive_rois = (data_dict['positive'] + data_dict['negative']).sum()
    positive_rois = data_dict['positive'].sum()
    negative_rois = data_dict['negative'].sum()
    print(f"  {name}:")
    print(f"    N = {n_sessions} sessions, {total_rois} total ROIs")
    print(f"    Responsive: {responsive_rois} ({100*responsive_rois/total_rois:.1f}%)")
    print(f"      - Positive: {positive_rois} ({100*positive_rois/responsive_rois:.1f}% of responsive)")
    print(f"      - Negative: {negative_rois} ({100*negative_rois/responsive_rois:.1f}% of responsive)")
    print(f"    Non-responsive: {total_rois - responsive_rois} ({100*(total_rois - responsive_rois)/total_rois:.1f}%)")

# Run Kruskal-Wallis H-test for multi-group comparison
H_sess, p_sess = kruskal(*session_data)
print(f"\nKruskal-Wallis H-test (Global): H = {H_sess:.4f}, p = {p_sess:.4g}")

# Post-hoc pairwise Mann-Whitney U tests with Bonferroni
sig_pairs = []
if p_sess < 0.05:
    print("\nPost-hoc Pairwise Comparisons (Bonferroni Corrected):")
    pairs = list(itertools.combinations(range(len(datasets)), 2))
    num_comparisons = len(pairs)
    
    for i, j in pairs:
        _, p_raw = mannwhitneyu(session_data[i], session_data[j], alternative="two-sided")
        p_adjusted = min(p_raw * num_comparisons, 1.0)
        
        name1 = dataset_names[i]
        name2 = dataset_names[j]
        sig_marker = "*" if p_adjusted < 0.05 else "NS"
        
        print(f"  {name1} vs {name2}: p = {p_adjusted:.4g} [{sig_marker}]")
        
        if p_adjusted < 0.05:
            sig_pairs.append((i, j))
else:
    print("\nResult is not significant. No post-hoc tests performed.")

print("="*70)

# ====================================================
# PIE CHART - Responsiveness Breakdown
# ====================================================

# Define colors: light gray for non-responsive, group-specific colors for positive/negative
# Use the SAME colors from your original bar chart for consistency
non_responsive_color = '#d3d3d3'  # Light gray for non-responsive (consistent across all groups)

# Dataset-specific colors from your original responsiveness plot
# SST-cells_WT_Adult_V1: #12522eff (blue)
# SST-cells_WT_Adult_Ventricule_V1: tab:grey (grey) 
# SST-cells_cond-GluN1-KO_Adult_V1: #5d1490ff (purple)
# For pie chart: use dataset color for positive, darker shade for negative
dataset_colors = {
    'SST-cells_WT_Adult_V1': {'positive': '#12522eff', 'negative': '#0a381e'},
    'SST-cells_WT_Adult_Ventricule_V1': {'positive': '#666666', 'negative': '#404040'},
    'SST-cells_cond-GluN1-KO_Adult_V1': {'positive': '#5d1490ff', 'negative': '#3a0b5d'}
}

labels_pie = ['Non-responsive', 'Positive Response', 'Negative Response']

# Create a figure with subplots for each dataset
fig, axes = plt.subplots(1, len(datasets), figsize=(15, 5))
if len(datasets) == 1:
    axes = [axes]

for idx, (d, data_dict, ax, name) in enumerate(zip(datasets, all_data.values(), axes, dataset_names)):
    # Aggregate across all sessions for this dataset
    total_rois = data_dict['total'].sum()
    positive_rois = data_dict['positive'].sum()
    negative_rois = data_dict['negative'].sum()
    non_responsive_rois = total_rois - (positive_rois + negative_rois)
    
    # Calculate percentages
    sizes = [non_responsive_rois, positive_rois, negative_rois]
    total = sum(sizes)
    
    # Get dataset-specific colors
    # Match the dataset name to the color scheme
    color_key = None
    for key in dataset_colors:
        if key in d:
            color_key = key
            break
    
    if color_key:
        pos_color = dataset_colors[color_key]['positive']
        neg_color = dataset_colors[color_key]['negative']
    else:
        # Default colors if no match - use first dataset's colors
        pos_color = '#12522eff'
        neg_color = '#0a381e'
    
    # Color scheme: light gray for non-responsive, group color for positive/negative
    colors_for_this_pie = [non_responsive_color, pos_color, neg_color]
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels_pie,
        colors=colors_for_this_pie,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 12, 'color': 'black'}
    )
    
    # Add title
    ax.set_title(name, fontsize=12, fontname='Arial', color='black', pad=10)
    
    # Equal aspect ratio ensures pie is drawn as a circle
    ax.axis('equal')
    
    # Adjust label distances
    for text in texts:
        text.set_fontsize(7)
        text.set_fontname('Arial')
    for autotext in autotexts:
        autotext.set_fontsize(12)
        autotext.set_fontname('Arial')

# Add overall figure title
fig.suptitle('Responsiveness Distribution: Non-responsive vs Positive vs Negative', 
             fontsize=10, fontname='Arial', color='black', y=1.05)

# Adjust layout
plt.tight_layout()

# ====================================================
# AESTHETIC ADJUSTMENTS
# ====================================================

# Background
fig.patch.set_facecolor('white')

# Adjust spacing
plt.subplots_adjust(top=0.85, wspace=0.3)

# Show/save figure
# fig.savefig(os.path.expanduser('~/Desktop/Final-Figures-PhD/Responsiveness_Pie_Chart.svg'), 
#             dpi=300, bbox_inches='tight')

# %%

