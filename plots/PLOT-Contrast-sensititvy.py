# %%
import sys, os, tempfile
import numpy as np
from scipy import stats

sys.path += [os.path.join(os.path.expanduser('~'),\
                          'lab-notebook', 'cibele', 'physion', 'src')]

import physion.utils.plot_tools as pt
pt.set_style('dark')

from physion.analysis.protocols.contrast_sensitivity\
        import plot_contrast_sensitivity, plot_contrast_responsiveness

summary_folder = \
    os.path.expanduser('~/CURATED/Cibele/summary')






#%%
#######################################################################################
### GAIN OF CONTRAST SENSITITIVITY WITH STATISTICS ###
#######################################################################################

import os
import numpy as np
from scipy.stats import mannwhitneyu
import itertools

def count_animals_from_npy(folder, dataset_name):
    # Find file matching dataset name
    matches = [f for f in os.listdir(folder)
               if dataset_name in f and f.endswith(".npy")]
    
    if len(matches) == 0:
        raise FileNotFoundError(f"No file found for dataset {dataset_name}")
    if len(matches) > 1:
        raise ValueError(f"Multiple matches for {dataset_name}: {matches}")
    
    fpath = os.path.join(folder, matches[0])
    
    # Load the object array
    arr = np.load(fpath, allow_pickle=True)

    # ---------- ADDED: ROI counts ----------
    n_rois_original = arr[0]["nROIs_original"]   # ORIGINAL ROI count
    n_rois_final = arr[0]["nROIs_final"]         # FINAL ROI count
    # ---------------------------------------

    # Extract subjects from each entry (one per session)
    subjects = [entry['subject'] for entry in arr]
    
    # Unique animal IDs
    unique_subjects = sorted(set(subjects))
    
    return len(unique_subjects), unique_subjects, n_rois_original, n_rois_final


def compute_session_gain(entry):
    """
    Gain = slope between lowest and highest contrast:
    (mean response at max contrast - mean response at min contrast) / Δcontrast
    Assumes contrasts normalized 0→1 (Δcontrast = 1).
    """
    R = entry["Responses"]  # shape (n_cells, n_contrasts)
    mean_low = R[:, 0].mean()
    mean_high = R[:, -1].mean()
    return mean_high - mean_low


def extract_gains_from_dataset(folder, dataset_name):
    matches = [f for f in os.listdir(folder)
               if dataset_name in f and f.endswith(".npy")]
    if len(matches) == 0:
        raise FileNotFoundError(dataset_name)
    if len(matches) > 1:
        raise ValueError(matches)

    arr = np.load(os.path.join(folder, matches[0]), allow_pickle=True)
    
    gains = [compute_session_gain(entry) for entry in arr]
    return np.array(gains)


pt.set_style()

DATASETS = [#'Deconvolved_PYR-PV-SynGCaMP_WT_P15-P19_V1_angle-0.0',
            'Deconvolved_SST-cells_cond-GluN1-KO_P15-P19_V1_angle-0.0',
            'Deconvolved_SST-cells_cond-GluN1-KO_P20-P23_V1_angle-0.0',
            'Deconvolved_SST-cells_cond-GluN1-KO_P24-P27_V1_angle-0.0',
            'Deconvolved_SST-cells_cond-GluN1-KO_Adult_V1_angle-0.0',
            #'Deconvolved_SST-cells_WT_P15-P19_V1_angle-0.0',
            #'Deconvolved_SST-cells_WT_P20-P23_V1_angle-0.0',
            #'Deconvolved_SST-cells_WT_P24-P27_V1_angle-0.0',   
            #'Deconvolved_PV-cells_WT_Adult_V1_angle-0.0',
            #'Deconvolved_PYR-SynGCaMP_WT_Adult_V1_angle-0.0',
            #'Deconvolved_SST-cells_cond-GluN1-KO_Adult_V1_angle-0.0', 
            #'PYR-PV-SynGCaMP_WT_P24-P27_V1_angle-0.0',
            ]

# ====================================================
# PLOT GAIN OF CONTRAST SENSITIVITY
# ====================================================

fig, ax = plot_contrast_sensitivity(DATASETS,\
                        average_by='sessions',
                        #average_by='ROIs',
                        #average_by='subjects',
                        colors = ['#dbc0f2ff', '#b38fd6ff', '#9740d2ff', '#5d1490ff', '#c6e7bbff', '#99b993ff', '#428e58ff', ],
                        path=summary_folder,
                        #gain_plot='bar'
                        )


animal_counts = {}
for d in DATASETS:
    n_animals, subjects, n_rois_orig, n_rois_final = count_animals_from_npy(summary_folder, d)
    animal_counts[d] = (n_animals, subjects)

    # ---------- ADDED: print original + final ROIs ----------
    print(f"{d}: {n_animals} animals → {subjects} | "
          f"nROIs_original = {n_rois_orig}, nROIs_final = {n_rois_final}")
    # --------------------------------------------------------

# ====================================================
# STATISTICS - Gain Comparison
# ====================================================

# Extract gains from datasets
gain_dict = {}

print("\n=== Mean gain per dataset ===")
for d in DATASETS:
    gains = extract_gains_from_dataset(summary_folder, d)
    gain_dict[d] = gains

    mean_gain = gains.mean()
    sem_gain = gains.std() / np.sqrt(len(gains))

    print(f"{d}:  mean gain = {mean_gain:.3f}  ± {sem_gain:.3f}  (n={len(gains)})")

print("\n=== Mann–Whitney U tests (gain) ===")
names = list(gain_dict.keys())
sig_pairs = []

for i in range(len(names)):
    for j in range(i+1, len(names)):
        d1, d2 = names[i], names[j]
        U, p = mannwhitneyu(gain_dict[d1], gain_dict[d2], alternative='two-sided')
        sig_marker = "*" if p < 0.05 else "NS"
        print(f"{d1}  vs  {d2}:  U={U:.1f},  p={p:.5f} [{sig_marker}]")
        
        if p < 0.05:
            sig_pairs.append((i, j))

# ====================================================
# ADD SIGNIFICANCE BARS TO GAIN PLOT
# ====================================================
from matplotlib.collections import PolyCollection

# 1) Find the gain axes (the one titled 'gain')
ax_gain = None
for a in fig.axes:
    if a.get_title() == 'gain':
        ax_gain = a
        break
if ax_gain is None:
    ax_gain = fig.axes[1]   # fallback: second panel

if sig_pairs:
    fig.canvas.draw()

    # 2) x position of each violin
    x_positions = []
    for child in ax_gain.get_children():
        if isinstance(child, PolyCollection):
            for path in child.get_paths():
                x_positions.append(path.vertices[:, 0].mean())
    x_positions = sorted(x_positions)
    if len(x_positions) != len(DATASETS):
        x_positions = list(range(len(DATASETS)))   # fallback

    # 3) Starting height: above the highest data point
    ymin, ymax = ax_gain.get_ylim()
    yrange = ymax - ymin
    y0 = ymax + 0.05 * yrange      # height of the first bar
    step = 0.15 * yrange           # spacing between stacked bars
    tick = 0.03 * yrange           # length of the bar's end ticks

    # 4) Draw thin bar + asterisk on top
    for level, (i, j) in enumerate(sig_pairs):
        x1, x2 = x_positions[i], x_positions[j]
        y = y0 + level * step

        ax_gain.plot([x1, x1, x2, x2],
                     [y - tick, y, y, y - tick],
                     lw=0.6, color='k', clip_on=False, solid_capstyle='butt')

        ax_gain.text((x1 + x2) / 2, y, '*',
                     ha='center', va='bottom', fontsize=9,
                     color='k', clip_on=False)

# Move the 'gain' title up so it clears the significance bars
n_levels = len(sig_pairs)
title_pad = 6 + 12 * n_levels   # points; grows with the number of stacked bars
ax_gain.set_title('gain', pad=title_pad)


# ====================================================
# SAVE (bbox_inches='tight' avoids clipping the bars)
# ====================================================
fig.savefig(os.path.expanduser('~/Desktop/Final-Figures-PhD/Contrast_Gain.svg'),
            bbox_inches='tight')







# %%
################################################################
####### Contrast Responsiveness (Positive and Negative) ########
#################################################################

fig, ax = plot_contrast_responsiveness(DATASETS,\
                         sign='negative',
                         nROIs='final', # "original" or "final", before/after dFoF criterion
                        colors = ['#c6e7bbff', '#99b993ff', '#428e58ff', '#12522eff'],
                        path=summary_folder,
                        fig_args=dict(ax_scale=(1.5,2)))
ax.set_ylim([0,100]) # --> To have the same eixo 
fig.savefig(os.path.expanduser('~/Desktop/Final-Figures-PhD/Contrast_Responsiveness_Negative.svg'))          

fig, ax = plot_contrast_responsiveness(DATASETS,\
                         sign='positive',
                         nROIs='final', # "original" or "final", before/after dFoF criterion
                        colors = ['#c6e7bbff', '#99b993ff', '#428e58ff', '#12522eff'],
                        path=summary_folder,
                        fig_args=dict(ax_scale=(1.5,2)))
ax.set_ylim([0,100]) # --> To have the same eixo 
fig.savefig(os.path.expanduser('~/Desktop/Final-Figures-PhD/Contrast_Responsiveness_Positive.svg'))    







#%%
####### Responsivenes together #####
import os
import numpy as np
import matplotlib.pyplot as plt
import physion.utils.plot_tools as pt

# ====================================================
# SEU SCRIPT ORIGINAL (COM AS SUAS ALTERAÇÕES)
# ====================================================
def count_animals_from_npy(folder, dataset_name):
    matches = [f for f in os.listdir(folder) if dataset_name in f and f.endswith(".npy")]
    if len(matches) == 0:
        raise FileNotFoundError(f"No file found for dataset {dataset_name}")
    if len(matches) > 1:
        raise ValueError(f"Multiple matches for {dataset_name}: {matches}")
    
    fpath = os.path.join(folder, matches[0])
    arr = np.load(fpath, allow_pickle=True)

    n_rois_original = arr[0]["nROIs_original"]
    n_rois_final = arr[0]["nROIs_final"]

    subjects = [entry['subject'] for entry in arr]
    unique_subjects = sorted(set(subjects))
    
    return len(unique_subjects), unique_subjects, n_rois_original, n_rois_final

pt.set_style()

DATASETS = [
    'Deconvolved_SST-cells_WT_P15-P19_V1_angle-0.0',
    'Deconvolved_SST-cells_cond-GluN1-KO_P15-P19_V1_angle-0.0',
    'Deconvolved_SST-cells_WT_P20-P23_V1_angle-0.0',
    'Deconvolved_SST-cells_cond-GluN1-KO_P20-P23_V1_angle-0.0',
    'Deconvolved_SST-cells_WT_P24-P27_V1_angle-0.0',
    'Deconvolved_SST-cells_cond-GluN1-KO_P24-P27_V1_angle-0.0',
    'Deconvolved_SST-cells_WT_Adult_V1_angle-0.0',
    'Deconvolved_SST-cells_cond-GluN1-KO_Adult_V1_angle-0.0',
    
   
]
summary_folder = os.path.expanduser("~/CURATED/Cibele/summary")
COLORS = ['#c6e7bbff', '#dbc0f2ff', '#99b993ff', '#b38fd6ff', '#428e58ff', '#9740d2ff', '#12522eff', '#5d1490ff' ]

# Contagem de animais
for d in DATASETS:
    n_animals, subjects, n_rois_orig, n_rois_final = count_animals_from_npy(summary_folder, d)
    print(f"{d}: {n_animals} animals | nROIs_orig = {n_rois_orig}, nROIs_final = {n_rois_final}")

# 1. Gráfico de Gain normal
fig_gain, ax_gain = plot_contrast_sensitivity(
    DATASETS, average_by='sessions', colors=COLORS, path=summary_folder
)
os.makedirs(os.path.expanduser('~/Desktop/Final-Figures-PhD'), exist_ok=True)
fig_gain.savefig(os.path.expanduser('~/Desktop/Final-Figures-PhD/Contrast_Gain.svg'), bbox_inches='tight')

# 2. Gerar os gráficos separados de Positivo e Negativo (sem mexer no physion)
# Como você não pode mudar o physion, geramos os dois arquivos separadamente 
# e você pode juntá-los perfeitamente no Illustrator/Inkscape (o que costuma 
# dar mais controle estético para publicação no PhD).

fig_pos, ax_pos = plot_contrast_responsiveness(
    DATASETS,
    sign='positive',
    nROIs='final',
    colors=COLORS,
    path=summary_folder,
    fig_args=dict(ax_scale=(1.5, 2))
)
ax_pos.set_ylim([0, 80])
ax_pos.set_ylabel("% pos. resp.")
fig_pos.savefig(os.path.expanduser('~/Desktop/Final-Figures-PhD/Contrast_Responsiveness_Positive.svg'), bbox_inches='tight')

fig_neg, ax_neg = plot_contrast_responsiveness(
    DATASETS,
    sign='negative',
    nROIs='final',
    colors=COLORS,
    path=summary_folder,
    fig_args=dict(ax_scale=(1.5, 2))
)
# Se quiser simular o espelho visualmente abrindo separados, você ajusta o ylim:
ax_neg.set_ylim([80, 0]) 
ax_neg.set_ylabel("% neg. resp.")
fig_neg.savefig(os.path.expanduser('~/Desktop/Final-Figures-PhD/Contrast_Responsiveness_Negative.svg'), bbox_inches='tight')

print("\nGráficos separados salvos com sucesso no Desktop!")









#%%




#%%

#########################################################################################
##### PIE CHARTS FOR CONTRAST 1.0 RESPONSIVENESS (STATISTICAL SIGNIFICANCE FILTER) ######
################################# From the non deconvolved data #########################

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Publication quality styling configuration (Clean SVG lines)
plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none'})

def plot_contrast_1_hatched_pie_charts(folder, dataset_name, base_color):
    """
    Calculates cell responsiveness at Contrast 1.0 using statistical significance arrays.
    Returns the figure object so you can handle saving/showing yourself.
    """
    # Enforce matching strictly for files containing 'Sensitivities_', your dataset name, and 'angle-0.0'
    matches = [
        f for f in os.listdir(folder) 
        if 'Sensitivities_' in f and dataset_name in f and 'angle-0.0' in f and f.endswith(".npy")
    ]
    if len(matches) == 0:
        raise FileNotFoundError(f"No file found for dataset '{dataset_name}' containing 'Sensitivities_' and 'angle-0.0'")
        
    fpath = os.path.join(folder, matches[0])
    arr = np.load(fpath, allow_pickle=True)
    
    total_pos_cells = 0
    total_neg_cells = 0
    total_non_cells = 0
    
    # Accumulate cell counts based on real statistical significance arrays
    for entry in arr:
        # Get significance booleans at the last contrast column (-1 index -> Contrast 1.0)
        is_pos_significant = entry["significant_pos"][:, -1]
        is_neg_significant = entry["significant_neg"][:, -1]
        
        for pos_sig, neg_sig in zip(is_pos_significant, is_neg_significant):
            if pos_sig:         # If positive response is statistically significant
                total_pos_cells += 1
            elif neg_sig:       # If negative response is statistically significant
                total_neg_cells += 1
            else:               # If no significant response passes the test, it's Non-Responsive
                total_non_cells += 1
                
    grand_total = total_pos_cells + total_neg_cells + total_non_cells
    if grand_total == 0:
        raise ValueError(f"No cells found in dataset {dataset_name}")
        
    slices = [total_pos_cells, total_neg_cells, total_non_cells]
    categories = ['Positive', 'Negative', 'Non-Responsive']
    
    active_slices = []
    active_labels = []
    hatch_list = []
    face_colors = []
    
    for count, cat in zip(slices, categories):
        if count > 0:
            pct = (count / grand_total) * 100
            active_slices.append(count)
            active_labels.append(f"{cat}\n{count} cells\n({pct:.1f}%)")
            
            if cat == 'Positive':
                face_colors.append(base_color)
                hatch_list.append('')
            elif cat == 'Negative':
                rgb = mcolors.to_rgb(base_color)
                light_tint = [(c * 0.2) + 0.8 for c in rgb] 
                face_colors.append(light_tint)
                hatch_list.append('////') # Stripe pattern
            else:
                face_colors.append('#f0f0f0ff')
                hatch_list.append('')

    # Create the figure natively
    fig, ax = plt.subplots(figsize=(5, 5))

    wedges, texts = ax.pie(
        active_slices, 
        labels=active_labels, 
        colors=face_colors,
        startangle=90,
        textprops=dict(fontweight='medium', fontsize=9.5),
        wedgeprops=dict(edgecolor='w', linewidth=1.5, antialiased=True)
    )
    
    # Direct object-level styling instead of global rcParams configuration (fixes black artifact)
    for wedge, pattern in zip(wedges, hatch_list):
        if pattern:
            wedge.set_hatch(pattern)
            wedge.set_edgecolor(base_color) 
            wedge.set_linewidth(1.5)

    clean_name = dataset_name.replace('Sensitivities_', '').replace('_angle-0.0', '')
    ax.set_title(f"Cell Responsiveness at Contrast 1.0\n{clean_name}\n(Total Population n = {grand_total} ROIs)", 
                 fontsize=11, fontweight='bold', pad=20)
    
    return fig


# ==============================================================================
# BATCH EXECUTION: PROCESS AND SAVE ALL GROUPS SIMULTANEOUSLY
# ==============================================================================

# Define your exact 5 groups, file targets, and corresponding color specs
datasets_to_process = [
    # 1. PV (Pink/Spring)
   # {"name": "PV-cells_WT_Adult_V1", "color": "#b30a7bff"},
   # {"name": "PV-cells_WT_P24-P27", "color": "#e31593ff"},
   # {"name": "PV-cells_WT_P20-P23", "color": "#ff9ab5ff"},
    
    # 2. SST (Green)
   # {"name": "SST-cells_WT_Adult_V1", "color": "#12522eff"},
   # {"name": "SST-cells_WT_P24-P27", "color": "#428e58ff"},
   # {"name": "SST-cells_WT_P20-P23", "color": "#99b993ff"},
   # {"name": "SST-cells_WT_P15-P19", "color": "#c6e7bbff"},
    
    # 3. Pyramidal (Blue)
    #{"name": "PYR-SynGCaMP_WT_Adult_V1", "color": "#005d8aff"},
    #{"name": "PYR-PV-SynGCaMP_WT_P24-P27_V1", "color": "#008fbbff"},
    #{"name": "PYR-PV-SynGCaMP_WT_P20-P23_V1", "color": "#6caaccff"},
    #{"name": "PYR-PV-SynGCaMP_WT_P15-P19_V1", "color": "#b8cfe0ff"},
    
    # 4. PV KO (Red)
    #{"name": "PV-cells_cond-GluN1-KO_Adult_V1", "color": "#9e0004ff"},
    
    # 5. SST KO (Purple)
    {"name": "SST-cells_cond-GluN1-KO_Adult_V1", "color": "#5d1490ff"},
    {"name": "SST-cells_cond-GluN1-KO_P24-P27_V1", "color": "#9740d2ff"},
    {"name": "SST-cells_cond-GluN1-KO_P20-P23_V1", "color": "#b38fd6ff"},
    {"name": "SST-cells_cond-GluN1-KO_P15-P19_V1", "color": "#dbc0f2ff"}
]

output_dir = os.path.expanduser('~/Desktop/Final-Figures-PhD')
os.makedirs(output_dir, exist_ok=True)

print("🚀 Starting automated processing for all groups (Angle 0.0 only)...")
saved_count = 0

for dataset in datasets_to_process:
    try:
        # Generate the figure using the statistical filter function
        fig = plot_contrast_1_hatched_pie_charts(
            folder = summary_folder, 
            dataset_name = dataset["name"], 
            base_color = dataset["color"]
        )
        
        # Build a safe, unique filename for each group
        safe_filename = f"Pie_Chart_Contrast1_{dataset['name']}.svg"
        save_path = os.path.join(output_dir, safe_filename)
        
        # Export as clean editable vector file
        fig.savefig(save_path, format='svg', bbox_inches='tight')
        plt.close(fig)
        
        print(f"  ✔️ Successfully saved: {safe_filename}")
        saved_count += 1
        
    except FileNotFoundError:
        print(f"  Skipping: No file found containing '{dataset['name']}' with 'Sensitivities_' and 'angle-0.0'")
    except Exception as e:
        print(f"  Error processing '{dataset['name']}': {e}")

print(f"\n Batch run complete! {saved_count} unique charts generated and saved directly to your Desktop folder.")




#########################################################################################
##### PIE CHARTS FOR ANY CONTRAST RESPONSIVENESS (STATISTICAL SIGNIFICANCE FILTER) ######
################################# From the non deconvolved data #########################

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Publication quality styling configuration (clean SVG text)
plt.rcParams.update({
    'font.size': 11,
    'svg.fonttype': 'none',
    'axes.titlesize': 12
})


def plot_any_contrast_hatched_pie_charts(folder, dataset_name, base_color):
    """
    Calculates cell responsiveness across ANY contrast condition using
    statistical significance arrays.
    Returns the figure object so you can handle saving/showing yourself.
    """
    # Match files containing 'Sensitivities_', the dataset name, and 'angle-0.0'
    matches = [
        f for f in os.listdir(folder)
        if 'Sensitivities_' in f and dataset_name in f
        and 'angle-0.0' in f and f.endswith(".npy")
    ]
    if len(matches) == 0:
        raise FileNotFoundError(
            f"No file found for dataset '{dataset_name}' "
            f"containing 'Sensitivities_' and 'angle-0.0'")

    fpath = os.path.join(folder, matches[0])
    arr = np.load(fpath, allow_pickle=True)

    total_pos_cells = 0
    total_neg_cells = 0
    total_non_cells = 0

    # Accumulate cell counts based on the significance arrays
    for entry in arr:
        # Significant at ANY contrast column
        is_pos_significant = entry["significant_pos"].any(axis=1)
        is_neg_significant = entry["significant_neg"].any(axis=1)

        for pos_sig, neg_sig in zip(is_pos_significant, is_neg_significant):
            if pos_sig:        # significant positive response at >= 1 contrast
                total_pos_cells += 1
            elif neg_sig:      # significant negative response at >= 1 contrast
                total_neg_cells += 1
            else:              # no significant response at any contrast
                total_non_cells += 1

    grand_total = total_pos_cells + total_neg_cells + total_non_cells
    if grand_total == 0:
        raise ValueError(f"No cells found in dataset {dataset_name}")

    slices = [total_pos_cells, total_neg_cells, total_non_cells]
    categories = ['Positive', 'Negative', 'Non-Responsive']

    active_slices = []
    active_labels = []
    hatch_list = []
    face_colors = []

    for count, cat in zip(slices, categories):
        if count > 0:
            pct = (count / grand_total) * 100
            active_slices.append(count)
            active_labels.append(f"{cat}\n{count} cells\n({pct:.1f}%)")

            if cat == 'Positive':
                face_colors.append(base_color)
                hatch_list.append('')
            elif cat == 'Negative':
                rgb = mcolors.to_rgb(base_color)
                light_tint = [(c * 0.2) + 0.8 for c in rgb]
                face_colors.append(light_tint)
                hatch_list.append('////')  # stripe pattern
            else:
                face_colors.append('#f0f0f0ff')
                hatch_list.append('')

    fig, ax = plt.subplots(figsize=(5, 5))

    wedges, texts = ax.pie(
        active_slices,
        labels=active_labels,
        colors=face_colors,
        startangle=90,
        textprops=dict(fontweight='medium', fontsize=10),
        wedgeprops=dict(edgecolor='w', linewidth=1.5, antialiased=True)
    )

    # Object-level hatch styling (avoids the black artifact from global rcParams)
    for wedge, pattern in zip(wedges, hatch_list):
        if pattern:
            wedge.set_hatch(pattern)
            wedge.set_edgecolor(base_color)
            wedge.set_linewidth(1.5)

    clean_name = dataset_name.replace('Sensitivities_', '').replace('_angle-0.0', '')
    ax.set_title(
        f"Cell Responsiveness (Any Contrast)\n{clean_name}\n"
        f"(Total Population n = {grand_total} ROIs)",
        fontweight='bold', pad=20)

    return fig


# ==============================================================================
# SETTINGS
# ==============================================================================

# True  -> show each figure AND save it as SVG
# False -> only show (preview), nothing is written to disk
SAVE = True

# Groups, dataset name patterns, and colors
datasets_to_process = [
    # 1. PV
    # {"name": "PV-cells_WT_Adult_V1", "color": "#b30a7bff"},
    # {"name": "PV-cells_WT_P24-P27", "color": "#e31593ff"},
    # {"name": "PV-cells_WT_P20-P23", "color": "#ff9ab5ff"},

    # 2. SST
    # {"name": "SST-cells_WT_Adult_V1", "color": "#12522eff"},
    # {"name": "SST-cells_WT_P24-P27", "color": "#428e58ff"},
    # {"name": "SST-cells_WT_P20-P23", "color": "#99b993ff"},
    # {"name": "SST-cells_WT_P15-P19", "color": "#c6e7bbff"},

    # 3. Pyramidal
    # {"name": "PYR-SynGCaMP_WT_Adult_V1", "color": "#005d8aff"},
    # {"name": "PYR-PV-SynGCaMP_WT_P24-P27_V1", "color": "#008fbbff"},
    # {"name": "PYR-PV-SynGCaMP_WT_P20-P23_V1", "color": "#6caaccff"},
    # {"name": "PYR-PV-SynGCaMP_WT_P15-P19_V1", "color": "#b8cfe0ff"},

    # 4. PV KO
    # {"name": "PV-cells_cond-GluN1-KO_Adult_V1", "color": "#9e0004ff"},

    # 5. SST KO
    {"name": "SST-cells_cond-GluN1-KO_Adult_V1", "color": "#5d1490ff"},
    {"name": "SST-cells_cond-GluN1-KO_P24-P27_V1", "color": "#9740d2ff"},
    {"name": "SST-cells_cond-GluN1-KO_P20-P23_V1", "color": "#b38fd6ff"},
    {"name": "SST-cells_cond-GluN1-KO_P15-P19_V1", "color": "#dbc0f2ff"}
]

# Reuse summary_folder if it is already defined in your session
# (e.g. from your gain-plot code); otherwise fall back to this path.
try:
    summary_folder
except NameError:
    summary_folder = os.path.expanduser("~/CURATED/Cibele/summary")

output_dir = os.path.expanduser('~/Desktop/Final-Figures-PhD')
os.makedirs(output_dir, exist_ok=True)

# ---------- DIAGNOSTICS: check paths, files and backend ----------
print("summary_folder:", summary_folder, "| exists:", os.path.isdir(summary_folder))
print("output_dir    :", output_dir)
print("matplotlib backend:", matplotlib.get_backend())
if matplotlib.get_backend().lower() == 'agg':
    print("WARNING: 'agg' backend cannot display figures. "
          "Run %matplotlib inline (or %matplotlib qt) in a cell first.")

if os.path.isdir(summary_folder):
    sens_files = sorted(f for f in os.listdir(summary_folder)
                        if f.startswith('Sensitivities_') and 'angle-0.0' in f)
    print(f"{len(sens_files)} 'Sensitivities_*angle-0.0*.npy' files found:")
    for f in sens_files:
        print("   ", f)


# ==============================================================================
# BATCH EXECUTION: SHOW EACH FIGURE FIRST, THEN SAVE
# ==============================================================================

print("🚀 Starting automated processing for all groups (Any Contrast)...")
saved_count = 0
figs = []  # keep references so the figures stay open

for dataset in datasets_to_process:
    try:
        fig = plot_any_contrast_hatched_pie_charts(
            folder=summary_folder,
            dataset_name=dataset["name"],
            base_color=dataset["color"]
        )
        figs.append(fig)

        if SAVE:
            safe_filename = f"Pie_Chart_AnyContrast_{dataset['name']}.svg"
            save_path = os.path.join(output_dir, safe_filename)
            fig.savefig(save_path, format='svg', bbox_inches='tight')
            print(f"  ✔️ Saved: {safe_filename}")
            saved_count += 1

        # NOTE: no plt.close(fig) here, so the figure stays open for display

    except FileNotFoundError:
        print(f"  Skipping: No file found containing '{dataset['name']}' "
              f"with 'Sensitivities_' and 'angle-0.0'")
    except Exception as e:
        print(f"  Error processing '{dataset['name']}': {e}")

print(f"\nBatch run complete! {len(figs)} figures created, {saved_count} saved.")
if len(figs) == 0:
    print("No figures were created. Check the 'Skipping' / 'Error' lines above "
          "and the file list printed in the previous cell.")

# Display all figures (needed when running as a plain .py file;
# in the VS Code Interactive Window / notebook cells they also appear inline)
plt.show()
# %%
