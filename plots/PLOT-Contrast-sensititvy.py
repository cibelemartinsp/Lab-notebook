# %%
import sys, os, tempfile
import numpy as np
from scipy import stats

sys.path += [os.path.join(os.path.expanduser('~'),\
                          'code', 'cibele', 'physion', 'src')]

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
            #'Deconvolved_SST-cells_cond-GluN1-KO_P15-P19_V1_angle-0.0',
            #'Deconvolved_SST-cells_cond-GluN1-KO_P20-P23_V1_angle-0.0',
            #'Deconvolved_SST-cells_cond-GluN1-KO_P24-P27_V1_angle-0.0',
            #'Deconvolved_SST-cells_cond-GluN1-KO_Adult_V1_angle-0.0',
            #'Deconvolved_SST-cells_WT_P15-P19_V1_angle-0.0',
            #'Deconvolved_SST-cells_WT_P20-P23_V1_angle-0.0',
            #'Deconvolved_SST-cells_WT_P24-P27_V1_angle-0.0',   
            #'Deconvolved_PV-cells_WT_Adult_V1_angle-0.0',
            #'Deconvolved_PYR-SynGCaMP_WT_Adult_V1_angle-0.0',
            #'Deconvolved_SST-cells_cond-GluN1-KO_Adult_V1_angle-0.0', 
            'Deconvolved_PYR-SynGCaMP_WT_Adult_V1_angle-0.0',
            ]

# ====================================================
# PLOT GAIN OF CONTRAST SENSITIVITY
# ====================================================

fig, ax = plot_contrast_sensitivity(DATASETS,\
                        #average_by='sessions',
                        average_by='ROIs',
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

#######################################################################################
### NAKA-RUSHTON FIT OF CONTRAST RESPONSE CURVES ###
#######################################################################################
#
#   R(c) = R0 + Rmax * c^n / (c^n + C50^n)
#
#   R0   : baseline / offset
#   Rmax : response range (maximum above baseline)  -> "response gain"
#   C50  : semi-saturation contrast                 -> "contrast gain"
#   n    : slope / exponent
#
#   derived from the fitted curve:
#   AUC          : area under (R - R0) over log10(contrast), tested range
#   threshold    : contrast where R crosses R0 + THRESHOLD_FRACTION*Rmax
#   dynamic range: log10(C90/C10) (contrast span between 10% and 90% of Rmax, in log units)
#
#   Runs directly on the Sensitivities_*.npy summaries (no need to re-open the NWBs).
#   NB: the summaries store post-pre responses (so baseline is already ~0) and do
#   not keep the pre-stimulus trial-by-trial values, so the "R0 + 2*sigma_baseline"
#   threshold is replaced by a fraction of Rmax.
#######################################################################################

import os, itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import mannwhitneyu, kruskal, sem

# ---------------- SETTINGS ----------------
NR_DATASETS = [
    {"name": "Deconvolved_SST-cells_WT_Adult_V1_angle-0.0", "color": "#12522eff"},
    {"name": "Deconvolved_PV-cells_WT_Adult_V1_angle-0.0", "color": "#b30a7bff"},
    {"name": "Deconvolved_PYR-SynGCaMP_WT_Adult_V1_angle-0.0", "color": "#005d8aff"},
    #{"name": "Deconvolved_SST-cells_cond-GluN1-KO_Adult_V1_angle-0.0",   "color": "#5d1490ff"},
]
NR_AVERAGE_BY = 'sessions'     # 'sessions', 'subjects' or 'ROIs'  (one fit per unit)
NR_ONLY_RESPONSIVE = False     # True -> keep only ROIs significant (positive) at >=1 contrast
THRESHOLD_FRACTION = 0.10      # threshold = R0 + 10% of Rmax
MIN_R2 = 0.5                   # fits below this R2 are excluded from the stats
C50_BOUNDS = (0.01, 2.0)       # fits with C50 or n stuck at a bound are excluded too
N_BOUNDS = (0.5, 8.0)
NR_XSCALE = 'linear'          # 'linear' or 'log' contrast axis (see set_contrast_axis below)
NR_SAVE = False
NR_OUTPUT_DIR = os.path.expanduser('~/Desktop/Final-Figures-PhD')

try:
    summary_folder
except NameError:
    summary_folder = os.path.expanduser('~/CURATED/Cibele/summary')


# ---------------- MODEL ----------------
def naka_rushton(c, R0, Rmax, C50, n):
    return R0 + Rmax * c**n / (c**n + C50**n)

def contrast_at_fraction(f, C50, n):
    """contrast where R = R0 + f*Rmax (analytical inverse of NR)"""
    return C50 * (f / (1. - f))**(1. / n)

def fit_naka_rushton(contrast, response):
    contrast, response = np.asarray(contrast, float), np.asarray(response, float)
    span = max(np.ptp(response), 1e-6)
    p0 = [response.min(), span, np.median(contrast), 2.]
    bounds = ([response.min() - span, 0.,          C50_BOUNDS[0], N_BOUNDS[0]],
              [response.max(),        10. * span,  C50_BOUNDS[1], N_BOUNDS[1]])
    try:
        popt, _ = curve_fit(naka_rushton, contrast, response,
                            p0=p0, bounds=bounds, maxfev=20000)
    except RuntimeError:
        return None
    R0, Rmax, C50, n = popt
    resid = response - naka_rushton(contrast, *popt)
    ss_tot = np.sum((response - response.mean())**2)
    R2 = 1. - np.sum(resid**2) / ss_tot if ss_tot > 0 else np.nan

    # derived metrics
    logc = np.linspace(np.log10(contrast.min()), np.log10(contrast.max()), 200)
    AUC = np.trapezoid(naka_rushton(10**logc, *popt) - R0, logc)
    C10, C90 = contrast_at_fraction(0.1, C50, n), contrast_at_fraction(0.9, C50, n)

    return dict(R0=R0, Rmax=Rmax, C50=C50, n=n, R2=R2, AUC=AUC,
                threshold=contrast_at_fraction(THRESHOLD_FRACTION, C50, n),
                C10=C10, C90=C90, dynamic_range=np.log10(C90 / C10),
                C50_in_range=contrast.min() <= C50 <= contrast.max(),
                # parameter pegged at a bound = curve not constrained by the data
                at_bound=bool(np.isclose(C50, C50_BOUNDS, rtol=1e-3).any() or
                              np.isclose(n, N_BOUNDS, rtol=1e-3).any()))


# ---------------- DATA ----------------
def load_nr_responses(name):
    Sensitivities = np.load(os.path.join(summary_folder, 'Sensitivities_%s.npy' % name),
                            allow_pickle=True)
    if NR_ONLY_RESPONSIVE:
        Sensitivities = [dict(S, Responses=S['Responses'][S['significant_pos'].any(axis=1)])
                         for S in Sensitivities]
    contrast = np.asarray(Sensitivities[0]['contrast'])
    subjects = np.array([S['subject'] for S in Sensitivities])

    if NR_AVERAGE_BY == 'sessions':
        R = [S['Responses'].mean(axis=0) for S in Sensitivities if len(S['Responses'])]
    elif NR_AVERAGE_BY == 'subjects':
        R = [np.concatenate([S['Responses'] for S, s in zip(Sensitivities, subjects)
                             if s == subj]).mean(axis=0)
             for subj in np.unique(subjects)]
    else:  # ROIs
        R = list(np.concatenate([S['Responses'] for S in Sensitivities]))
    return contrast, np.array(R)


NR_RESULTS = {}
for d in NR_DATASETS:
    contrast, R = load_nr_responses(d['name'])
    fits = [fit_naka_rushton(contrast, r) for r in R]
    good = [f for f in fits if f is not None and f['R2'] >= MIN_R2 and not f['at_bound']]
    NR_RESULTS[d['name']] = dict(contrast=contrast, responses=R, fits=fits, good=good,
                                 mean_fit=fit_naka_rushton(contrast, R.mean(axis=0)),
                                 color=d['color'])
    n_bound = sum(f is not None and f['at_bound'] for f in fits)
    print(f"{d['name']}: {len(good)}/{len(fits)} {NR_AVERAGE_BY} kept "
          f"(R2 >= {MIN_R2}, {n_bound} excluded for C50/n at bound)")


# contrast axis of the curve plots (real contrast values as tick labels in both cases)
#   'linear' : the tested contrasts are evenly spaced (0.05 -> 1.00, steps of ~0.135),
#              so each one gets its own, evenly spaced tick
#   'log'    : the classical way to show Naka-Rushton curves: the fit becomes a symmetric
#              S-shape centred on C50. Matplotlib would label it 10^-1, 10^0, so we
#              replace those with plain numbers. NB: high contrasts get squeezed on the right.
#   (only the display changes: AUC is always computed over log10(contrast))
def set_contrast_axis(ax, contrast, scale=None):
    scale = scale or globals().get('NR_XSCALE', 'linear')
    contrast = np.asarray(contrast)
    if scale == 'log':
        ax.set_xscale('log')
        ax.minorticks_off()
        ticks = [t for t in [0.05, 0.1, 0.2, 0.5, 1.] if contrast.min() <= t <= contrast.max()]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f'{t:g}' for t in ticks], fontsize=7)
    else:
        ax.set_xticks(contrast)
        ax.set_xticklabels([f'{c:.2f}' for c in contrast], rotation=45, fontsize=7)


# ---------------- FIGURE 1: curves + fits ----------------
fig1, ax = plt.subplots(figsize=(4.5, 3.5))
c_fine = np.logspace(np.log10(0.05), 0, 200)
for name, res in NR_RESULTS.items():
    m, s = res['responses'].mean(axis=0), sem(res['responses'], axis=0)
    ax.errorbar(res['contrast'], m, yerr=s, fmt='o', color=res['color'], ms=4, capsize=2)
    mf = res['mean_fit']
    if mf is not None:
        ax.plot(c_fine, naka_rushton(c_fine, mf['R0'], mf['Rmax'], mf['C50'], mf['n']),
                color=res['color'],
                label=f"{name.replace('Deconvolved_', '').replace('_angle-0.0', '')}"
                      f"  C50={mf['C50']:.2f}{' (at bound!)' if mf['at_bound'] else ''},"
                      f" Rmax={mf['Rmax']:.3f}, R²={mf['R2']:.2f}")
set_contrast_axis(ax, contrast)
ax.set_xlabel('contrast')
ax.set_ylabel('$\\delta$ response')
ax.set_title('Naka-Rushton fit (on group mean)')
ax.legend(fontsize=6, frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1.))


# ---------------- significance stars ----------------
def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''

def add_significance(ax, pairs):
    """ pairs: list of (x1, x2, p) -> bracket + stars above the data, for p<0.05 only """
    pairs = [(x1, x2, p) for x1, x2, p in pairs if p < 0.05]
    if not pairs:
        return
    y0, y1 = ax.get_ylim()
    step = 0.08 * (y1 - y0)
    for k, (x1, x2, p) in enumerate(sorted(pairs, key=lambda q: abs(q[1] - q[0]))):
        y = y1 + k * step
        ax.plot([x1, x1, x2, x2], [y - 0.3 * step, y, y, y - 0.3 * step], color='k', lw=0.8)
        ax.text((x1 + x2) / 2, y - 0.25 * step, stars(p), ha='center', va='bottom',
                fontsize=12, fontweight='bold')
    ax.set_ylim(y0, y1 + (len(pairs) + 0.5) * step)


# ---------------- FIGURE 2: parameters per unit ----------------
PARAMS = ['R0', 'Rmax', 'C50', 'n', 'AUC', 'threshold', 'dynamic_range']
fig2, AX = plt.subplots(1, len(PARAMS), figsize=(2. * len(PARAMS), 3.))
names = list(NR_RESULTS.keys())
for ax, p in zip(AX, PARAMS):
    for i, name in enumerate(names):
        vals = np.array([f[p] for f in NR_RESULTS[name]['good']])
        if len(vals) == 0:
            continue
        col = NR_RESULTS[name]['color']
        ax.bar(i, vals.mean(), yerr=sem(vals) if len(vals) > 1 else 0,
               color=col, alpha=0.4, capsize=2)
        ax.scatter(i + np.random.uniform(-0.15, 0.15, len(vals)), vals, s=8, color=col)
    ax.set_title(p)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n_.split('_')[-3] for n_ in names], rotation=60, fontsize=7)
fig2.tight_layout()


# ---------------- STATISTICS ----------------
# contrast gain  -> C50 changes ; response gain -> Rmax changes
print(f"\n=== Naka-Rushton parameters (mean ± sem over {NR_AVERAGE_BY}, good fits only) ===")
for name in names:
    good = NR_RESULTS[name]['good']
    if not good:
        print(f"{name}: no good fit"); continue
    print(f"\n{name}  (n={len(good)})")
    for p in PARAMS:
        v = np.array([f[p] for f in good])
        print(f"   {p:14s} {v.mean():8.3f} ± {sem(v) if len(v) > 1 else 0:.3f}")
    print(f"   C50 inside tested range: {sum(f['C50_in_range'] for f in good)}/{len(good)}")

for ax, p in zip(AX, PARAMS):
    groups = [[f[p] for f in NR_RESULTS[n_]['good']] for n_ in names]
    pairs = []
    groups_ok = [g for g in groups if len(g) > 1]
    if len(groups_ok) > 2:
        print(f"\n{p}: Kruskal-Wallis p = {kruskal(*groups_ok).pvalue:.4f}")
    for ((i1, n1), g1), ((i2, n2), g2) in itertools.combinations(zip(enumerate(names), groups), 2):
        if len(g1) > 1 and len(g2) > 1:
            pval = mannwhitneyu(g1, g2).pvalue
            print(f"   {p}: {n1.split('_')[-3]} vs {n2.split('_')[-3]}  "
                  f"Mann-Whitney p = {pval:.4f}  {stars(pval)}")
            pairs.append((i1, i2, pval))
    add_significance(ax, pairs)

if NR_SAVE:
    os.makedirs(NR_OUTPUT_DIR, exist_ok=True)
    fig1.savefig(os.path.join(NR_OUTPUT_DIR, 'NakaRushton_curves.svg'), bbox_inches='tight')
    fig2.savefig(os.path.join(NR_OUTPUT_DIR, 'NakaRushton_params.svg'), bbox_inches='tight')

plt.show()

# %%

#######################################################################################
### NAKA-RUSHTON: GROUP-LEVEL FIT WITH BOOTSTRAP CONFIDENCE INTERVALS ###
#######################################################################################
#
#   Per-unit fits are often unconstrained (curves not saturating by contrast 1).
#   More robust: fit the GROUP MEAN curve, and get its uncertainty by resampling
#   units (sessions / subjects / ROIs) with replacement, re-averaging, re-fitting.
#
#   -> 95% CI for each parameter of each group
#   -> group comparisons: bootstrap distribution of the difference (B - A);
#      two-sided p = 2 * min(P(diff<=0), P(diff>=0))
#
#   Requires the previous cell (NR_RESULTS, fit_naka_rushton, naka_rushton).
#######################################################################################

N_BOOT = 1000
CI = 95
BOOT_SEED = 0

# contrast axis of the curve plots (real contrast values as tick labels in both cases)
#   'linear' : the tested contrasts are evenly spaced (0.05 -> 1.00, steps of ~0.135),
#              so each one gets its own, evenly spaced tick
#   'log'    : the classical way to show Naka-Rushton curves: the fit becomes a symmetric
#              S-shape centred on C50. Matplotlib would label it 10^-1, 10^0, so we
#              replace those with plain numbers. NB: high contrasts get squeezed on the right.
#   (only the display changes: AUC is always computed over log10(contrast))
def set_contrast_axis(ax, contrast, scale=None):
    scale = scale or globals().get('NR_XSCALE', 'linear')
    contrast = np.asarray(contrast)
    if scale == 'log':
        ax.set_xscale('log')
        ax.minorticks_off()
        ticks = [t for t in [0.05, 0.1, 0.2, 0.5, 1.] if contrast.min() <= t <= contrast.max()]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f'{t:g}' for t in ticks], fontsize=7)
    else:
        ax.set_xticks(contrast)
        ax.set_xticklabels([f'{c:.2f}' for c in contrast], rotation=45, fontsize=7)
BOOT_PARAMS = ['R0', 'Rmax', 'C50', 'n', 'AUC', 'threshold', 'dynamic_range']

rng = np.random.default_rng(BOOT_SEED)

# significance stars (also defined in the previous cell, repeated here so this
# cell runs even if the previous one was executed before they were added)
def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''

def add_significance(ax, pairs):
    """ pairs: list of (x1, x2, p) -> bracket + stars above the data, for p<0.05 only """
    pairs = [(x1, x2, p) for x1, x2, p in pairs if p < 0.05]
    if not pairs:
        return
    y0, y1 = ax.get_ylim()
    step = 0.08 * (y1 - y0)
    for k, (x1, x2, p) in enumerate(sorted(pairs, key=lambda q: abs(q[1] - q[0]))):
        y = y1 + k * step
        ax.plot([x1, x1, x2, x2], [y - 0.3 * step, y, y, y - 0.3 * step], color='k', lw=0.8)
        ax.text((x1 + x2) / 2, y - 0.25 * step, stars(p), ha='center', va='bottom',
                fontsize=12, fontweight='bold')
    ax.set_ylim(y0, y1 + (len(pairs) + 0.5) * step)

for name, res in NR_RESULTS.items():
    R = res['responses']
    boot = []
    for _ in range(N_BOOT):
        f = fit_naka_rushton(res['contrast'], R[rng.integers(0, len(R), len(R))].mean(axis=0))
        if f is not None:
            boot.append(f)
    res['boot'] = {p: np.array([f[p] for f in boot]) for p in BOOT_PARAMS}
    res['boot_at_bound'] = np.mean([f['at_bound'] for f in boot])


# ---------------- PRINT: estimate [CI] per group ----------------
lo, hi = (100 - CI) / 2, 100 - (100 - CI) / 2
print(f"\n=== Group-mean Naka-Rushton fit, {N_BOOT} bootstraps over {NR_AVERAGE_BY}, {CI}% CI ===")
for name, res in NR_RESULTS.items():
    mf = res['mean_fit']
    print(f"\n{name}  (N={len(res['responses'])} {NR_AVERAGE_BY})")
    for p in BOOT_PARAMS:
        b = res['boot'][p]
        print(f"   {p:14s} {mf[p]:8.3f}   [{np.percentile(b, lo):8.3f}, {np.percentile(b, hi):8.3f}]")
    print(f"   bootstrap fits with C50/n at bound: {100 * res['boot_at_bound']:.0f}%"
          + ("   <-- C50/n poorly constrained, interpret with care" if res['boot_at_bound'] > 0.2 else ""))


# ---------------- PRINT: pairwise comparisons ----------------
print(f"\n=== Pairwise differences (B - A), bootstrap {CI}% CI and p ===")
BOOT_PAIRS = {p: [] for p in BOOT_PARAMS}
for p in BOOT_PARAMS:
    print(f"\n{p}")
    for (i1, n1), (i2, n2) in itertools.combinations(enumerate(names), 2):
        diff = NR_RESULTS[n2]['boot'][p] - NR_RESULTS[n1]['boot'][p]
        pval = min(1., 2 * min(np.mean(diff <= 0), np.mean(diff >= 0)))
        print(f"   {n2.split('_')[-3]:8s} - {n1.split('_')[-3]:8s}  "
              f"{np.median(diff):8.3f}  [{np.percentile(diff, lo):8.3f}, {np.percentile(diff, hi):8.3f}]"
              f"  p = {pval:.3f}  {stars(pval)}")
        BOOT_PAIRS[p].append((i1, i2, pval))


# ---------------- FIGURE 3: curves with bootstrap bands ----------------
fig3, AX = plt.subplots(1, 1 + len(BOOT_PARAMS), figsize=(4.5 + 1.8 * len(BOOT_PARAMS), 3.2),
                        gridspec_kw=dict(width_ratios=[2.5] + [1] * len(BOOT_PARAMS)))
ax = AX[0]
for name, res in NR_RESULTS.items():
    col, b = res['color'], res['boot']
    curves = np.array([naka_rushton(c_fine, *(b[k][i] for k in ['R0', 'Rmax', 'C50', 'n']))
                       for i in range(len(b['C50']))])
    ax.fill_between(c_fine, np.percentile(curves, lo, axis=0), np.percentile(curves, hi, axis=0),
                    color=col, alpha=0.25, lw=0)
    mf = res['mean_fit']
    ax.plot(c_fine, naka_rushton(c_fine, mf['R0'], mf['Rmax'], mf['C50'], mf['n']), color=col)
    ax.plot(res['contrast'], res['responses'].mean(axis=0), 'o', color=col, ms=3)
set_contrast_axis(ax, res['contrast'])
ax.set_xlabel('contrast')
ax.set_ylabel('$\\delta$ response')
ax.set_title(f'group fit ± {CI}% bootstrap CI', fontsize=9)

for ax, p in zip(AX[1:], BOOT_PARAMS):
    for i, name in enumerate(names):
        res = NR_RESULTS[name]
        b = res['boot'][p]
        ax.errorbar(i, res['mean_fit'][p],
                    yerr=[[res['mean_fit'][p] - np.percentile(b, lo)],
                          [np.percentile(b, hi) - res['mean_fit'][p]]],
                    fmt='o', color=res['color'], capsize=3)
    ax.set_title(p, fontsize=9)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n_.split('_')[-3] for n_ in names], rotation=60, fontsize=7)
    ax.set_xlim(-0.5, len(names) - 0.5)
    add_significance(ax, BOOT_PAIRS[p])
fig3.tight_layout()

if NR_SAVE:
    os.makedirs(NR_OUTPUT_DIR, exist_ok=True)
    fig3.savefig(os.path.join(NR_OUTPUT_DIR, 'NakaRushton_bootstrap.svg'), bbox_inches='tight')

plt.show()


# %%

#######################################################################################
### PER-ROI NAKA-RUSHTON: HOW MANY CELLS SATURATE BEFORE FULL CONTRAST? ###
#######################################################################################
#
#   Same analysis as Fig. 4 of Yoshida, Chen, Geisler & Seidemann (bioRxiv 2026),
#   "Wide dynamic range of contrast-encoding in neural responses in macaque V1":
#
#   1) one Naka-Rushton fit per ROI:  R = Rmax * c^n / (c^n + C50^n) + R0
#   2) C90 = contrast where R reaches R0 + 90% of Rmax  ->  C90 = C50 * 9^(1/n)
#   3) C90 <= 1  : the ROI saturates before full contrast
#      C90 >  1  : the ROI keeps rising up to full contrast
#                  (C90 is then extrapolated beyond the tested range, so we only
#                   know that it does NOT saturate by contrast 1, not where it would)
#
#   Differences with the cell above (and why):
#   - fits are NOT discarded when C50/n reach a bound: a high C50 is precisely what a
#     non-saturating cell looks like, so discarding it would bias the count
#   - bounds as in the paper: n in [0, 3], C50 up to 9x the max tested contrast
#     (paper: up to 900% for stimuli up to 900%; we keep the same "9x" ratio)
#   - inclusion as in the paper: R2 > ROI_MIN_R2 and a real visual response.
#     The paper uses Rmax >= 5 spikes/s and Rmax/SR >= 3, which have no equivalent
#     on post-pre deconvolved values, so we use: positive significant response at
#     >= 1 contrast (significant_pos, from the summary) + Rmax > 0.
#   - least-squares fit on trial-averaged responses (paper: maximum likelihood on spikes)
#
#   ROIs are pooled over sessions (as units are pooled in the paper).
#######################################################################################

import os, itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import mannwhitneyu, fisher_exact

# ---------------- SETTINGS ----------------
ROI_NR_DATASETS = [
    {"name": "Deconvolved_SST-cells_WT_Adult_V1_angle-0.0",           "color": "#12522eff"},
    {"name": "Deconvolved_SST-cells_cond-GluN1-KO_Adult_V1_angle-0.0", "color": "#5d1490ff"},
    {"name": "Deconvolved_PV-cells_WT_Adult_V1_angle-0.0",            "color": "#c0392bff"},
    {"name": "Deconvolved_PYR-SynGCaMP_WT_Adult_V1_angle-0.0",        "color": "#7f7f7fff"},
]
ROI_N_BOUNDS = (0.05, 3.)        # paper: 0 - 3 (0 excluded: flat curve, C90 undefined)
ROI_C50_BOUNDS = (0.01, 9.)      # paper: 0 - 900% for stimuli up to 900%  -> 9x max contrast
ROI_MIN_R2 = 0.8                 # paper: R2 > 0.8
ROI_ONLY_RESPONSIVE = True       # keep ROIs with a positive significant response at >=1 contrast
ROI_NR_SAVE = False
ROI_NR_OUTPUT_DIR = os.path.expanduser('~/Desktop/Final-Figures-PhD')

try:
    summary_folder
except NameError:
    summary_folder = os.path.expanduser('~/CURATED/Cibele/summary')


# ---------------- FIT ----------------
def nr_curve(c, R0, Rmax, C50, n):
    return R0 + Rmax * c**n / (c**n + C50**n)

def fit_roi(contrast, response):
    span = max(np.ptp(response), 1e-9)
    p0 = [response.min(), span, np.median(contrast), 1.5]
    bounds = ([response.min() - span, 0.,         ROI_C50_BOUNDS[0], ROI_N_BOUNDS[0]],
              [response.max(),        10. * span, ROI_C50_BOUNDS[1], ROI_N_BOUNDS[1]])
    try:
        popt, _ = curve_fit(nr_curve, contrast, response, p0=p0, bounds=bounds, maxfev=20000)
    except (RuntimeError, ValueError):
        return None
    R0, Rmax, C50, n = popt
    ss_tot = np.sum((response - response.mean())**2)
    R2 = 1. - np.sum((response - nr_curve(contrast, *popt))**2) / ss_tot if ss_tot > 0 else np.nan
    return dict(R0=R0, Rmax=Rmax, C50=C50, n=n, R2=R2,
                C90=C50 * 9.**(1. / n))     # R0 + 0.9*Rmax  <=>  (c/C50)^n = 9


ROI_NR = {}
for d in ROI_NR_DATASETS:
    Sensitivities = np.load(os.path.join(summary_folder, 'Sensitivities_%s.npy' % d['name']),
                            allow_pickle=True)
    contrast = np.asarray(Sensitivities[0]['contrast'], float)
    R = np.concatenate([S['Responses'] for S in Sensitivities])
    responsive = np.concatenate([S['significant_pos'].any(axis=1) for S in Sensitivities])
    if ROI_ONLY_RESPONSIVE:
        R = R[responsive]
    fits = [fit_roi(contrast, r) for r in R]
    keep = np.array([f is not None and f['R2'] > ROI_MIN_R2 and f['Rmax'] > 0 for f in fits])
    P = {k: np.array([f[k] for f, ok in zip(fits, keep) if ok])
         for k in ['R0', 'Rmax', 'C50', 'n', 'R2', 'C90']}
    ROI_NR[d['name']] = dict(P, color=d['color'], contrast=contrast,
                             responses=R[keep], n_total=len(responsive), n_tested=len(R),
                             saturating=P['C90'] <= contrast.max())
    print(f"{d['name']}: {keep.sum()}/{len(R)} ROIs kept (R2 > {ROI_MIN_R2})"
          + (f"  [{len(R)}/{len(responsive)} ROIs responsive]" if ROI_ONLY_RESPONSIVE else ''))

roi_names = list(ROI_NR.keys())
short = lambda name: name.replace('Deconvolved_', '').replace('_V1_angle-0.0', '')


# ---------------- PRINT: % saturating ----------------
print(f"\n=== Per-ROI Naka-Rushton: C90 <= {contrast.max():g} (saturate) vs > (keep rising) ===")
for name in roi_names:
    r = ROI_NR[name]
    N, ns = len(r['C90']), r['saturating'].sum()
    if N == 0:
        print(f"{short(name):35s} no ROI kept"); continue
    print(f"{short(name):35s} N={N:4d}   saturate: {ns:4d} ({100 * ns / N:4.1f}%)   "
          f"keep rising: {N - ns:4d} ({100 * (N - ns) / N:4.1f}%)   "
          f"median C50={np.median(r['C50']):.2f}  n={np.median(r['n']):.2f}  "
          f"C50 at upper bound: {np.mean(np.isclose(r['C50'], ROI_C50_BOUNDS[1], rtol=1e-3)) * 100:.0f}%")

print("\n=== Group comparisons ===")
print("   % saturating: Fisher exact test ; C50, n: Mann-Whitney U")
for n1, n2 in itertools.combinations(roi_names, 2):
    a, b = ROI_NR[n1], ROI_NR[n2]
    if len(a['C90']) == 0 or len(b['C90']) == 0:
        continue
    table = [[a['saturating'].sum(), (~a['saturating']).sum()],
             [b['saturating'].sum(), (~b['saturating']).sum()]]
    p_sat = fisher_exact(table)[1]
    p_c50 = mannwhitneyu(a['C50'], b['C50']).pvalue
    p_n = mannwhitneyu(a['n'], b['n']).pvalue
    print(f"   {short(n1)} vs {short(n2)}:  %sat p={p_sat:.2g}   C50 p={p_c50:.2g}   n p={p_n:.2g}")


# ---------------- FIGURE 4-like ----------------
#   row 1 (paper Fig. 4A): C50 vs n, one dot per ROI (dashed line = max tested contrast)
#   row 2 (paper Fig. 4B): distribution of C90 (dashed line = max tested contrast)
#   row 3 (paper Fig. 5C-like): mean normalised curve of saturating vs non-saturating ROIs
fig4, AX = plt.subplots(3, len(roi_names), figsize=(3. * len(roi_names), 8.3), squeeze=False)
c_fine = np.linspace(contrast.min(), contrast.max(), 200)
for j, name in enumerate(roi_names):
    r = ROI_NR[name]
    col, cmax = r['color'], r['contrast'].max()
    sat = r['saturating']

    ax = AX[0, j]
    ax.scatter(r['n'][~sat], r['C50'][~sat], s=6, color=col, alpha=0.35, lw=0, label='C90 > 1')
    ax.scatter(r['n'][sat], r['C50'][sat], s=6, color='k', alpha=0.6, lw=0, label='C90 ≤ 1')
    ax.axhline(cmax, ls='--', color='grey', lw=0.8)
    ax.set_yscale('log')
    ax.set_ylim(ROI_C50_BOUNDS[0] * 0.8, ROI_C50_BOUNDS[1] * 1.3)
    ax.set_xlim(0, ROI_N_BOUNDS[1] + 0.1)
    ax.set_xlabel('n')
    ax.set_title(f"{short(name)}\nN={len(sat)} ROIs", fontsize=8)
    if j == 0:
        ax.set_ylabel('C50')
        ax.legend(fontsize=6, frameon=False, markerscale=2)

    ax = AX[1, j]
    if len(sat):
        logC90 = np.log10(np.clip(r['C90'], 1e-2, 1e3))
        ax.hist(logC90, bins=np.linspace(-2, 3, 41), color=col, alpha=0.7)
        ax.axvline(np.log10(cmax), ls='--', color='grey', lw=0.8)
        ax.set_title(f"saturate before {cmax:g}: {100 * sat.mean():.0f}%", fontsize=8)
    ax.set_xticks([-2, -1, 0, 1, 2, 3])
    ax.set_xticklabels(['0.01', '0.1', '1', '10', '100', '≥1000'], fontsize=7)
    ax.set_xlabel('C90 (contrast)')
    if j == 0:
        ax.set_ylabel('# ROIs')

    ax = AX[2, j]
    for mask, c_, lab in [(sat, 'k', 'C90 ≤ 1'), (~sat, col, 'C90 > 1')]:
        if mask.sum() == 0:
            continue
        Rn = r['responses'][mask] / np.abs(r['responses'][mask]).max(axis=1, keepdims=True)
        m, s = Rn.mean(axis=0), Rn.std(axis=0) / np.sqrt(len(Rn))
        ax.errorbar(r['contrast'], m, yerr=s, fmt='o', ms=3, color=c_, capsize=2,
                    label=f"{lab} (n={mask.sum()})")
    ax.set_xlabel('contrast')
    ax.set_xticks(r['contrast'])
    ax.set_xticklabels([f'{c:.2f}' for c in r['contrast']], rotation=45, fontsize=6)
    if j == 0:
        ax.set_ylabel('normalised response')
    ax.legend(fontsize=6, frameon=False)
fig4.tight_layout()


# ---------------- FIGURE: % saturating per group ----------------
fig5, ax = plt.subplots(figsize=(1. + 0.8 * len(roi_names), 3.))
for i, name in enumerate(roi_names):
    sat = ROI_NR[name]['saturating']
    if len(sat):
        ax.bar(i, 100 * sat.mean(), color=ROI_NR[name]['color'])
        ax.text(i, 100 * sat.mean() + 1, f"{sat.sum()}/{len(sat)}", ha='center', fontsize=6)
ax.set_xticks(range(len(roi_names)))
ax.set_xticklabels([short(n_) for n_ in roi_names], rotation=60, ha='right', fontsize=7)
ax.set_ylabel('% ROIs saturating\nbefore full contrast (C90 ≤ 1)')
ax.set_ylim(0, 100)
fig5.tight_layout()

if ROI_NR_SAVE:
    os.makedirs(ROI_NR_OUTPUT_DIR, exist_ok=True)
    fig4.savefig(os.path.join(ROI_NR_OUTPUT_DIR, 'NakaRushton_perROI_C50-n-C90.svg'), bbox_inches='tight')
    fig5.savefig(os.path.join(ROI_NR_OUTPUT_DIR, 'NakaRushton_perROI_fraction-saturating.svg'), bbox_inches='tight')

plt.show()

# %%
