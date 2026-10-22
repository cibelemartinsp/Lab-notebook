# Lab-notebook

Analysis and figure scripts for two-photon calcium imaging of V1 interneurons (SST, PV) and pyramidal cells, in WT and cond-GluN1-KO mice, across development (P15-19, P20-23, P24-27, Adult).

## Repository layout

| Folder / file | Content |
|---|---|
| `physion/` | Submodule: [yzerlaut/physion](https://github.com/yzerlaut/physion) (data reading, analysis and plotting library) |
| `utils/plot_tools/` | Submodule: [yzerlaut/plot_tools](https://github.com/yzerlaut/plot_tools) |
| `analysis/` | Scripts that read the NWB files and build the summary `.npy` files |
| `plots/` | Scripts that read the summary `.npy` files (or the NWB files) and make the figures |
| `script.sh` | Runs the three dataset analyses (`Contrast-Dataset`, `Tuning-Dataset`, `Temporal-Dynamics`) |
| `full-analysis-pipeline.sh` | Builds the NWBs, fills the DataTables and makes per-session summary PDFs |

Clone with submodules:

```
git clone --recurse-submodules https://github.com/cibelemartinsp/Lab-notebook.git
```

## Data flow

```
~/CURATED/Cibele/<dataset>/NWBs/*.nwb
        │   analysis/*.py   (datasets defined in analysis/Dataset_Organization.py)
        ▼
~/CURATED/Cibele/summary/*.npy
        │   plots/*.py
        ▼
~/Desktop/Final-Figures-PhD/*.svg
```

| Analysis script | Writes to `summary/` | Used by |
|---|---|---|
| `analysis/Tuning-Dataset.py` | `Tunings_<dataset>_contrast-<c>.npy` | `PLOT-Tuning-Dataset.py` |
| `analysis/Contrast-Dataset.py` | `Sensitivities_<quantity>_<dataset>_angle-<a>.npy` | `PLOT-Contrast-sensititvy.py`, `Tuning-WT-Young-vs-Adult.py` |
| `analysis/Temporal-Dynamics.py` | `Deconvolved_<dataset>_contrast-<c>.npy` | `PLOT-Temporal-dynamics.py` |

Young datasets are split by age group (`P15-P19`, `P20-P23`, `P24-P27`), as defined by `AGE_INTERVALS` in `analysis/Dataset_Organization.py`.

## Plot scripts (`plots/`)

All scripts are organised in `# %%` cells, to be run cell by cell (VS Code / Jupyter). The datasets to plot are chosen by commenting/uncommenting entries in the `DATASETS` / `datasets` lists.

### `PLOT-Tuning-Dataset.py` — orientation tuning

Input: `summary/Tunings_<dataset>_contrast-<c>.npy` (from `analysis/Tuning-Dataset.py`).

| Cell | Figure | Output file |
|---|---|---|
| Orientation tuning curves | Mean normalised tuning curve per group, averaged by session; prints the number of animals per group | `Orientation_Tuning_Curves_SESSION.svg` |
| Selectivity index | Orientation selectivity index per group (SST-KO P15-19 → Adult) | `Select.Index.svg` |
| Cumulative distribution | Cumulative fraction of ROI selectivity index, one figure per contrast (0.5 and 1.0) | `accumulative-distribution_0.5.svg`, `accumulative-distribution_1.0.svg` |
| Responsiveness pie charts | Fraction of non-responsive / positive / negative ROIs per group (SST WT, SST WT Ventricule, SST KO adult) | not saved (savefig commented out) |

### `PLOT-Contrast-sensititvy.py` — contrast sensitivity

Input: `summary/Sensitivities_Deconvolved_<dataset>_angle-0.0.npy` (from `analysis/Contrast-Dataset.py`).

| Cell | Figure | Output file |
|---|---|---|
| Gain of contrast sensitivity | Contrast-response curves + gain per group (averaged by session), with Mann–Whitney U tests between all pairs and significance bars; prints animals / ROI counts | `Contrast_Gain.svg` |
| Contrast responsiveness | % of positively and negatively responsive ROIs vs contrast | `Contrast_Responsiveness_Positive.svg`, `Contrast_Responsiveness_Negative.svg` |
| SST WT vs KO, all ages | Same three figures with the 8 SST groups (WT and KO at each age) side by side. **Overwrites the files above.** | `Contrast_Gain.svg`, `Contrast_Responsiveness_Positive.svg`, `Contrast_Responsiveness_Negative.svg` |
| Pie charts, contrast 1.0 | Responsive (positive / negative) vs non-responsive ROIs at full contrast, one chart per group | `Pie_Chart_Contrast1_<dataset>.svg` |
| Pie charts, any contrast | Same, counting an ROI as responsive if it responds at any contrast | `Pie_Chart_AnyContrast_<dataset>.svg` |

### `PLOT-Temporal-dynamics.py` — response time course

Input: `summary/Deconvolved_<dataset>_contrast-1.0.npy` (from `analysis/Temporal-Dynamics.py`).

Figure: peak-normalised (min-max) deconvolved response vs time from stimulus onset, for several groups (currently SST WT P24-27, SST WT Adult, PV WT Adult, PYR WT Adult). Includes a zoom on the onset (0–0.3 s) and statistics on the late window (0.8–2 s) and on the response delay.

Output: `Neuron_style_panel_D_3groups.svg`

### `Plot-Single-Contrast-Response` — example contrast responses (no `.py` extension)

Input: NWB files directly (protocol `ff-gratings-2orientations-8contrasts-15repeats`).

| Cell | Figure | Output file |
|---|---|---|
| Single session example | Interactive: lists the sessions of a dataset (default `PV-cells_WT_Young_V1/NWBs/P20-23`), you pick one; plots trial-averaged ΔF/F of example ROIs at each contrast + contrast-response inset. Press Enter to save | `TRIAL_AVG_NEGATIVE_RESP_SESS_<n>.svg` |
| Grand average | Trial-averaged response at each contrast, averaged over all ROIs of all sessions of a dataset (default `SST-cells_cond-GluN1-KO_Adult_V1`) + contrast-response inset | `CLEAN_GRAND_REPRESENTATIVE_AVERAGE.svg` |

### `traces-PLOTs.py` — example raw / deconvolved traces

Input: NWB files directly, from `~/CURATED/Cibele/<DATASET_NAME>/NWBs`.

| Cell | Figure | Output file |
|---|---|---|
| Leaderboard + interactive selector | Ranks the sessions of `DATASET_NAME` (filtered by `TARGET_PROTOCOL`) by evoked response or SNR, then lets you browse sessions and ROIs interactively; plots ΔF/F and OASIS-deconvolved traces under the visual stimulus icons. Young datasets are split by age group | `TRACES_DECONV_<dataset>[_<age>]_<session>.svg` |
| Selected combinations | Plots the hand-picked sessions / ROIs listed in `selected_combinations` (SST, PV, PYR; P20-23 and adult) together in one figure | `TRACES_COMBINED_selected_combinations.svg` |

### `Visual-Areas-Segmentation.py` — retinotopic maps (intrinsic imaging)

Input: `~/DATA/intrinsic_img/PYR_V1/PyrF_12/14-19-09/raw-maps.npy` and the vasculature image `demo-mouse.tif`.

Figures: vasculature image, altitude and azimuth retinotopic maps, visual area segmentation (sign map, areas over vasculature), phase-shift illustration. Adapted from Zhuang et al., eLife (2017).
The last cell is a test: ΔF/F and a simple Wiener deconvolution for one PV adult NWB file.

Output: `~/Desktop/fig.svg` (summary figure only).

### `Tuning-WT-Young-vs-Adult.py` — multipanel draft

Input: `summary/Sensitivities_Deconvolved_SST-cells_WT_Adult_V1_angle-{0.0,90.0}.npy`.

Draft of a multipanel figure layout (panels A–E); only the first panel is filled, with the contrast sensitivity of SST WT adults at 0° and 90°. Not saved.

## Notes

- All figures are saved as `.svg` in `~/Desktop/Final-Figures-PhD/` (folder created if missing), except where noted.
- The scripts find `physion` in different ways: `~/lab-notebook/cibele/physion/src`, `../physion/src` (run from `plots/`), or `~/DATA/physion/src`. Check the `sys.path` line at the top of each script matches where this repo is on your machine.
