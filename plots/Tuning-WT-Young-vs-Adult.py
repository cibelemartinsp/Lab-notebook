# %%
import os, sys
import numpy as np

sys.path += [os.path.join(os.path.expanduser('~'),\
                    'lab-notebook', 'cibele', 'physion', 'src')]

from physion.utils import plot_tools as pt
pt.set_style('manuscript')

# %%

panels = {
    'g1': {'left': 1.5, 'bottom': 4, 'width': 2., 'height': 1.4},
    # 'g2': {'left': 5.5, 'bottom': 4, 'width': 2., 'height': 1.4},
    # 'g3': {'left': 9.5, 'bottom': 4, 'width': 2., 'height': 1.4},
    # 'g4': {'left': 13.5, 'bottom': 4, 'width': 2., 'height': 1.4},
    # 'g5': {'left': 1.5, 'bottom': 1.1, 'width': 15., 'height': 1.4},
}
labels = {
    'A': {'left': 1.1, 'bottom': 5.8},
    'B': {'left': 5.1, 'bottom': 5.8},
    'C': {'left': 9.1, 'bottom': 5.8},
    'D': {'left': 13.1, 'bottom': 5.8},
    'E': {'left': 1.1, 'bottom': 2.9},
}

fig = pt.multipanel_figure(panels, 
                           figsize=('2-columns', 12))
pt.add_labels(fig, labels)

# %%
sys.path += ['../analysis']
from Dataset_Organization import summary_folder
from physion.analysis.protocols.contrast_sensitivity\
        import plot_contrast_sensitivity, plot_contrast_responsiveness
fig, ax = plot_contrast_sensitivity(\
                        ['Deconvolved_SST-cells_WT_Adult_V1_angle-0.0',
                         'Deconvolved_SST-cells_WT_Adult_V1_angle-90.0'],
                        #   average_by='ROIs',
                        ax=panels['g1']['ax'],
                        path=summary_folder)

# %%
