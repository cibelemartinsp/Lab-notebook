# %%
import os, sys 
import numpy as np

sys.path += [os.path.join(os.path.expanduser('~'),\
                          'lab-notebook', 'cibele', 'physion', 'src')]
from physion.analysis.read_NWB\
                import scan_folder_for_NWBfiles, Data
from physion.analysis.episodes.build import EpisodeData
from physion.analysis.protocols.orientation_tuning\
                import compute_tuning_response_per_cells

DATASET = scan_folder_for_NWBfiles(
    os.path.join(
        os.path.expanduser('~'),
        'CURATED', 'Cibele', 'SST-cells_WT_Adult_V1' , 'NWBs',
    ),
    for_protocol='ff-gratings-2orientations-8contrasts-15repeats',
)

from Preprocessing_Settings import get_dFoF_params
dFoF_params = get_dFoF_params('SST')


# %%
def process_file(filename, i, c):

    # to be a valid datafile:
    nMIN_ROIs = 4


# %%
