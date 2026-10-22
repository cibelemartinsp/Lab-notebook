    
def get_dFoF_params(dataset):

    if dataset[:4]=='PYR-':
        # means pyramidal cells
        dFoF_parameters = dict(\
                roi_to_neuropil_fluo_inclusion_factor=0., # no factor here
                neuropil_correction_factor = 0.7,
                method_for_F0 = 'sliding_percentile',
                percentile=5., # percent
                sliding_window = 5*60, # seconds
        )
    else:
        # means interneurons
        dFoF_parameters = dict(\
                roi_to_neuropil_fluo_inclusion_factor=1.15,
                neuropil_correction_factor = 0.7,
                method_for_F0 = 'sliding_percentile',
                percentile=5., # percent
                sliding_window = 5*60, # seconds
        )

    return dFoF_parameters
