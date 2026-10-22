# %% [markdown]
# # Build Deconvolved Responses across Conditions

# %%
import os

import os, sys , shutil 
import multiprocessing
import numpy as np

sys.path += [os.path.join(os.path.expanduser('~'),\
                'lab-notebook', 'cibele', 'physion', 'src')]

from physion.analysis.read_NWB\
                import scan_folder_for_NWBfiles, Data
from physion.analysis.episodes.build import EpisodeData

parallelized, debug = False, False 

# load the dataset locations:
from Dataset_Organization import datasets_func, summary_folder
datasets = datasets_func('contrast', [0.5, 1.0])

from Preprocessing_Settings import get_dFoF_params

# to be a valid dataset:
nMIN_DATAFILES = 2
                
# %%
def process_file(filename, i, c):

    # to be a valid datafile:
    nMIN_ROIs = 4

    # CELL-dependent calcium pre-processing params 
    dFoF_parameters = get_dFoF_params(c)

    TAU_DECONVOLUTION = 0.8

    # statistical test for visually-evoked-responses
    stat_test_props=dict(interval_pre=[-1.,0],
                         interval_post=[1.,2.],                                   
                         test='ttest',                                            
                         sign='positive')

    response_significance_threshold=5e-2

    print('%i) ' % (i+1), 'analyzing file: %s  [...] ' % filename)
    data = Data(filename, verbose=False)
    protocol_name=[p for p in data.protocols if '8orientation' in p][0]
    data.build_dFoF(**dFoF_parameters, verbose=False)

    quantities = ['Deconvolved']
    #if 'Running-Speed' in data.nwbfile.acquisition:
    #    quantities += ['running_speed']

    if data.nROIs>=nMIN_ROIs:

        try:

            # deconvolve first:
            data.build_Deconvolved(Tau=TAU_DECONVOLUTION)
            # process episodes
            Episodes = EpisodeData(data, 
                                    quantities=quantities,
                                    protocol_name=protocol_name, 
                                    verbose=debug)

            contrast = float(c.split('contrast-')[1][:3])
            epCond = Episodes.find_episode_cond(key='contrast',
                                                value=contrast)

            significant = np.zeros(data.nROIs, dtype=bool)
            for r in range(data.nROIs):
                    cell_resp = Episodes.pre_post_statistics(stat_test_props=stat_test_props,
                                                             episode_cond=epCond,
                                                             repetition_keys=['repeat', 'angle'],
                                                             response_significance_threshold=\
                                                                response_significance_threshold,
                                                             response_args=dict(quantity='Deconvolved',
                                                                                index=r),
                                                            verbose=debug)
                    cond = (cell_resp['contrast']==contrast)
                    significant[r] = cell_resp['significant'][cond][0]
                    
            Response = {
                    't':Episodes.t,
                    'Deconvolved':Episodes.Deconvolved[epCond,:,:].mean(axis=0),
                    'significant':significant,
                    'nROIs_original': data.original_nROIs,
                    'nROIs_final': data.nROIs,
                    'subject':data.nwbfile.subject.subject_id,
            }

            np.save(os.path.join(summary_folder, 'temp', 
                                 'tempResponse-%s-%i.npy' % (c, i)),
                    Response)
            print('      [v] --> included, n=%i ROIs ' % data.nROIs)

        except BaseException as be:

            print('                        [-------------------------------]')
            print(be)
            print()
            print(filename)
            print('nROIs=%i' % data.nROIs, ', protocols=%s' % data.protocols)
            print(Episodes.varied_parameters)
            print('      [X] --> discarded, problem in datafile, CHECK [!!]')
            print('                        [-------------------------------]')

    else:
        print('      [X] --> discarded, n=%i ROIs ' % data.nROIs)


if __name__=='__main__':

    from physion.assembling.dataset import read_spreadsheet

    cpus = multiprocessing.cpu_count()-1 # leaving 1 cpu for the rest

    # temporary folder for parallelization
    os.makedirs(os.path.join(summary_folder, 'temp'), exist_ok=True)

    Nstart = 0
    Nend = len(datasets)

    for n in range(Nstart, Nend):

        c = list(datasets.keys())[n]

        table = datasets[c]['datafolder'].replace('NWBs', 'DataTable.xlsx')

        dataset_table, subjects_table, analysis =\
                read_spreadsheet(table, get_metadata_from='table')
        print()
        print()
        print('=================================================================')
        print('-----------------------------------------------------------------')
        print('------- %i) computing : %s ' % (n+1, c))
        print('-----------------------------------------------------------------')
        print()

        DATASET = scan_folder_for_NWBfiles(datasets[c]['datafolder'])
        
        # FILTER
        # 1) protocol type: orientation tuning
        cond = np.array([np.sum(['8orientation' in p for p in protocols])\
                        for protocols in DATASET['protocols']], dtype=bool)
        # 2) age condition
        if datasets[c]['age_interval'] is not None:
            cond = cond &\
                (DATASET['ages']>=datasets[c]['age_interval'][0]) &\
                (DATASET['ages']<=datasets[c]['age_interval'][1])


        if len(DATASET['files'][cond])>nMIN_DATAFILES:

            if parallelized:
                ################################################
                ###    parallelization here !   #################
                ################################################

                nruns = int(len(DATASET['files'][cond])/cpus)+1

                for r in range(nruns):
                    i0 = r*cpus
                    imax = np.min([i0+cpus, len(DATASET['files'][cond])]) 
                    print(' - running set of files %i:%i' % (i0, imax))

                    # start the processes
                    procs = []
                    for i in range(i0,imax):
                        proc = multiprocessing.Process(\
                                            target=process_file, 
                                            args=(DATASET['files'][cond][i], i, c))
                        procs.append(proc)
                        proc.start()

                    # complete the processes
                    for proc in procs:
                        proc.join()

            else:
                #####################################
                ###### UN-PARALLELIZED VERSION ######
                for i, f in enumerate(DATASET['files'][cond]):
                    process_file(f, i, c)
                #####################################

            # now that we have stored all datafile outputs
            Responses = []
            for i, f in enumerate(DATASET['files'][cond]):

                if os.path.isfile(os.path.join(summary_folder, 'temp', 
                                              'tempResponse-%s-%i.npy' % (c, i))):
                    Response = np.load(os.path.join(summary_folder, 'temp', 
                                                'tempResponse-%s-%i.npy' % (c, i)),
                                        allow_pickle=True).item()
                    Responses.append(Response)

            # # saving data
            np.save(os.path.join(summary_folder, 'Deconvolved_%s.npy' % c), 
                    Responses)

        else:
            print()
            print('   [!!]   DATASET NOT LARGE ENOUGH   [!!] ')
            print('               only N=%i sessions available' %\
                                        len(DATASET['files'][cond]))
            print('   [!!]   DATASET not analyzed       [!!] ')
            print()

        print('-----------------------------------------------------------------')
        print('=================================================================')
    # shutil.rmtree(os.path.join(summary_folder, 'temp'))

# %%
if False:

    import sys, os
    sys.path += ['physion/src']
    import physion.utils.plot_tools as pt
    from scipy import stats
    from physion.dataviz.episodes.temporal_dynamics\
          import plot_response_dynamics

    summary_folder = os.path.join(os.path.expanduser('~'), 
                                'CURATED', 'Cibele', 'summary')
 
    fig, ax = plot_response_dynamics(\
                        ['PV-cells_WT_Adult_V1_contrast-1.0', 
                         'PV-cells_WT_Adult_V1_contrast-0.5'],
                         colors=['tab:red', 'lightpink'],
                        # average_by='ROIs',
                        average_by='subjects',
                        path=summary_folder)
    

# %%
