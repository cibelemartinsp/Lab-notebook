# %% [markdown]
# # Build Tuning Dataset across Conditions

# %%
import os, sys , shutil 
import multiprocessing
import numpy as np

sys.path += [os.path.join(os.path.expanduser('~'),\
                          'lab-notebook', 'cibele', 'physion', 'src')]
from physion.analysis.read_NWB\
                import scan_folder_for_NWBfiles, Data
from physion.analysis.episodes.build import EpisodeData
from physion.analysis.protocols.orientation_tuning\
                import compute_tuning_response_per_cells

parallelized, debug = False, False 

# load the dataset locations:
from Dataset_Organization import datasets_func, summary_folder
datasets = datasets_func('contrast', [0.5, 1.0])

from Preprocessing_Settings import get_dFoF_params

# %%
def process_file(filename, i, c):

    # to be a valid datafile:
    nMIN_ROIs = 4

    # CELL-dependent calcium pre-processing params 
    dFoF_parameters = get_dFoF_params(c)

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

    # FIX: Temporarily disable running_speed to bypass the resampling unpack bug
    quantities = ['dFoF']
    # if 'Running-Speed' in data.nwbfile.acquisition:
    #     quantities += ['running_speed']

    if data.nROIs>=nMIN_ROIs:

        try:
            Episodes = EpisodeData(data, 
                                    quantities=quantities,
                                    protocol_name=protocol_name, 
                                    verbose=debug)

            Tuning = compute_tuning_response_per_cells(data, Episodes, 
                                                        quantity='dFoF', 
                                                        stat_test_props = stat_test_props, 
                                                        response_significance_threshold =\
                                                            response_significance_threshold, 
                                                        contrast =\
                                                            float(c.split('contrast-')[1][:3]),
                                                        verbose=debug)
            Tuning['datafile'] = filename
            Tuning['nROIs_original'] = data.original_nROIs
            Tuning['nROIs_final'] = data.nROIs
            Tuning['nROIs_responsive'] = np.sum(Tuning['significant_ROIs'])
            Tuning['subject'] = data.nwbfile.subject.subject_id

            np.save(os.path.join(summary_folder, 'temp', 
                                 'Tuning-%s-%i.npy' % (c, i)),
                    Tuning)
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

        #   FILTER:
        # ----------
        # 1) protocol type: orientation tuning
        cond = np.array([np.sum(['8orientation' in p for p in protocols])\
                        for protocols in DATASET['protocols']], dtype=bool)
        # 2) age condition (not None only if young)
        if (datasets[c]['age_interval'] is not None):
            cond = cond &\
                (DATASET['ages']>=datasets[c]['age_interval'][0]) &\
                (DATASET['ages']<=datasets[c]['age_interval'][1])

        if np.sum(cond)>0:
            # MINIMUM NUMBER OF SESSION (1 for now)

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
            Tunings = []
            for i, f in enumerate(DATASET['files'][cond]):

                if os.path.isfile(os.path.join(summary_folder, 'temp', 
                                                'Tuning-%s-%i.npy' % (c, i))):
                    Tuning = np.load(os.path.join(summary_folder, 'temp', 
                                                'Tuning-%s-%i.npy' % (c, i)),
                                        allow_pickle=True).item()
                    Tunings.append(Tuning)

            # # saving data
            np.save(os.path.join(summary_folder, 'Tunings_%s.npy' % c), Tunings)

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
    from Dataset_Organization_cibele import summary_folder
    from physion.analysis.protocols.orientation_tuning\
        import plot_orientation_tuning_curve, plot_selectivity

    fig, ax = plot_selectivity(\
                            ['PV-cells_WT_Adult_V1_contrast-1.0', 
                             'PV-cells_WT_Adult_V1_contrast-0.5'],
                            #   average_by='ROIs',
                            #  using='fit',
                            path=summary_folder)
        
    fig, ax = plot_orientation_tuning_curve(\
                            ['PV-cells_WT_Adult_V1_contrast-1.0', 
                             'PV-cells_WT_Adult_V1_contrast-0.5'],
                                            # average_by='ROIs',
                            path=summary_folder)
    # %%
