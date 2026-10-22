# %% [markdown]
# # Build Sensitivity Dataset across Conditions

# %%
import os, sys , shutil 
import multiprocessing
import numpy as np

sys.path += [os.path.join(os.path.expanduser('~'),\
                    'lab-notebook', 'cibele', 'physion', 'src')]

from physion.analysis.read_NWB\
                         import scan_folder_for_NWBfiles, Data
from physion.analysis.episodes.build import EpisodeData
from physion.analysis.protocols.contrast_sensitivity\
                        import compute_sensitivity_per_cells

from physion.analysis.episodes.build import EpisodeData

parallelized, debug = False, False 

# load the dataset locations:
from Dataset_Organization import datasets_func, summary_folder
datasets = datasets_func('angle', [0., 90.])

from Preprocessing_Settings import get_dFoF_params
quantity = 'Deconvolved'

# %%
# to be a valid dataset:
nMIN_DATAFILES = 2

def process_file(filename, i, c, quantity):

    # to be a valid datafile:
    nMIN_ROIs = 4

    # CELL-dependent calcium pre-processing params 
    dFoF_parameters = get_dFoF_params(c)

    # statistical test for visually-evoked-responses
    if quantity=='Deconvolved':
        stat_test_props=dict(interval_pre=[-1.,-0.0],
                            interval_post=[0.0, 1.0],                                   
                            test='ttest',                                            
                            sign='positive')
    else:
        stat_test_props=dict(interval_pre=[-1.,0],
                            interval_post=[1.,2.],                                   
                            test='ttest',                                            
                            sign='positive')


    response_significance_threshold=5e-2

    print('%i) ' % (i+1), 'analyzing file: %s  [...] ' % filename)
    data = Data(filename, verbose=False)
    protocol_name=[p for p in data.protocols if '8contrast' in p][0]
    data.build_dFoF(**dFoF_parameters, verbose=False)

    if data.nROIs>=nMIN_ROIs:

        if quantity=='Deconvolved':
            data.build_Deconvolved()

        # try:
        if True:
            Episodes = EpisodeData(data, 
                                    quantities=[quantity],
                                    protocol_name=protocol_name, 
                                    verbose=False)

            Sensitivity = compute_sensitivity_per_cells(data, Episodes, 
                                                        quantity=quantity,
                                                        stat_test_props=stat_test_props, 
                                                        response_significance_threshold = response_significance_threshold, 
                                                        angle = float(c.split('angle-')[1][:3]))

            Sensitivity['datafile'] = filename
            Sensitivity['nROIs_original'] = data.original_nROIs
            Sensitivity['nROIs_final'] = data.nROIs
            Sensitivity['subject'] = data.nwbfile.subject.subject_id

            np.save(os.path.join(summary_folder, 'temp', 
                                 'Sensitivity-%s-%s-%i.npy' % (quantity, c, i)),
                    Sensitivity)
            print('      [v] --> included, n=%i ROIs ' % data.nROIs)
        # except BaseException as be:
        #     print('                        [-------------------------------]')
        #     print(be)
        #     print()
        #     print('      [X] --> discarded, problem in datafile, CHECK [!!]')
        #     print('                        [-------------------------------]')

    else:
        print('      [X] --> discarded, n=%i ROIs ' % data.nROIs)

if __name__=='__main__':

    import physion
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
        # 1) protocol type: contrast sensitivity
        cond = np.array([np.sum(['8contrast' in p for p in protocols])\
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
                                            args=(DATASET['files'][cond][i], 
                                                  i, c, quantity))
                        procs.append(proc)
                        proc.start()

                    # complete the processes
                    for proc in procs:
                        proc.join()
            else:
                #####################################
                ###### UN-PARALLELIZED VERSION ######
                for i, f in enumerate(DATASET['files'][cond]):
                    process_file(f, i, c, quantity)
                #####################################

            # now that we have stored all datafile outputs
            Sensitivities = []
            for i, f in enumerate(DATASET['files'][cond]):

                if os.path.isfile(os.path.join(summary_folder, 'temp', 
                                              'Sensitivity-%s-%s-%i.npy' % (quantity, c, i))):
                    Sensitivity = np.load(os.path.join(summary_folder, 'temp', 
                                                'Sensitivity-%s-%s-%i.npy' % (quantity, c, i)),
                                        allow_pickle=True).item()
                    Sensitivities.append(Sensitivity)

            # # saving data
            np.save(os.path.join(summary_folder, 'Sensitivities_%s_%s.npy' % (quantity, c)), 
                    Sensitivities)

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
#f = '/home/user/CURATED/Cibele/SST-cells_cond-GluN1-KO_Adult_V1/NWBs/2026_02_18-14-42-43.nwb'
#d = Data(f, verbose=True)

# %%
if False:
    from Dataset_Organization_cibele import summary_folder
    from physion.analysis.protocols.contrast_sensitivity\
            import plot_contrast_sensitivity, plot_contrast_responsiveness
    fig, ax = plot_contrast_sensitivity(\
                            ['PV-cells_WT_Adult_V1_contrast-1.0', 
                             'PV-cells_WT_Adult_V1_contrast-0.5'],
                            #   average_by='ROIs',
                            path=summary_folder)


# %%
