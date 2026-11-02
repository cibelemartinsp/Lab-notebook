# %% [markdown]
# # Build Natural-Images Dataset across Conditions
#
# protocol: "10NaturalImages-2contrasts-10repeats"
#   10 natural images x 2 contrasts (0.5, 1.0) x 10 repeats, 2s presentation
#
# output (one file per condition, one entry per session):
#   ~/CURATED/Cibele/summary/NaturalImages_<quantity>_<condition>.npy

# %%
import os, sys
import multiprocessing
import numpy as np

sys.path += [os.path.join(os.path.expanduser('~'),\
                    'code', 'cibele', 'physion', 'src')]

from physion.analysis.read_NWB\
                         import scan_folder_for_NWBfiles, Data
from physion.analysis.episodes.build import EpisodeData

parallelized, debug = False, False

# load the dataset locations:
from Dataset_Organization import base_path, summary_folder, AGE_INTERVALS

from Preprocessing_Settings import get_dFoF_params
quantity = 'Deconvolved'

PROTOCOL_KEY = 'NaturalImages'

# pyramidal-cell datasets recorded with the natural-images protocol
NI_FOLDERS = [
    "PYR-SynGCaMP_WT_Adult_V1",
    "PYR-PV-SynGCaMP_WT_Young_V1",
]

def datasets_func():
    datasets = {}
    for c in NI_FOLDERS:
        datasets[c] = {'datafolder':os.path.join(base_path, c, 'NWBs'),
                       'age_interval':None}
        # we split young animals into age groups
        if 'Young' in c:
            for interval in AGE_INTERVALS:
                datasets[c.replace('Young', 'P%i-P%i' % interval)] =\
                    {'datafolder':os.path.join(base_path, c, 'NWBs'),
                     'age_interval':interval}
    return datasets

datasets = datasets_func()

# time binning of the stored trial-averaged traces (s)
TRACE_BIN = 0.1


# %%
# to be a valid dataset:
nMIN_DATAFILES = 2

def compute_natural_images_per_cells(data, Episodes,
                                     stat_test_props,
                                     response_significance_threshold=0.05,
                                     quantity='dFoF'):
    """
    returns per-ROI responses organized as (nROIs, nContrasts, nImages)
    """
    summaries = {}
    for sign in ['positive', 'negative']:
        stat_test_props['sign'] = sign
        summaries[sign] = Episodes.pre_post_statistics(\
                            response_args=dict(quantity=quantity),
                            response_significance_threshold=response_significance_threshold,
                            stat_test_props=stat_test_props,
                            repetition_keys=['repeat'],
                            loop_over_cells=True,
                            verbose=False)

    images = np.unique(summaries['positive']['Image-ID'])
    contrasts = np.unique(summaries['positive']['contrast'])

    def reshape(S, key):
        # (nROIs, nConditions) -> (nROIs, nContrasts, nImages)
        out = np.full((S[key].shape[0], len(contrasts), len(images)), np.nan)
        for k, (im, c) in enumerate(zip(S['Image-ID'], S['contrast'])):
            out[:, np.flatnonzero(contrasts==c)[0],
                   np.flatnonzero(images==im)[0]] = S[key][:, k]
        return out

    # trial-averaged traces, binned in time: (nROIs, nContrasts, nImages, nTimes)
    nbin = max(1, int(round(TRACE_BIN/(Episodes.t[1]-Episodes.t[0]))))
    nT = len(Episodes.t)//nbin
    t = Episodes.t[:nT*nbin].reshape(nT, nbin).mean(axis=1)
    resp = getattr(Episodes, quantity)  # (nEpisodes, nROIs, nTimes)
    Traces = np.zeros((resp.shape[1], len(contrasts), len(images), nT), dtype=np.float32)

    # single-trial post-pre responses (same intervals as the stat test),
    # for decoding & reliability: (nROIs, nContrasts, nImages, nRepeats), NaN-padded
    pre = Episodes.compute_interval_cond(stat_test_props['interval_pre'])
    post = Episodes.compute_interval_cond(stat_test_props['interval_post'])
    trial_values = resp[:, :, post].mean(axis=-1) - resp[:, :, pre].mean(axis=-1)
    nRepeats = max(np.sum(Episodes.find_episode_cond(key=['contrast', 'Image-ID'],
                                                     value=[c, im]))
                   for c in contrasts for im in images)
    Trials = np.full((resp.shape[1], len(contrasts), len(images), nRepeats),
                     np.nan, dtype=np.float32)

    for ic, c in enumerate(contrasts):
        for ii, im in enumerate(images):
            cond = Episodes.find_episode_cond(key=['contrast', 'Image-ID'],
                                              value=[c, im])
            Traces[:, ic, ii, :] = resp[cond, :, :nT*nbin].mean(axis=0)\
                                        .reshape(resp.shape[1], nT, nbin).mean(axis=-1)
            Trials[:, ic, ii, :cond.sum()] = trial_values[cond].T

    return {'Responses':reshape(summaries['positive'], 'value'),
            'significant_pos':reshape(summaries['positive'], 'significant').astype(bool),
            'significant_neg':reshape(summaries['negative'], 'significant').astype(bool),
            # uncorrected p-values, to correct for the 20 tests per ROI at plot time
            'pval_pos':reshape(summaries['positive'], 'pval'),
            'pval_neg':reshape(summaries['negative'], 'pval'),
            'Image-ID':images,
            'contrast':contrasts,
            't':t,
            'Traces':Traces,
            'Responses_trials':Trials}


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
    protocol_name=[p for p in data.protocols if PROTOCOL_KEY in p][0]
    try:
        data.build_dFoF(**dFoF_parameters, verbose=False)
    except AttributeError as be:
        # e.g. no ROI segmentation stored in the NWB
        print('      [X] --> discarded, problem in datafile, CHECK [!!]  (%s)' % be)
        return

    if data.nROIs>=nMIN_ROIs:

        if quantity=='Deconvolved':
            data.build_Deconvolved(quantity='dFoF')

        Episodes = EpisodeData(data,
                               quantities=[quantity],
                               protocol_name=protocol_name,
                               verbose=False)

        Summary = compute_natural_images_per_cells(data, Episodes,
                                    quantity=quantity,
                                    stat_test_props=stat_test_props,
                                    response_significance_threshold=response_significance_threshold)

        Summary['datafile'] = filename
        Summary['nROIs_original'] = data.original_nROIs
        Summary['nROIs_final'] = data.nROIs
        Summary['subject'] = data.nwbfile.subject.subject_id

        np.save(os.path.join(summary_folder, 'temp',
                             'NaturalImages-%s-%s-%i.npy' % (quantity, c, i)),
                Summary)
        print('      [v] --> included, n=%i ROIs ' % data.nROIs)

    else:
        print('      [X] --> discarded, n=%i ROIs ' % data.nROIs)

if __name__=='__main__':

    cpus = multiprocessing.cpu_count()-1 # leaving 1 cpu for the rest

    # temporary folder for parallelization
    os.makedirs(os.path.join(summary_folder, 'temp'), exist_ok=True)

    for n, c in enumerate(datasets):

        print()
        print('=================================================================')
        print('------- %i) computing : %s ' % (n+1, c))
        print('-----------------------------------------------------------------')

        DATASET = scan_folder_for_NWBfiles(datasets[c]['datafolder'])
        # filenames = [\
        #     '2025_05_28-15-31-36',
        #     '2025_05_28-16-02-19',
        #     '2025_05_28-16-31-07',
        #     '2025_05_28-17-28-58',
        #     '2025_05_28-17-56-09',
        #     '2025_05_30-13-58-09',
        #     '2025_05_30-14-24-46',
        #     '2025_05_30-14-52-22',
        #     '2025_05_30-16-04-32',
        #     '2025_06_02-14-28-43',
        #     '2025_06_02-14-54-59',
        #     '2025_06_02-15-21-08',
        #     '2025_06_02-15-47-26',
        #     '2025_06_04-10-15-37',
        #     '2025_06_04-10-43-45',
        #     '2025_06_04-11-21-27',
        #     '2025_08_13-17-41-22',
        #     '2025_08_14-14-38-53',
        #     '2025_08_14-15-05-31',
        #     '2025_08_14-15-31-15',
        #     '2025_08_14-15-56-29',
        #     '2025_08_14-16-30-06',
        #     '2025_08_14-16-56-01',
        #     '2025_08_19-14-12-33',
        #     '2025_08_20-12-51-30',
        #     '2025_08_20-13-43-55',
        #     '2025_08_20-14-15-58',
        #     '2025_08_20-14-41-11',
        #     '2025_08_20-15-07-03',
        #     '2025_08_22-14-35-45',
        #     '2025_08_22-15-01-15',
        #     '2025_08_22-15-26-18',
        #     '2025_08_22-15-51-10',
        #     '2026_04_22-14-22-07',
        #     '2026_04_24-14-58-26',
        #     '2026_04_24-15-23-49',
        #     '2026_04_28-13-13-25',
        #     '2026_04_28-14-02-51',
        #     '2026_04_28-15-07-12',
        #     '2026_04_28-15-32-23',
        #     '2026_04_28-15-57-52',
        #     '2026_04_28-16-22-36',
        #     '2026_04_29-11-06-34',
        #     '2026_04_29-11-32-23',
        #     '2026_04_29-12-16-32',
        #     '2026_04_29-14-25-40',
        #     '2026_04_29-14-50-41',
        #     '2026_04_29-15-44-40',
        #     '2026_04_30-11-41-31',
        #     '2026_04_30-12-06-33',
        #     '2026_04_30-12-32-07',
        #     '2026_04_30-12-58-29',
        #     '2026_04_30-13-23-36',
        #     '2026_04_30-13-54-07',
        #     '2026_04_30-14-19-04',
        #     '2026_04_30-14-44-04',
        #     ]
        # DATASET = scan_folder_for_NWBfiles(datasets[c]['datafolder'], from_filenames=filenames)

        # FILTER
        # 1) protocol type: natural images
        cond = np.array([np.sum([PROTOCOL_KEY in p for p in protocols])\
                        for protocols in DATASET['protocols']], dtype=bool)

        # 2) age condition
        if datasets[c]['age_interval'] is not None:
            cond = cond &\
                (DATASET['ages']>=datasets[c]['age_interval'][0]) &\
                (DATASET['ages']<=datasets[c]['age_interval'][1])

        files = DATASET['files'][cond]

        if len(files)>nMIN_DATAFILES:

            if parallelized:
                for i0 in range(0, len(files), cpus):
                    procs = [multiprocessing.Process(target=process_file,
                                                     args=(files[i], i, c, quantity))
                             for i in range(i0, min(i0+cpus, len(files)))]
                    for proc in procs:
                        proc.start()
                    for proc in procs:
                        proc.join()
            else:
                for i, f in enumerate(files):
                    process_file(f, i, c, quantity)

            # now that we have stored all datafile outputs
            Summaries = []
            for i, f in enumerate(files):
                fn = os.path.join(summary_folder, 'temp',
                                  'NaturalImages-%s-%s-%i.npy' % (quantity, c, i))
                if os.path.isfile(fn):
                    Summaries.append(np.load(fn, allow_pickle=True).item())

            # saving data
            np.save(os.path.join(summary_folder, 'NaturalImages_%s_%s.npy' % (quantity, c)),
                    Summaries)
            print(' --> saved N=%i sessions' % len(Summaries))

        else:
            print()
            print('   [!!]   DATASET NOT LARGE ENOUGH   [!!] ')
            print('               only N=%i sessions available' % len(files))
            print('   [!!]   DATASET not analyzed       [!!] ')
            print()

# %%
