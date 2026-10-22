# %%
import os, sys 
import numpy as np
import matplotlib.pylab as plt

sys.path += [os.path.join(os.path.expanduser('~'),\
                          'lab-notebook', 'cibele', 'physion', 'src')]
from physion.analysis.read_NWB\
                import scan_folder_for_NWBfiles, Data
from physion.analysis.episodes.build import EpisodeData
from physion.utils import plot_tools as pt

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
from physion.analysis.episodes.build import EpisodeData
from physion.dataviz.episodes.trial_average import plot as plot_trial_average

def cell_sensitivity_example_fig(filename,
                                 angle=0,
                                stat_test_props = dict(interval_pre=[-1,0], 
                                                       interval_post=[1,2],
                                                       test='ttest',
                                                       sign='positive'),
                                response_significance_threshold = 0.01,
                                Nsamples = 10, # how many cells we show
                                 color='k',
                                seed=10):
    
    np.random.seed(seed)
    
    data = Data(filename)
    data.build_visual_stim()

    EPISODES = EpisodeData(data,
                           quantities=['dFoF'],
                           protocol_id=np.flatnonzero(['8contrasts' in p for p in data.protocols]),
                        #    with_visual_stim=True,
                           verbose=True)
    #EPISODES.build_visual_stim(data) 
    fig, AX = pt.plt.subplots(Nsamples, len(EPISODES.varied_parameters['contrast']), 
                          figsize=(7,7))
    plt.subplots_adjust(right=0.75, left=0.1, top=0.97, bottom=0.05, wspace=0.1, hspace=0.8)
    
    for Ax in AX:
        for ax in Ax:
            ax.axis('off')

    for i, r in enumerate(np.random.choice(np.arange(data.nROIs), 
                                           min([Nsamples, data.nROIs]), replace=False)):

        # SHOW trial-average
        plot_trial_average(EPISODES,
                           condition=(EPISODES.angle==angle),
                           column_key='contrast',
                           #color_key='contrast',
                           color=color,
                           quantity='dFoF',
                           Ybar=1., Ybar_label='1dF/F',
                           Xbar=1., Xbar_label='1s',
                           index=r,
                           with_stat_test=True,
                           stat_test_props=stat_test_props,
                           with_screen_inset=False,
                           AX=[AX[i]], no_set=False)
        AX[i][0].annotate('roi #%i  ' % (r+1), (0,0), ha='right', xycoords='axes fraction')

        # SHOW summary angle dependence
        inset = pt.inset(AX[i][-1], (2.2, 0.2, 1.2, 0.8))

        contrasts, y, sy, responsive_contrasts = [], [], [], []
        responsive = False

        for c, contrast in enumerate(EPISODES.varied_parameters['contrast']):

            stats = EPISODES.stat_test_for_evoked_responses(episode_cond=\
                                            EPISODES.find_episode_cond(key=['angle', 'contrast'],
                                                                       value=[angle, contrast]),
                                                            response_args=dict(quantity='dFoF', index=r),
                                                            **stat_test_props)

            contrasts.append(contrast)
            y.append(np.mean(stats.y-stats.x))    # means "post-pre"
            sy.append(np.std(stats.y-stats.x))    # std "post-pre"

            if stats.significant(threshold=response_significance_threshold):
                responsive = True
                responsive_contrasts.append(contrast)

        pt.scatter(contrasts, np.array(y), 
                   sy=np.array(sy), ax=inset, ms=1, lw=1, color=color)
        inset.plot(contrasts, 0*np.array(contrasts), 'k:', lw=0.5)
        inset.set_ylabel('$\\delta$ $\\Delta$F/F     ', fontsize=7)
        inset.set_xticks([0,1])
        #inset.set_xticklabels(['%i'%a if (i%2==0) else '' for i, a in enumerate(contrasts)], fontsize=7)
    inset.set_xlabel('contrast', fontsize=7)

        
    return fig

iSession = 0 # session index
fig = cell_sensitivity_example_fig(\
    DATASET['files'][iSession])
plt.show()
# %%
