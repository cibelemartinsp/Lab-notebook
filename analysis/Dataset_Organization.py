import os


folders = [
    #"SST-cells_cond-GluN1-KO_Adult_V1",
    #"PV-cells_WT_Adult_V1", 
    #"PV-cells_WT_Young_V1", 
    #"PV-cells_cond-GluN1-KO_Adult_V1", 
    #"PYR-PV-SynGCaMP_WT_Young_V1",
    "PYR-SynGCaMP_WT_Adult_V1",
    #"SST-cells_cond-GluN1-KO_Young_V1",
    #"SST-cells_WT_Adult_V1",
    #"SST-cells_WT_Young_V1",
    #"SST-cells_cond-GluN1-KO_Adult_V1_Taddy",
    #"SST-cells_WT_Adult_V1_Taddy"
]
base_path = os.path.expanduser('~/CURATED/Cibele/')

summary_folder = os.path.join(os.path.expanduser('~'), 
                              'CURATED', 'Cibele', 'summary')


# age intervals in Yound
AGE_INTERVALS = [\
    (15,19), (20,23), (24,27)]

# to be a valid dataset:
nMIN_DATAFILES = 2

parallelized = False
debug = False

def datasets_func(\
        key='contrast',
        values=[0.5,1.0]):

    datasets = {}
    for c in folders:

        for val in values:

            datasets[c+'_%s-%.1f' % (key, val)] =\
                {'datafolder':os.path.join(base_path, c, 'NWBs'), 
                    'age_interval':None}
            
            # we split young animals into age groups
            if 'Young' in c:
                for interval in AGE_INTERVALS:
                    datasets[c.replace('Young', 'P%i-P%i' % interval)+'_%s-%.1f' % (key,val)] =\
                        {'datafolder':os.path.join(base_path, c, 'NWBs'), 
                            'age_interval':interval}
    return datasets

if __name__=='__main__':
    from pprint import pprint
    pprint(datasets)

