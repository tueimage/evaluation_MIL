import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_results(config):
    
    path_results = config['results_path']


    # bag_auc     = [0.74, 0.84, 0.99]
    # apj         = [0.23, 0.66, 0.54]
    # sp          = [0.32, 0.88, 0.82]
    # data_names  = ['X-Ray', 'MURA', 'Pascal VOC']
    # scores      = ['apj', 's']

    d = { 'Dataset':          ['X-Ray', 'MURA', 'Pascal VOC', 'X-Ray', 'MURA', 'Pascal VOC'],
          'Stability Index':  ['APJ',   'APJ',  'APJ',        'S',     'S',    'S'         ],
          'Stability Score':  [0.23,    0.61,   0.54,         0.32,    0.84,   0.82        ],
          'Bag AUC':          [0.72,    0.84,   0.99,         0.72,    0.84,   0.99        ]}
    
    df = pd.DataFrame(data=d)
    del d
    
    sns.set_style("whitegrid")
    fig = plt.figure()
    ax = sns.scatterplot(x     = 'Stability Score',
                         y     = 'Bag AUC',
                         data  = df,
                         style = 'Stability Index',
                         hue   = 'Dataset',
                         s     = 100)
    plt.xlim([0, 1])
    plt.ylim([0, 1.1])
    # ax = sns.scatterplot(x=apj, y=bag_auc, style=data_names)
    fig.savefig(path_results + 'scatter.jpg', bbox_inches='tight', dpi=300)
    plt.show()