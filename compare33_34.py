import OneDCNN33 as func1
import OneDCNN34 as func2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time


def compare(path):
    vals1 = func1.main(path)
    vals2 = func2.main(path)

    data1 = vals1 + vals2

    df = pd.DataFrame(data1, columns=['vals'])
    df['exp_id'] = ['33 cancer types'] * len(vals1) + ['33 cancer types + normal'] * len(vals2)

    sns.set(style="ticks", palette="pastel")

    ax = sns.boxplot(x='exp_id', y='vals', hue='exp_id', data=df, dodge=False, width=0.5)
    ax.legend().set_visible(False)
    ax.set_xlabel('')
    plt.savefig('box_compare.png', bbox_inches='tight', format='png', dpi=100)



if __name__ == '__main__':
    path = '../tcga_data/preprocessed_0229/preprocessed_11093x7080_p3.pkl'
    start_time = time.time()
    compare(path)
    end_time = time.time()-start_time
    print('='*20)
    print('The whole training and predicting process finish in %.1f h' % (end_time /3600))