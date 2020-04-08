# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 01:16:30 2020

@author: Leo
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
import math
plt.style.use('classic')
plt.rcParams['figure.facecolor'] = '1'


def prebarplot(x, err, name):
    '''
    x: array of 2, first is the average value of 33 cancer
       second is the average value of 33 + 1 normal
    error: same as x
    '''
    x_label = ['33', '33+1']
    y_b1 = 0.935
    y_b2 = 0.926
    width = 0.3
    y_pos = (width/6, width*2)
    plt.bar(x=y_pos, height=x, yerr=err, width=0.2, align='center', color=['blue', 'orange'])
    plt.axhline(y=y_b2, xmin=0.605, xmax=0.73, color='r', linewidth=4)
    plt.axhline(y=y_b1, xmin=0.237, xmax=0.364, color='r', linewidth=4)
    plt.grid()
    plt.xticks(y_pos, x_label)
    plt.ylabel('Average Accuracy')
    plt.xlabel('Cases')
    plt.gca().set_xlim([-0.4,1.1])
    plt.savefig('result_%s.png' % name, bbox_inches='tight', format='png', dpi=1000)

def loaddata(path):
    x = np.loadtxt(path, delimiter=',')
    return x

def noiseplot(x1,x2,x3,x4, name):
    '''
     x: array of array of noisy input
    '''
    x = np.vstack((x1.T/100, x2.T/100, x3.T/100, x4.T))
    cols = ['10%', '50%','100%', '200%', '500%']
    df = pd.DataFrame(x, columns=cols)
    exp = ['1D CNN'] * 10 + ['2D Vanilla CNN '] * 10 + ['2D Hybrid CNN'] * 10 + ['KNN'] * 10
    df['models'] = exp
    temp = pd.melt(df, id_vars=['models'])
    
    sns.set(style="ticks", palette="bright")
    #sns.set_style("whitegrid")
    ax = sns.barplot(x='variable', y='value', hue='models', data=temp)
    ax.set_xlabel('Noise Level', fontsize=12)
    ax.set_ylabel('Average Accuracy', fontsize=12)
    ax.yaxis.grid(True) 
    ax.xaxis.grid(False)
    plt.title('Average Performance of Robustness', fontsize=18)
    plt.legend(title='Model', loc='center right',bbox_to_anchor=(1.3, 0.85))
    plt.savefig('noise_result_%s.png' % name, bbox_inches='tight', format='png', dpi=1000)

def featureplot(x):
    '''
    input: x - an array that contain average 
    bar plot for feature selection result
    x-axis: different case
    y-average performance
    '''
    pass

def loadPara(path):
    '''
    load the parameter result from the txt file
    return: parameter combinations, statistical values
            parameter combinations: first -- dense layer size
                                    second -- kernel size
                                    third -- filter number
    '''
    lines = []
    values = []
    paras  = []
    with open (txtfile, 'r') as txt:
        for line in txt:
            if '#' not in line and 'params' not in line: 
                line = re.findall(r'[\d\.\d]+', line)
                lines.append(line)
                paras.append((int(line[0]), int(line[2]), int(line[3])))
                values.append([round(abs(float(x)),3) for x in line[4:]])
                values.append([round(abs(float(x)),3) for x in line[4:]])
    
    return paras, values

def mkheatmaps(paras, values):
    '''
    '''
    dense_layer_size = [64, 128, 256, 512]
    kernel_size = [5, 25, 50, 100, 200]
    filter_number = [8, 16, 32, 64]
    
    fig, axes = plt.subplots(5, 4, figsize=(25,12))
    for i, val1 in enumerate(kernel_size):
        data = np.zeros((4,4,4))
        for j, val2 in enumerate(filter_number):
            for m, val3 in enumerate(dense_layer_size):
                for n, val4 in enumerate(paras):
                    d, k, f = val4
                    if k == val1 and d == val3 and f == val2:
                        data[0, j, m] = values[n][0]
                        data[1, j, m] = values[n][1]
                        data[2, j, m] = values[n][2]
                        data[3, j, m] = values[n][3]
        # plot the subplot for train mean, train std
        # test mean, test std
        for z in range(data.shape[0]):
            sns.heatmap(data[z], annot=True, xticklabels=dense_layer_size, yticklabels=filter_number, 
                        ax=axes[i,z]).set(title = 'kernel size=%s' % str(val1), 
                                          xlabel = 'dense layer size', ylabel = 'filter number' )
            axes[i,z].set_ylim(data[z].shape[0], 0)

    fig.tight_layout(pad=1)
    plt.savefig('parameters.png', format='png', dpi=200) 
                        
    
    
    
#%%
paths = 'E:/Dropbox/Course/11785/Project/results/'
x33_1 = loaddata(paths + '1dcnn33.csv')
x33_2 = loaddata(paths + '2dcnn33v.csv')
x33_3 = loaddata(paths + '2dcnn33h.csv')
x33_4 = KNN_33csv[1:,]

x34_1 = loaddata(paths + '1dcnn34.csv')
x34_2 = loaddata(paths + '2dcnn34v.csv')
x34_3 = loaddata(paths + '2dcnn34h.csv')
x34_4 = KNN_34csv[1:,:]

noiseplot(x33_1, x33_2, x33_3, x33_4, '33')
noiseplot(x34_1, x34_2, x34_3, x34_4, '34')
    
#%%
prebarplot([0.9547, 0.9464], [0.61/100, 0.65/100],'1DCNN')
prebarplot([0.9501, 0.9491], [0.43/100, 0.37/100],'2DCNNV')
prebarplot([0.9567, 0.9447], [0.44/100, 1.02/100],'2DCNNH')

#%%
x1 = np.random.rand(10, 5)
x2 = np.random.rand(10, 5)
x3 = np.random.rand(10, 5)
x = np.vstack((x1, x2, x3))
cols = ['10%', '50%','100%', '200%', '500%']
df = pd.DataFrame(x, columns=cols)
exp = ['1D CNN'] * 10 + ['2D Vanilla CNN '] * 10 + ['2D Hybrid CNN'] * 10 
df['models'] = exp
temp = pd.melt(df, id_vars=['models'])

sns.set(style="ticks", palette="bright")

#sns.set_style("whitegrid")
ax = sns.boxplot(x='variable', y='value', hue='models', data=temp)
ax.set_xlabel('Noise Level', fontsize=12)
ax.set_ylabel('Average Accuracy', fontsize=12)
ax.yaxis.grid(True) 
ax.xaxis.grid(False)
plt.title('Average Performance of Robustness', fontsize=18)
plt.legend(title='Model', loc='center right',bbox_to_anchor=(1.3, 0.85))
#%%
x = [0.94, 0.95]
err = [0.44/100, 0.53/100]
x_label = ['33', '33+1']
width = 0.3
y_b1 = 0.935
y_b2 = 0.926
y_pos = (width/6, width*2)
plt.bar(x=y_pos, height=x, yerr=err, width=0.2, align='center', color=['blue', 'orange'])
plt.axhline(y=y_b2, xmin=0.605, xmax=0.73, color='r', linewidth=4)
plt.axhline(y=y_b1, xmin=0.237, xmax=0.364, color='r', linewidth=4)
plt.grid()
#plt.xticks(y_pos, x_label)
plt.ylabel('Average Accuracy')
plt.xlabel('model')
plt.gca().set_xlim([-0.4,1.1])

#%%
import re 
numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
txtfile = "E:/Dropbox/Course/11785/Project/ZSdata/0402_model_selection/1dcnn_ht_3394.txt"

lines = []
values = []
paras  = []
with open (txtfile, 'r') as txt:
    for line in txt:
        if '#' not in line and 'params' not in line: 
            line = re.findall(r'[-+]?[\d\.\d]+', line)
            lines.append(line)
            paras.append((int(line[0]), int(line[2]), int(line[3])))
            #values.append([round(abs(float(x)),3) for x in line[4:]])
            #values.append([round(float(x),3) if float(x) > 0 else round(math.e ** float(x),3) for x in line[4:]])
            values.append([(float(x)) for x in line[4:]])
'''
re.findall(r'[\d\.\d]+', a[-1])
'''

#%%
#plt.subplots_adjust(hspace=1)
dense_layer_size = [64, 128, 256, 512]
kernel_size = [5, 25, 50, 100, 200]
filter_number = [8, 16, 32, 64]

fig, axes = plt.subplots(5, 4, figsize=(30,20))
for i, val1 in enumerate(kernel_size):
    data = np.zeros((4,4,4))
    for j, val2 in enumerate(filter_number):
        for m, val3 in enumerate(dense_layer_size):
            for n, val4 in enumerate(paras):
                d, k, f = val4
                if k == val1 and d == val3 and f == val2:
                    data[0, j, m] = values[n][0]
                    data[1, j, m] = values[n][1]
                    data[2, j, m] = values[n][2]
                    data[3, j, m] = values[n][3]
    # plot the subplot for train mean, train std
    # test mean, test std
    for z in range(data.shape[0]):
        sns.heatmap(data[z], annot=True, xticklabels=dense_layer_size, yticklabels=filter_number, ax=axes[i,z]).set(title = 'kernel size=%s' % str(val1), xlabel = 'dense layer size', ylabel = 'filter number' )
        axes[i,z].set_ylim(data[z].shape[0], 0)
        #plt.title('kernel size=%s' % str(val1))
#plt.xlabel('filter number')
#plt.ylabel('dense layer size')
fig.tight_layout(pad=1)
plt.savefig('parameters.png', format='png', dpi=200) 
#a = sns.heatmap(data[0], annot=True, xticklabels=filter_number, yticklabels=dense_layer_size)  
#a.set_ylim(data[0].shape[0], 0)

#%%
dic = {
"ACC" :	4,	
"BLCA":	3,	
"BRCA":	6,	
"CESC":	3,	
"CHOL":	3,	
"COAD":	6,	
"DLBC":	10,	
"ESCA":	3,	
"GBM":	8,	
"HNSC":	8,	
"KICH":	14,	
"KIRC":	4,	
"KIRP":	4,	
"LAML":	18,	
"LGG":	5,	
"LIHC":	10,	
"LUAD":	2,	
"LUSC":	3,	
"MESO":	12,	
"OV" : 3,
"PAAD":	4,	
"PCPG":	4,	
"PRAD":	3,	
"READ":	2,	
"SARC":	4,	
"SKCM":	2,	
"STAD":5,	
"TGCT":	2,	
"THCA":2,	
"THYM":	1,	
"UCEC":	2,	
"UCS":	2,	
"UVM":	2	
}

plt.figure(figsize=(20,5))
vals = list(dic.values())
keys = list(dic.keys())
y_pos = list(range(len(vals)))
plt.bar(x=y_pos, height=vals)
plt.grid(axis='y')
plt.xticks(y_pos, keys, rotation=90)
plt.ylabel('Number of maker genes')
plt.gca().set_xlim([-2,34])
plt.savefig('maker_gene_num.png', bbox_inches='tight', format='png', dpi=200)