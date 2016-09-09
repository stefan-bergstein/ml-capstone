
## %matplotlib inline

from sklearn import preprocessing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the dataset
data = pd.read_csv('data.csv',index_col=0, parse_dates=True)

print "== Load data " 
print "The dataset has {} data points with {} variables.".format(*data.shape)

# ax = data.plot(kind='line', subplots=True, figsize=(60,120), legend=True)


print "== Remove constant or unneeded data  " 
data.drop(['[CPU]Nice%','[CPU]Irq%','[CPU]Steal%','[MEM]Tot', '[MEM]Shared', '[MEM]Locked', '[MEM]SwapTot', \
           '[MEM]Clean', '[MEM]Laundry', '[MEM]HugeTotal', '[MEM]HugeFree', '[MEM]HugeRsvd', '[NET]RxCmpTot', \
           '[NET]RxMltTot', '[NET]TxCmpTot', '[NET]RxErrsTot', '[NET]TxErrsTot', \
           '[CPU]L-Avg1', '[CPU]L-Avg5', '[CPU]L-Avg15', \
           'benchmark' ], axis = 1, inplace=True )

print(data.shape)

print "== Scale data " 

scaler = preprocessing.StandardScaler()
data.ix[:,0:-1] = scaler.fit_transform( data.ix[:,0:-1] )

print "== Generate histogram " 

fig, axes = plt.subplots(15, 3, figsize=(20, 40))

cpu = data[data.workload == 'cpu']
vm = data[data.workload == 'vm']
hdd = data[data.workload == 'hdd']

ax = axes.ravel()

for i in range(data.shape[1]-1):
    _, bins = np.histogram(data.ix[:, i], bins=50)
    ax[i].hist(cpu.ix[:, i], bins=bins, color='b', alpha=1.0)
    ax[i].hist(vm.ix[:, i], bins=bins, color='r', alpha=1.0)
    ax[i].hist(hdd.ix[:, i], bins=bins, color='g', alpha=1.0)
    
    ax[i].set_title(data.columns[i])
    ax[i].set_yticks(())

fig.tight_layout()
