import sys
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

#
# 1. read collectl output file
# 2. read log file with the load tests informations (lables)
# 3. merge data into one data set
# 4. compute benchmark based on expert rules
# 5. save dataset into data.csv
# 6. create table with dataset attribues
# 

# read collectl output file
fi = "myfile.out-20160903.tab"

df = pd.read_csv(fi, header=15)

if df.columns[0] != '#Date':
    print "Header row not at line 16 ... exiting"
    sys.exit(0)

# Create a timestamp column

df['ts'] = pd.to_datetime(df['#Date'].map(str) + "-" + df['Time'], format='%Y%m%d-%H:%M:%S')

# Set df['ts'] as the index and delete the columns
df.index = df['ts']
del df['ts']
del df['#Date']
del df['Time']

print "* df shape:", df.shape

# Load the log from the load simulation

load_file = 'load-20160903-21.35.21.log'

dfl = pd.read_csv(load_file)

# Create a timestamp column
dfl['ts'] = pd.to_datetime(dfl['date'].map(str) + "-" + dfl['time'], format='%Y%m%d-%H:%M:%S')

# Set df['ts'] as the index and delete the columns
dfl.index = dfl['ts']
del dfl['ts']
del dfl['date']
del dfl['time']

print "* dfl shape:", dfl.shape

# resample to 1 second before the out join
d = dfl.resample('1S').pad()

print "* d shape:", d.shape

# Join the two tables
data = pd.concat([df, d], axis=1, join='outer')

# remove any na rows
data = data.dropna(how='any')

print "The dataset has {} data points with {} variables each.".format(*data.shape)

print data.columns

data['workload'] = np.where(data.m1==1, 'cpu',
		          np.where(data.m2==1,'vm',
   		               np.where(data.m3==1,'hdd',
   			          np.where(data.m4==1,'io', 'idle'))))

#
# Compute the benchmarks
#

cpu90 = data['[CPU]Totl%'].quantile(q=0.9)
print "CPU Total 90%: ", cpu90

proc90 = data['[CPU]ProcRun'].quantile(q=0.9)
print "CPU ProcRun 90%: ", proc90

data['cpu-bm'] = np.where(data['[CPU]Totl%']>=cpu90, 'cpu',
                       np.where(data['[CPU]ProcRun']>=proc90, 'cpu','none'))

mem10pct = data['[MEM]Tot'].mean() * 0.1
print "10% Mem: ", mem10pct

swapout90 = data['[MEM]SwapOut'].quantile(q=0.95)
print "SwapOut 90%: ", swapout90

data['mem-bm'] = np.where( (data['[MEM]Free'] < mem10pct)  \
             | ( data['[MEM]SwapOut'] > swapout90 ) , 'mem','none')

dsk80 = data['[DSK]KbTot'].quantile(q=0.80)
print "DISK KB Total / sec 80%: ", dsk80

cpuwait90 = data['[CPU]Wait%'].quantile(q=0.9)
print "CPU Wait 90%: ", cpuwait90

data['dsk-bm'] = np.where(data['[DSK]KbTot']>=dsk80, 'dsk',
                       np.where(data['[CPU]Wait%']>=cpuwait90, 'dsk','none'))

data['benchmark'] = np.where(data['mem-bm'] == 'mem', 'vm',
		           np.where(data['dsk-bm'] == 'dsk' ,'hdd',
   		             np.where(data['cpu-bm'] == 'cpu','cpu', 'idle')))


# Drop some columns ...
   			          
del data['m1']
del data['m2']
del data['m3']
del data['m4']
del data['cpu-bm']
del data['mem-bm']
del data['dsk-bm']

data.to_csv("data.csv")

features = data.drop(['workload','benchmark' ], axis = 1)

stats = pd.DataFrame({ # 'Metric' : features.columns,
                    'Count' : features.count(axis=0),
                    'Mean' : features.mean(axis=0),
                    'Stdev' : features.std(axis=0),
                    'Max' : features.max(axis=0),
                    'Min' : features.min(axis=0),
                    '25%' : features.quantile(q=0.25, axis=0)  ,
                    '50%' : features.quantile(q=0.5, axis=0)  ,
                    '75%' : features.quantile(q=0.75, axis=0)  })
print(stats)

stats.to_csv("data_attribs.csv")

fc = f1_score(data['workload'] , data['benchmark'], labels=[ 'cpu', 'vm',  'hdd', 'idle'], average=None)

print "* F1 Scors for 'cpu', 'vm',  'hdd', 'idle':"
print fc

print "* micro: globally by counting the total true positives, false negatives and false positives:"
print f1_score(data['workload'] , data['benchmark'], average='micro')

print "* macro: for each label, and find their unweighted mean. This does not take label imbalance into account."
print f1_score(data['workload'] , data['benchmark'], average='macro')

print "* weighted: for each label, and find their average, weighted by support (the number of true instances for each label)"
print f1_score(data['workload'] , data['benchmark'], average='weighted')
