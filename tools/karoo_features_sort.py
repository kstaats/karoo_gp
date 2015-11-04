# Karoo Feature Set Prep
# Prepare a balanced feature set
# by Kai Staats, MSc UCT / AIMS and Arun Kumar, PhD

import sys
import numpy as np

filename = sys.argv[1] # 'data/pixel_classifier/kat7-20150924-SUBSET.csv'
samples = 5000

# do NOT use readline as that is very, very slow
# ideally use 'pandas', a numpy replacement which loads data 5x faster (future version)
# for now, we'll just load the damn data as we have ample RAM

data = np.loadtxt(filename, skiprows = 1, delimiter = ',', dtype = float) #; data_x = data_x[:,0:-1]
# header = need to read the first line to retain the variables

print '\ndata loaded'
print ' data.shape:', data.shape

# find the indices where the final column = 0 or 1 and record the row num accordingly
data_0_list = np.where(data[:,-1] == 0)
data_1_list = np.where(data[:,-1] == 1)

data_0 = np.random.choice(data_0_list[0], samples, replace = False)
data_1 = np.random.choice(data_1_list[0], samples, replace = False)
print '\nrandom, unique rows generated'
print ' data_0.shape:', data_0.shape
print ' data_1.shape:', data_1.shape

print '\nready to merge data_0 and data_1 with real values'

data_0_new = data[data_0]
data_1_new = data[data_1]
data_new = np.vstack((data_0_new, data_1_new))
print ' data_new.shape', data_new.shape

# np.savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ')

# need to append the header
np.savetxt('data_new.csv', data_new, delimiter = ',')
print '\n data saved as data_new.csv'
