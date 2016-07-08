# Karoo Dataset Builder
# by Kai Staats, MSc UCT / AIMS and Arun Kumar, PhD
# version 0.9.1.2

import sys
import numpy as np

np.set_printoptions(linewidth = 320) # set the terminal to print 320 characters before line-wrapping in order to view Trees

'''
In machine learning, it is often the case that your engaged dataset is derived from a larger parent. In constructing 
the subset, if we grab a series of datapoints (rows in a .csv) from the larger dataset in sequential order, only from 
the top, middle, or bottom, we will likely bias the new dataset and incorrectly train the machine learning algorithm. 
Therefore, it is imperative that we engage a random function, guided only by the number of data points for each class.

This script can be used *before* karoo_normalise.py, and assumes no header has yet been applied to the .csv.
'''

### USER INTERACTION ###
if len(sys.argv) == 1: print '\n\t\033[31mERROR! You have not assigned an input file. Try again ...\033[0;0m'; sys.exit()
elif len(sys.argv) > 2: print '\n\t\033[31mERROR! You have assigned too many command line arguments. Try again ...\033[0;0m'; sys.exit()
else: filename = sys.argv[1]

n = range(1,101)
while True:
	try:
		labels = raw_input('\n\tEnter number of unique class labels, or 0 for a regression dataset (default 2): ')
		if labels not in str(n) and labels not in '': raise ValueError()
		# if labels == '0': labels = 1; break
		labels = labels or 2; labels = int(labels); break
	except ValueError: print '\n\t\033[32mEnter a number from 0 including 100. Try again ...\033[0;0m'

n = range(10,10001)
while True:
	try:
		samples = raw_input('\n\tEnter number of desired datapoints per class (default 100): ')
		if samples not in str(n) and samples not in '': raise ValueError()
		if samples == '0': samples = 10; break
		samples = samples or 100; samples = int(samples); break
	except ValueError: print '\n\t\033[32mEnter a number from 10 including 10000. Try again ...\033[0;0m'


### LOAD THE ORIGINAL DATASET ###
print '\n\t\033[36m\n\tLoading dataset:', filename, '\033[0;0m\n'
data = np.loadtxt(filename, delimiter = ',') # load data
data_sort = np.empty(shape = [0, data.shape[1]]) # build an empty array of the proper dimensions


### SORT DATA by LABEL ###
for label in range(labels):
	data_list = np.where(data[:,-1] == label) # build a list of all rows which end in the current label

	data_select = np.random.choice(data_list[0], samples, replace = False) # select user defined 'samples' from list
	print data_select
	
	data_sort = np.append(data_sort, data[data_select], axis = 0)


### SAVE THE SORTED DATASET ###
file_tmp = filename.split('.')[0]
np.savetxt(file_tmp + '-SORT.csv', data_sort, delimiter = ',')

print '\n\t\033[36mThe sorted dataset has been written to the file:', file_tmp + '-SORT.csv', '\033[0;0m'


