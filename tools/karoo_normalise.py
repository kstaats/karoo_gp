# Karoo Data Normalisation
# by Kai Staats, MSc UCT
# version 0.9.1.9

import sys
import numpy as np

np.set_printoptions(linewidth = 320) # set the terminal to print 320 characters before line-wrapping in order to view Trees

'''
This script works with a raw dataset to prepare a new, normalised dataset. It does so by comparing all values in each 
given column, finding the maximum and minimum values, and then modifying each value to fall between a high of 1 and 
low of 0. The modified values are written to a new file, the original remaining untouched.

This script can be used *after* karoo_features_sort.py, and assumes no header has yet been applied to the .csv.
'''

def normalise(array):

	'''
	The formula was derived from stn.spotfire.com/spotfire_client_help/norm/norm_normalizing_columns.htm 
	'''
	
	norm = []
	array_norm = []
	array_min = np.min(array)
	array_max = np.max(array)
	
	for col in range(1, len(array) + 1):
		norm = float((array[col - 1] - array_min) / (array_max - array_min))
		norm = round(norm, fp) # force to 4 decimal points
		array_norm = np.append(array_norm, norm)
		
	return array_norm


### USER INTERACTION ###
if len(sys.argv) == 1: print '\n\t\033[31mERROR! You have not assigned an input file. Try again ...\033[0;0m'; sys.exit()
elif len(sys.argv) > 2: print '\n\t\033[31mERROR! You have assigned too many command line arguments. Try again ...\033[0;0m'; sys.exit()
else: filename = sys.argv[1]

n = range(1,9)
while True:
	try:
		fp = raw_input('\n\tEnter number of floating points desired in normalised data (default 4): ')
		if fp not in str(n) and fp not in '': raise ValueError()
		if fp == '0': fp = 1; break
		fp = fp or 4; fp = int(fp); break
	except ValueError: print '\n\t\033[32mEnter a number from 1 including 8. Try again ...\033[0;0m'


### LOAD THE DATA and PREPARE AN EMPTY ARRAY ###
print '\n\t\033[36mLoading dataset:', filename, '\033[0;0m\n'
data = np.loadtxt(filename, delimiter = ',') # load data
data_norm = np.zeros(shape = (data.shape[0], data.shape[1])) # build an empty dataset which matches the shape of the original


### NORMALISE THE DATA ###
for col in range(data.shape[1] - 1):
	print '\tnormalising column:', col
	
	colsum = []
	for row in range(data.shape[0]):
		colsum = np.append(colsum, data[row,col])
		
	data_norm[:,col] = normalise(colsum) # add each normalised column of data
	
data_norm[:,data.shape[1] - 1] = data[:,data.shape[1] - 1] # add the labels again


### SAVE THE NORMALISED DATA ###
file_tmp = filename.split('.')[0]
np.savetxt(file_tmp + '-NORM.csv', data_norm, delimiter = ',')

print '\n\t\033[36mThe normlised dataset has been written to the file:', file_tmp + '-NORM.csv', '\033[0;0m'


