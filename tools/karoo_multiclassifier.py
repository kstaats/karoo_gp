# Karoo Multiclass Classifer Test
# by Kai Staats, MSc UCT / AIMS
# version 0.9.2.1

'''
This is a toy script, designed to allow you to play with multiclass classification using the same underlying function
as employed by Karoo GP. Keep in mind that a linear multiclass classifier such as this is suited only for data which
itself has a linear (eg: time series) component, else GP will struggle to force the data to fit.
'''

from numpy import arange

while True:
	try:
		class_type = raw_input('\t Select (i)nfinite or (f)inite wing bins (default i): ')
		if class_type not in ('i','f',''): raise ValueError()
		class_type = class_type or 'i'; break
	except ValueError: print '\033[32mSelect from the options given. Try again ...\n\033[0;0m'

n = range(1,100)
while True:
	try:
		class_labels = raw_input('\t Enter the number of class labels / solutions (default 4): ')
		if class_labels not in str(n) and class_labels not in '': raise ValueError()
		if class_labels == '0': class_labels = 1; break
		class_labels = class_labels or 4; class_labels = int(class_labels); break
	except ValueError: print '\033[32m Enter a number from 3 including 100. Try again ...\n\033[0;0m'

skew = (class_labels / 2) - 1
min_val = 0 - skew - 1 # add a data point to the left

if class_labels & 1: max_val = 0 + skew + 3 # add a data point to the right if odd number of class labels
else: max_val = 0 + skew + 2 # add a data point to the right if even number of class labels

print '\n\t solutions =', range(class_labels)
print '\t results = [', min_val, '...', max_val,']'
print '\t skew =', skew, '\n'

if class_type == 'i':
	for result in arange(min_val, max_val, 0.5):
		for solution in range(class_labels):
		
			if solution == 0 and result <= 0 - skew: # check for the first class
				fitness = 1; print '\t\033[36m\033[1m class', solution, '\033[0;0m\033[36mas\033[1m', result, '\033[0;0m\033[36m<=', 0 - skew, '\033[0;0m'
				
			elif solution == class_labels - 1 and result > solution - 1 - skew: # check for the last class
				fitness = 1; print '\t\033[36m\033[1m class', solution, '\033[0;0m\033[36mas\033[1m', result, '\033[0;0m\033[36m>', solution - 1 - skew, '\033[0;0m'
				
			elif solution - 1 - skew < result <= solution - skew: # check for class bins between first and last
				fitness = 1; print '\t\033[36m\033[1m class', solution, '\033[0;0m\033[36mas', solution - 1 - skew, '<\033[1m', result, '\033[0;0m\033[36m<=', solution - skew, '\033[0;0m'
				
			else: fitness = 0 #; print '\t\033[36m no match for', result, 'in class', solution, '\033[0;0m' # no class match
			
		# print ''


if class_type == 'f':
	for result in arange(min_val, max_val, .5):
		for solution in range(class_labels):
		
			if solution - 1 - skew < result <= solution - skew: # check for discrete, finite class bins
				fitness = 1; print '\t\033[36m\033[1m class', solution, '\033[0;0m\033[36mas', solution - 1 - skew, '<\033[1m', result, '\033[0;0m\033[36m<=', solution - skew, '\033[0;0m'
				
			else: fitness = 0 #; print '\t\033[36m no match for', result, 'in class', solution, '\033[0;0m' # no class match
			
		# print ''


