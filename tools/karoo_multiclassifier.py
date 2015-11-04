# Karoo Multiclass Classifer Test
# Play with quantity of class labels against a range of results
# by Kai Staats, MSc UCT / AIMS

from numpy import arange

while True:
	try:
		class_type = raw_input('\t Select (i)finite or (f)inite class bins (default i): ')
		if class_type not in ('i','f',''): raise ValueError()
		class_type = class_type or 'i'; break
	except ValueError: print '\033[32mSelect from the options given. Try again ...\n\033[0;0m'

n = range(1,100)
while True:
	try:
		class_labels = raw_input('\t Enter the number of class labels (default 4): ')
		if class_labels not in str(n) and class_labels not in '': raise ValueError()
		if class_labels == '0': class_labels = 1; break
		class_labels = class_labels or 4; class_labels = int(class_labels); break
	except ValueError: print '\033[32m Enter a number from 3 including 100. Try again ...\n\033[0;0m'

skew = (class_labels / 2) - 1
min_val = 0 - skew - 1
if class_labels & 1: max_val = 0 + skew + 3
else: max_val = 0 + skew + 2

print '\n\t class_labels =', range(class_labels)
print '\t skew =', skew, '\n'

# a simple binary classifier, for comparison
	# if result <= 0 and label == 0: fitness = 1
	# elif result > 0 and label == 1: fitness = 1
	# else: fitness = 0

if class_type == 'i':
	for result in arange(min_val, max_val, .5):
		for label in range(class_labels):
		
			if label == 0 and result <= 0 - skew: # check for the first class
				fitness = 1; print '\t\033[36m\033[1m class', label, '\033[0;0m\033[36mas\033[1m', result, '\033[0;0m\033[36m<= boundary', 0 - skew, '\033[0;0m'
				
			elif label == class_labels - 1 and result > label - 1 - skew: # check for the last class
				fitness = 1; print '\t\033[36m\033[1m class', label, '\033[0;0m\033[36mas\033[1m', result, '\033[0;0m\033[36m> boundary', label - skew, '\033[0;0m'
				
			elif (label - 1) - skew < result <= label - skew: # check for class bins between first and last
				fitness = 1; print '\t\033[36m\033[1m class', label, '\033[0;0m\033[36mas boundary', (label - 1) - skew, '<\033[1m', result, '\033[0;0m\033[36m<=', 'boundary', label - skew, '\033[0;0m'
				
			else: fitness = 0; print '\t\033[36m no match for', result, 'in class', label, '\033[0;0m' # no class match
			
		print ''


if class_type == 'f':
	for result in arange(min_val, max_val, .5):
		for label in range(class_labels):
		
			if (label - 1) - skew < result <= label - skew: # check for discrete, finite class bins
				fitness = 1; print '\t\033[36m\033[1m class', label, '\033[0;0m\033[36mas boundary', (label - 1) - skew, '<\033[1m', result, '\033[0;0m\033[36m<=', 'boundary', label - skew, '\033[0;0m'
				
			else: fitness = 0; print '\t\033[36m no match for', result, 'in class', label, '\033[0;0m' # no class match
			
		print ''


