# Karoo GP Main
# Use Genetic Programming for Classification and Symbolic Regression
# by Kai Staats, MSc UCT / AIMS
# Much thanks to Emmanuel Dufourq and Arun Kumar for their support, guidance, and free psychotherapy sessions
# version 0.9.1.1

'''
A NOTE TO THE NEWBIE, EXPERT, AND BRAVE
Even if you are highly experienced in Genetic Programming, it is recommended that you review the 'Karoo Quick Start' before running 
this application. While your computer will not burst into flames nor will the sun collapse into a black hole if you do not, you will 
likely find more enjoyment of this particular flavour of GP with a little understanding of its intent and design.
'''

import sys # sys.path.append('modules/') # add the directory 'modules' to the current path 
import karoo_gp_base_class; gp = karoo_gp_base_class.Base_GP()

#++++++++++++++++++++++++++++++++++++++++++
#   User Defined Configuration            |
#++++++++++++++++++++++++++++++++++++++++++

'''
Karoo GP queries the user for key parameters, some of which may be adjusted during run-time 
at user invoked pauses. See the User Guide for meaning and value of each of the following parameters.

Future versions will enable all of these parameters to be configured via an external configuration file and/or 
command-line arguments passed at launch.
'''

gp.karoo_banner('main')

print ''

while True:
	try:
		gp.kernel = raw_input('\t Select (a)bs diff, (c)lassify, (m)atch, or (p)lay (default m): ')
		if gp.kernel not in ('a','b','c','m','p',''): raise ValueError()
		gp.kernel = gp.kernel or 'm'; break
	except ValueError: print '\033[32mSelect from the options given. Try again ...\n\033[0;0m'
	
if gp.kernel == 'c':
	
	n = range(1,101)
	while True:
		try:
			gp.class_labels = raw_input('\t Enter the number of class labels (default 3): ')
			if gp.class_labels not in str(n) and gp.class_labels not in '': raise ValueError()
			if gp.class_labels == '0': gp.class_labels = 3; break
			gp.class_labels = gp.class_labels or 3; gp.class_labels = int(gp.class_labels); break
		except ValueError: print '\033[32mSelect from the options given. Try again ...\n\033[0;0m'
		
	# while True:
		# try:
			# gp.class_type = raw_input('\t Select (f)inite or (i)finite classification (default i): ')
			# if gp.class_type not in ('f','i',''): raise ValueError()
			# gp.class_type = gp.class_type or 'i'; break
		# except ValueError: print '\033[32mSelect from the options given. Try again ...\n\033[0;0m'
		
while True:
	try:
		tree_type = raw_input('\t Select (f)ull, (g)row, or (r)amped 50/50 method (default r): ')
		if tree_type not in ('f','g','r',''): raise ValueError()
		tree_type = tree_type or 'r'; break
	except ValueError: print '\033[32mSelect from the options given. Try again ...\n\033[0;0m'
	
n = range(1,11)
while True:
	try:
		tree_depth_max = raw_input('\t Enter maximum depth of each Tree (default 3): ')
		if tree_depth_max not in str(n) and tree_depth_max not in '': raise ValueError()
		if tree_depth_max == '0': tree_depth_max = 1; break
		tree_depth_max = tree_depth_max or 3; tree_depth_max = int(tree_depth_max); break
	except ValueError: print '\033[32mEnter a number from 3 including 10. Try again ...\n\033[0;0m'
	
if gp.kernel == 'p': # if the Play kernel is selected
	gp.tree_pop_max = 1
	gp.display = 'm'

else: # if any other kernel is selected

	n = range(3,101)
	while True:
		try:
			gp.tree_depth_min = raw_input('\t Enter minimum number of nodes for any given Tree (default 3): ')
			if gp.tree_depth_min not in (str(n)) and gp.tree_depth_min not in (''): raise ValueError()
			if gp.tree_depth_min == '0': gp.tree_depth_min = 3; break
			gp.tree_depth_min = gp.tree_depth_min or 3; gp.tree_depth_min = int(gp.tree_depth_min); break
		except ValueError: print '\033[32mEnter a number from 3 to 2^(depth + 1) - 1 including 100. Try again ...\n\033[0;0m'
		
	n = range(10,1001)
	while True:
		try:
			gp.tree_pop_max = raw_input('\t Enter number of Trees in each Generation (default 100): ')
			if gp.tree_pop_max not in (str(n)) and gp.tree_pop_max not in (''): raise ValueError()
			if gp.tree_pop_max == '0': gp.tree_pop_max = 100; break
			gp.tree_pop_max = gp.tree_pop_max or 100; gp.tree_pop_max = int(gp.tree_pop_max); break
		except ValueError: print '\033[32mEnter a number from 10 including 1000. Try again ...\n\033[0;0m'
		
	n = range(1,101)
	while True:
		try:
			gp.generation_max = raw_input('\t Enter max number of Generations (default 10): ')
			if gp.generation_max not in (str(n)) and gp.generation_max not in (''): raise ValueError()
			if gp.generation_max == '0': gp.generation_max = 10; break
			gp.generation_max = gp.generation_max or 10; gp.generation_max = int(gp.generation_max); break
		except ValueError: print '\033[32mEnter a number from 1 including 100. Try again ...\n\033[0;0m'
		
	while True:
		try:
			gp.display = raw_input('\t Display (i)nteractive, (m)iminal, (g)eneration, or (s)ilent (default m): ')
			if gp.display not in ('i','m','g','s','db','t',''): raise ValueError()
			gp.display = gp.display or 'm'; break
		except ValueError: print '\033[32mSelect from the options given. Try again ...\n\033[0;0m'
		

# define the ratio between types of mutation, where all sum to 1.0; can be adjusted in 'i'nteractive mode
gp.evolve_repro = int(0.1 * gp.tree_pop_max) # percentage of subsequent population to be generated through Reproduction
gp.evolve_point = int(0.1 * gp.tree_pop_max) # percentage of subsequent population to be generated through Point Mutation
gp.evolve_branch = int(0.2 * gp.tree_pop_max) # percentage of subsequent population to be generated through Branch Mutation
gp.evolve_cross = int(0.6 * gp.tree_pop_max) # percentage of subsequent population to be generated through Crossover Reproduction

gp.tourn_size = 10 # qty of individuals entered into each tournament (standard 10); can be adjusted in 'i'nteractive mode
gp.cores = 1 # replace '1' with 'int(gp.core_count)' to auto-set to max; can be adjusted in 'i'nteractive mode
gp.precision = 4 # the number of floating points for the round function in 'fx_fitness_eval'; hard coded

# if len(sys.argv) == 2: # look for an argument when Karoo GP is launched
# 	gp.data_load = int(sys.argv[1]) # assign file for the data load method in karoo_base_class


#++++++++++++++++++++++++++++++++++++++++++
#   Construct First Generation of Trees   |
#++++++++++++++++++++++++++++++++++++++++++

'''
Karoo GP constructs the first generation of Trees. All subsequent generations evolve from priors, with no new Trees
constructed from scratch. All parameters which define the Trees were set by the user in the previous section.

If the user has selected 'Play' mode, this is the only generation to be constructed, and then GP Karoo terminates.
'''

gp.fx_karoo_data_load()
gp.generation_id = 1 # set initial generation ID

gp.population_a = ['Karoo GP by Kai Staats, Generation ' + str(gp.generation_id)] # an empty list which will store all Tree arrays, one generation at a time

gp.fx_karoo_construct(tree_type, tree_depth_max) # construct the first population of Trees	

if gp.kernel != 'p': print '\n We have constructed a population of', gp.tree_pop_max,'Trees for Generation 1\n'

else: # EOL for Play mode
	gp.fx_eval_tree_print(gp.tree) # print the current Tree
	gp.fx_tree_archive(gp.population_a, 'a') # save this one Tree to disk
	sys.exit()
	

#++++++++++++++++++++++++++++++++++++++++++
#   Evaluate First Generation of Trees    |
#++++++++++++++++++++++++++++++++++++++++++

'''
Karoo GP evaluates the first generation of Trees. This process flattens each GP Tree into a standard
equation by means of a recursive algorithm and subsequent processing by the SymPy library which 
simultaneously evaluates the Tree for its results, returns null for divide by zero, reorganises 
and then rewrites the expression in its simplest form.

If the user has defined only 1 generation, then this is the end of the run. Else, Karoo GP 
continues into multi-generational evolution.
'''

if gp.display != 's':
	print ' Evaluate the first generation of Trees ...'
	if gp.display == 'i': gp.fx_karoo_pause(0)

gp.fx_fitness_gym(gp.population_a) # 1) extract polynomial from each Tree; 2) evaluate fitness, store; 3) display
gp.fx_tree_archive(gp.population_a, 'a') # save the first generation of Trees to disk

# no need to continue if only 1 generation or fewer than 10 Trees were designated by the user
if gp.tree_pop_max < 10 or gp.generation_max == 1:
	gp.fx_karoo_eol(); sys.exit()
	

#++++++++++++++++++++++++++++++++++++++++++
#   Evolve Multiple Generations           |
#++++++++++++++++++++++++++++++++++++++++++

'''
Karoo GP moves into multi-generational evolution.

In the following four evolutionary methods, the global list of arrays 'gp.population_a' is repeatedly recycled as 
the prior generation from which the local list of arrays 'gp.population_b' is created, one array at a time. The ratio of
invocation of the four evolutionary processes for each generation is set by the parameters in the 'User Defined 
Configuration' (top).
'''

for gp.generation_id in range(2, gp.generation_max + 1): # loop through 'generation_max'

	print '\n Evolve a population of Trees for Generation', gp.generation_id, '...'
	gp.population_b = ['GP Tree by Kai Staats, Evolving Generation'] # initialise population_b to host the next generation
	
	gp.fx_fitness_gene_pool() # generate the viable gene pool (compares against gp.tree_depth_min)
	gp.fx_karoo_reproduce() # method 1 - Reproduction
	gp.fx_karoo_point_mutate() # method 2 - Point Mutation
	gp.fx_karoo_branch_mutate() # method 3 - Branch Mutation
	gp.fx_karoo_crossover_reproduce() # method 4 - Crossover Reproduction
	gp.fx_eval_generation() # evaluate all Trees in a single generation
	
	gp.population_a = gp.fx_evo_pop_copy(gp.population_b, ['GP Tree by Kai Staats, Generation ' + str(gp.generation_id)])
	

#++++++++++++++++++++++++++++++++++++++++++
#   "End of line, man!" --CLU             |
#++++++++++++++++++++++++++++++++++++++++++

gp.fx_tree_archive(gp.population_b, 'f') # save the final generation of Trees to disk
gp.fx_karoo_eol()

	
