# Karoo GP Main (desktop)
# Use Genetic Programming for Classification and Symbolic Regression
# by Kai Staats, MSc; see LICENSE.md
# Thanks to Emmanuel Dufourq and Arun Kumar for support during 2014-15 devel; TensorFlow support provided by Iurii Milovanov
# version 1.0.5

'''
A word to the newbie, expert, and brave--
Even if you are highly experienced in Genetic Programming, it is recommended that you review the 'Karoo User Guide' 
before running this application. While your computer will not burst into flames nor will the sun collapse into a black 
hole if you do not, you will likely find more enjoyment of this particular flavour of GP with a little understanding 
of its intent and design.

KAROO GP DESKTOP
This is the Karoo GP desktop application. It presents a simple yet functional user interface for configuring each
Karoo GP run. While this can be launched on a remote server, you may find that once you get the hang of using Karoo, 
and are in more of a production mode than one of experimentation, using karoo_gp_server.py is more to your liking as 
it provides both a scripted and/or command-line launch vehicle.

To launch Karoo GP desktop:

	$ python karoo_gp_main.py
	
	  (or from iPython)
	
	$ run karoo_gp_main.py


If you include the path to an external dataset, it will auto-load at launch:

	$ python karoo_gp_main.py /[path]/[to_your]/[filename].csv
'''

import sys # sys.path.append('modules/') to add the directory 'modules' to the current path 
import karoo_gp_base_class; gp = karoo_gp_base_class.Base_GP()
import time

#++++++++++++++++++++++++++++++++++++++++++
#   User Defined Configuration            |
#++++++++++++++++++++++++++++++++++++++++++

'''
Karoo GP queries the user for key parameters, some of which may be adjusted during run-time 
at user invoked pauses. See the User Guide for meaning and value of each of the following parameters.

Future versions will enable all of these parameters to be configured via an external configuration file and/or 
command-line arguments passed at launch.
'''

gp.karoo_banner()

print('')

menu = ['c','r','m','p','']
while True:
	try:
		gp.kernel = input('\t Select (c)lassification, (r)egression, (m)atching, or (p)lay (default m): ')
		if gp.kernel not in menu: raise ValueError()
		gp.kernel = gp.kernel or 'm'; break
	except ValueError: print('\t\033[32m Select from the options given. Try again ...\n\033[0;0m')
	except KeyboardInterrupt: sys.exit()
	
if gp.kernel == 'p':

	menu = ['f','g','']
	while True:
		try:
			tree_type = input('\t Select (f)ull or (g)row method (default f): ')
			if tree_type not in menu: raise ValueError()
			tree_type = tree_type or 'f'; break
		except ValueError: print('\t\033[32m Select from the options given. Try again ...\n\033[0;0m')
		except KeyboardInterrupt: sys.exit()

else:

	menu = ['f','g','r','']
	while True:
		try:
			tree_type = input('\t Select (f)ull, (g)row, or (r)amped 50/50 method (default r): ')
			if tree_type not in menu: raise ValueError()
			tree_type = tree_type or 'r'; break
		except ValueError: print('\t\033[32m Select from the options given. Try again ...\n\033[0;0m')
		except KeyboardInterrupt: sys.exit()
	
menu = list(range(1,11))
while True:
	try:
		tree_depth_base = input('\t Enter depth of the \033[3minitial\033[0;0m population of Trees (default 3): ')
		if tree_depth_base not in str(menu) or tree_depth_base == '0': raise ValueError()
		tree_depth_base = tree_depth_base or 3; tree_depth_base = int(tree_depth_base); break
	except ValueError: print('\t\033[32m Enter a number from 1 including 10. Try again ...\n\033[0;0m')
	except KeyboardInterrupt: sys.exit()
	


if gp.kernel == 'p': # if the Play kernel is selected
	gp.tree_depth_max = tree_depth_base
	gp.tree_pop_max = 1
	gp.display = 'm'

else: # if any other kernel is selected

	if tree_type == 'f': gp.tree_depth_max = tree_depth_base
	else: # if type is Full, the maximum Tree depth for the full run is equal to the initial population
	
		menu = list(range(tree_depth_base,11))
		while True:
			try:
				gp.tree_depth_max = input('\t Enter maximum Tree depth (default matches \033[3minitial\033[0;0m): ')
				if gp.tree_depth_max not in str(menu) or gp.tree_depth_max == '0': raise ValueError()
				gp.tree_depth_max = gp.tree_depth_max or tree_depth_base; gp.tree_depth_max = int(gp.tree_depth_max); break
				# gp.tree_depth_max = int(gp.tree_depth_max) - tree_depth_base; break
			except ValueError: print('\t\033[32m Enter a number >= the maximum Tree depth. Try again ...\n\033[0;0m')
			except KeyboardInterrupt: sys.exit()
			
	menu = list(range(3,101))
	while True:
		try:
			gp.tree_depth_min = input('\t Enter minimum number of nodes for any given Tree (default 3): ')
			if gp.tree_depth_min not in str(menu) or gp.tree_depth_min == '0': raise ValueError()
			gp.tree_depth_min = gp.tree_depth_min or 3; gp.tree_depth_min = int(gp.tree_depth_min); break
		except ValueError: print('\t\033[32m Enter a number from 3 to 2^(depth + 1) - 1 including 100. Try again ...\n\033[0;0m')
		except KeyboardInterrupt: sys.exit()
		
	menu = list(range(10,1001))
	while True:
		try:
			gp.tree_pop_max = input('\t Enter number of Trees in each population (default 100): ')
			if gp.tree_pop_max not in str(menu) or gp.tree_pop_max == '0': raise ValueError()
			gp.tree_pop_max = gp.tree_pop_max or 100; gp.tree_pop_max = int(gp.tree_pop_max); break
		except ValueError: print('\t\033[32m Enter a number from 10 including 1000. Try again ...\n\033[0;0m')
		except KeyboardInterrupt: sys.exit()
		
	menu = list(range(1,101))
	while True:
		try:
			gp.generation_max = input('\t Enter max number of generations (default 10): ')
			if gp.generation_max not in str(menu) or gp.generation_max == '0': raise ValueError()
			gp.generation_max = gp.generation_max or 10; gp.generation_max = int(gp.generation_max); break
		except ValueError: print('\t\033[32m Enter a number from 1 including 100. Try again ...\n\033[0;0m')
		except KeyboardInterrupt: sys.exit()
		
	menu = ['i','g','m','s','db','']
	while True:
		try:
			gp.display = input('\t Display (i)nteractive, (g)eneration, (m)iminal, (s)ilent, or (d)e(b)ug (default m): ')
			if gp.display not in menu: raise ValueError()
			gp.display = gp.display or 'm'; break
		except ValueError: print('\t\033[32m Select from the options given. Try again ...\n\033[0;0m')
		except KeyboardInterrupt: sys.exit()
		

# define the ratio between types of mutation, where all sum to 1.0; can be adjusted in 'i'nteractive mode
gp.evolve_repro = int(0.1 * gp.tree_pop_max) # quantity of a population generated through Reproduction
gp.evolve_point = int(0.0 * gp.tree_pop_max) # quantity of a population generated through Point Mutation
gp.evolve_branch = int(0.2 * gp.tree_pop_max) # quantity of a population generated through Branch Mutation
gp.evolve_cross = int(0.7 * gp.tree_pop_max) # quantity of a population generated through Crossover

gp.tourn_size = 7 # qty of individuals entered into each tournament (standard 10); can be adjusted in 'i'nteractive mode
gp.precision = 6 # the number of floating points for the round function in 'fx_fitness_eval'; hard coded


#++++++++++++++++++++++++++++++++++++++++++
#   Construct First Generation of Trees   |
#++++++++++++++++++++++++++++++++++++++++++

'''
Karoo GP constructs the first generation of Trees. All subsequent generations evolve from priors, with no new Trees
constructed from scratch. All parameters which define the Trees were set by the user in the previous section.

If the user has selected 'Play' mode, this is the only generation to be constructed, and then GP Karoo terminates.
'''

start = time.time() # start the clock for the timer
	
filename = '' # temp place holder
gp.fx_karoo_data_load(tree_type, tree_depth_base, filename)
gp.generation_id = 1 # set initial generation ID

gp.population_a = ['Karoo GP by Kai Staats, Generation ' + str(gp.generation_id)] # an empty list which will store all Tree arrays, one generation at a time

gp.fx_karoo_construct(tree_type, tree_depth_base) # construct the first population of Trees	

if gp.kernel != 'p': print('\n We have constructed a population of', gp.tree_pop_max,'Trees for Generation 1\n')

else: # EOL for Play mode
	gp.fx_display_tree(gp.tree) # print the current Tree
	gp.fx_archive_tree_write(gp.population_a, 'a') # save this one Tree to disk
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
	print('Evaluate the first generation of Trees')
	if gp.display == 'i': gp.fx_karoo_pause(0)

gp.fx_fitness_gym(gp.population_a) # generate expression, evaluate fitness, compare fitness
gp.fx_archive_tree_write(gp.population_a, 'a') # save the first generation of Trees to disk

# no need to continue if only 1 generation or fewer than 10 Trees were designated by the user
if gp.tree_pop_max < 10 or gp.generation_max == 1:
  gp.fx_archive_params_write('Desktop') # save run-time parameters to disk
  gp.fx_karoo_eol()
  sys.exit()
	

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

	print('\n Evolve a population of Trees for Generation', gp.generation_id, '...')
	gp.population_b = ['Karoo GP by Kai Staats, Evolving Generation'] # initialise population_b to host the next generation
	
	gp.fx_fitness_gene_pool() # generate the viable gene pool (compares against gp.tree_depth_min)
	gp.fx_karoo_reproduce() # method 1 - Reproduction
	gp.fx_karoo_point_mutate() # method 2 - Point Mutation
	gp.fx_karoo_branch_mutate() # method 3 - Branch Mutation
	gp.fx_karoo_crossover() # method 4 - Crossover Reproduction
	gp.fx_eval_generation() # evaluate all Trees in a single generation
	
	gp.population_a = gp.fx_evolve_pop_copy(gp.population_b, ['Karoo GP by Kai Staats, Generation ' + str(gp.generation_id)])
	

#++++++++++++++++++++++++++++++++++++++++++
#   "End of line, man!" --CLU             |
#++++++++++++++++++++++++++++++++++++++++++

print('\n \033[36m Karoo GP has an ellapsed time of \033[0;0m\033[31m%f\033[0;0m' % (time.time() - start), '\033[0;0m')

gp.fx_archive_tree_write(gp.population_b, 'f') # save the final generation of Trees to disk
gp.fx_karoo_eol()


