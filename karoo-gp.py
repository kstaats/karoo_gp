#!/bin/python3
# Karoo GP - Genetic Programming for Classification and Symbolic Regression

'''
A word to the newbie, expert, and brave--
Even if you are highly experienced in Genetic Programming, it is recommended that you review the 'Karoo User Guide' 
before running this application. While your computer will not burst into flames nor will the sun collapse into a black 
hole if you do not, you will likely find more enjoyment of this particular flavour of GP with a little understanding 
of its intent and design.

Without any command line arguments, Karoo GP relies upon user settings and the datasets located in karoo_gp/files/.

	$ python karoo_gp_main.py
	

If you include the path to an external dataset, it will auto-load at launch:

	$ python karoo_gp_main.py /[path]/[to_your]/[filename].csv
	

If you include one or more additional arguments, they will override the default values, as follows:

	-ker [r,c,m]			fitness function: (r)egression, (c)lassification, or (m)atching
	-typ [f,g,r]			Tree type: (f)ull, (g)row, or (r)amped half/half
	-bas [3...10]			maximum Tree depth for initial population
	-max [3...10]			maximum Tree depth for entire run
	-min [3 to 2^(bas +1) - 1]	minimum number of nodes
	-pop [10...1000]		number of trees in each generational population
	-gen [1...100]			number of generations
	-tor [7 per 100]		number of trees selected for tournament
	-evr [0.0...1.0]  		decimal percent of pop generated through Reproduction
	-evp [0.0...1.0]  		decimal percent of pop generated through Point Mutation
	-evb [0.0...1.0]  		decimal percent of pop generated through Branch Mutation
	-evc [0.0...1.0]  		decimal percent of pop generated through Crossover
	
If you include any of the above flags, then you *must* also include a flag to load an external dataset.

	-fil [path]/[to]/[data].csv	an external dataset


An example is given, as follows:

	$ python karoo_gp_server.py -ker c -typ r -bas 4 -fil [path]/[to]/[data].csv

'''

import os
import sys
import argparse
from karoo_gp import base_class, __version__
gp = base_class.Base_GP()

os.system('clear')
print ('\n\033[36m\033[1m')
print ('\t **   **   ******    *****    ******    ******       ******    ******')
print ('\t **  **   **    **  **   **  **    **  **    **     **        **    **')
print ('\t ** **    **    **  **   **  **    **  **    **     **        **    **')
print ('\t ****     ********  ******   **    **  **    **     **   ***  *******')
print ('\t ** **    **    **  ** **    **    **  **    **     **    **  **')
print ('\t **  **   **    **  **  **   **    **  **    **     **    **  **')
print ('\t **   **  **    **  **   **  **    **  **    **     **    **  **')
print ('\t **    ** **    **  **    **  ******    ******       ******   **')
print ('\033[0;0m')
print ('\t\033[36m Genetic Programming in Python with TensorFlow - by Kai Staats, version {}\033[0;0m'.format(__version__))
print ('')


#++++++++++++++++++++++++++++++++++++++++++
#   User Interface for Configuation       |
#++++++++++++++++++++++++++++++++++++++++++

if len(sys.argv) < 3: # either no command line argument, or only a filename is provided

	while True:
		try:
			query = input('\t Select (c)lassification, (r)egression, (m)atching, or (p)lay (default m): ')
			if query in ['c','r','m','p','']: kernel = query or 'm'; break
			else: raise ValueError()
		except ValueError: print ('\t\033[32m Select from the options given. Try again ...\n\033[0;0m')
		except KeyboardInterrupt: sys.exit()
		
	if kernel == 'p': # play mode
		while True:
			try:
				query = input('\t Select (f)ull or (g)row (default g): ')
				if query in ['f','g','']: tree_type = query or 'f'; break
				else: raise ValueError()
			except ValueError: print ('\t\033[32m Select from the options given. Try again ...\n\033[0;0m')
			except KeyboardInterrupt: sys.exit()
			
		while True:
			try:
				query = input('\t Enter the depth of the Tree (default 1): ')
				if query == '': tree_depth_base = 1; break
				elif int(query) in list(range(1,11)): tree_depth_base = int(query); break
				else: raise ValueError()
			except ValueError: print ('\t\033[32m Enter a number from 1 including 10. Try again ...\n\033[0;0m')
			except KeyboardInterrupt: sys.exit()
			
		tree_depth_max = tree_depth_base
		tree_depth_min = 3
		tree_pop_max = 1
		gen_max = 1
		tourn_size = 0
		display = 'm'
		#	evolve_repro, evolve_point, evolve_branch, evolve_cross, tourn_size, precision, filename are not required
	
	else: # if any other kernel is selected
		
		while True:
			try:
				query = input('\t Select (f)ull, (g)row, or (r)amped 50/50 method (default r): ')
				if query in ['f','g','r','']: tree_type = query or 'r'; break
				else: raise ValueError()
			except ValueError: print ('\t\033[32m Select from the options given. Try again ...\n\033[0;0m')
			except KeyboardInterrupt: sys.exit()
			
		while True:
			try:
				query = input('\t Enter depth of the \033[3minitial\033[0;0m population of Trees (default 3): ')
				if query == '': tree_depth_base = 3; break
				elif int(query) in list(range(1,11)): tree_depth_base = int(query); break
				else: raise ValueError()
			except ValueError: print ('\t\033[32m Enter a number from 1 including 10. Try again ...\n\033[0;0m')
			except KeyboardInterrupt: sys.exit()
			
		while True:
			try:
				query = input('\t Enter maximum Tree depth (default %s): ' %str(tree_depth_base))
				if query == '': tree_depth_max = tree_depth_base; break
				elif int(query) in list(range(tree_depth_base,11)): tree_depth_max = int(query); break
				else: raise ValueError()
			except ValueError: print ('\t\033[32m Enter a number from %s including 10. Try again ...\n\033[0;0m' %str(tree_depth_base))
			except KeyboardInterrupt: sys.exit()
			
		max_nodes = 2**(tree_depth_base+1)-1 # calc the max number of nodes for the given depth
		
		while True:
			try:
				query = input('\t Enter minimum number of nodes for any given Tree (default 3; max %s): ' %str(max_nodes))
				if query == '': tree_depth_min = 3; break
				elif int(query) in list(range(3,max_nodes + 1)): tree_depth_min = int(query); break
				else: raise ValueError()
			except ValueError: print ('\t\033[32m Enter a number from 3 including %s. Try again ...\n\033[0;0m' %str(max_nodes))
			except KeyboardInterrupt: sys.exit()
			
		#while True:
			#try:
				#query = input('\t Select (p)artial or (f)ull operator inclusion (default p): ')
				#if query == '': swim = 'p'; break
				#elif query in ['p','f']: swim = query; break
				#else: raise ValueError()
			#except ValueError: print ('\t\033[32m Select from the options given. Try again ...\n\033[0;0m')
			#except KeyboardInterrupt: sys.exit()
			
		while True:
			try:
				query = input('\t Enter number of Trees in each population (default 100): ')
				if query == '': tree_pop_max = 100; break
				elif int(query) in list(range(1,1001)): tree_pop_max = int(query); break
				else: raise ValueError()
			except ValueError: print ('\t\033[32m Enter a number from 1 including 1000. Try again ...\n\033[0;0m')
			except KeyboardInterrupt: sys.exit()
			
		# calculate the tournament size
		tourn_size = int(tree_pop_max * 0.07) # default 7% can be changed by selecting (g)eneration and then 'ts'
		if tourn_size < 2: tourn_size = 2 # forces some diversity for small populations
		if tree_pop_max == 1: tourn_size = 1 # in theory, supports the evolution of a single Tree - NEED TO FIX 2018 04/19
		
		while True:
			try:
				query = input('\t Enter max number of generations (default 10): ')
				if query == '': gen_max = 10; break
				elif int(query) in list(range(1,101)): gen_max = int(query); break
				else: raise ValueError()
			except ValueError: print ('\t\033[32m Enter a number from 1 including 100. Try again ...\n\033[0;0m')
			except KeyboardInterrupt: sys.exit()
			
		if gen_max > 1:
			while True:
				try:
					query = input('\t Display (i)nteractive, (g)eneration, (m)iminal, (s)ilent, or (d)e(b)ug (default m): ')
					if query in ['i','g','m','s','db','']: display = query or 'm'; break
					else: raise ValueError()
				except ValueError: print ('\t\033[32m Select from the options given. Try again ...\n\033[0;0m')
				except KeyboardInterrupt: sys.exit()
				
		else: display = 's' # display mode is not used, but a value must be passed
				
	### additional configuration parameters ###
	
	evolve_repro = int(0.1 * tree_pop_max) # quantity of a population generated through Reproduction
	evolve_point = int(0.1 * tree_pop_max) # quantity of a population generated through Point Mutation
	evolve_branch = int(0.2 * tree_pop_max) # quantity of a population generated through Branch Mutation
	evolve_cross = int(0.6 * tree_pop_max) # quantity of a population generated through Crossover
	filename = '' # not required unless an external file is referenced
	precision = 6 # number of floating points for the round function in 'fx_fitness_eval'
	swim = 'p' # require (p)artial or (f)ull set of features (operators) for each Tree entering the gene_pool
	mode = 'd' # pause at the (d)esktop when complete, awaiting further user interaction; or terminate in (s)erver mode
	

#++++++++++++++++++++++++++++++++++++++++++
#   Command Line for Configuation         |
#++++++++++++++++++++++++++++++++++++++++++

else: # 2 or more command line arguments are provided

	ap = argparse.ArgumentParser(description = 'Karoo GP Server')
	ap.add_argument('-ker', action = 'store', dest = 'kernel', default = 'c', help = '[c,r,m] fitness function: (r)egression, (c)lassification, or (m)atching')
	ap.add_argument('-typ', action = 'store', dest = 'type', default = 'r', help = '[f,g,r] Tree type: (f)ull, (g)row, or (r)amped half/half')
	ap.add_argument('-bas', action = 'store', dest = 'depth_base', default = 4, help = '[3...10] maximum Tree depth for the initial population')
	ap.add_argument('-max', action = 'store', dest = 'depth_max', default = 4, help = '[3...10] maximum Tree depth for the entire run')
	ap.add_argument('-min', action = 'store', dest = 'depth_min', default = 3, help = 'minimum nodes, from 3 to 2^(base_depth +1) - 1')
	ap.add_argument('-pop', action = 'store', dest = 'pop_max', default = 100, help = '[10...1000] number of trees per generation')
	ap.add_argument('-gen', action = 'store', dest = 'gen_max', default = 10, help = '[1...100] number of generations')
	ap.add_argument('-tor', action = 'store', dest = 'tor_size', default = 7, help = '[7 for each 100] recommended tournament size')
	ap.add_argument('-evr', action = 'store', dest = 'evo_r', default = 0.1, help = '[0.0-1.0] decimal percent of pop generated through Reproduction')
	ap.add_argument('-evp', action = 'store', dest = 'evo_p', default = 0.1, help = '[0.0-1.0] decimal percent of pop generated through Point Mutation')
	ap.add_argument('-evb', action = 'store', dest = 'evo_b', default = 0.2, help = '[0.0-1.0] decimal percent of pop generated through Branch Mutation')
	ap.add_argument('-evc', action = 'store', dest = 'evo_c', default = 0.6, help = '[0.0-1.0] decimal percent of pop generated through Crossover')
	ap.add_argument('-fil', action = 'store', dest = 'filename', default = '', help = '/path/to_your/[data].csv')
	
	args = ap.parse_args()

	# pass the argparse defaults and/or user inputs to the required variables
	kernel = str(args.kernel)
	tree_type = str(args.type)
	tree_depth_base = int(args.depth_base)
	tree_depth_max = int(args.depth_max)
	tree_depth_min = int(args.depth_min)
	tree_pop_max = int(args.pop_max)
	gen_max = int(args.gen_max)
	tourn_size = int(args.tor_size)
	evolve_repro = int(float(args.evo_r) * tree_pop_max)
	evolve_point = int(float(args.evo_p) * tree_pop_max)
	evolve_branch = int(float(args.evo_b) * tree_pop_max)
	evolve_cross = int(float(args.evo_c) * tree_pop_max)
	filename = str(args.filename)
	
	display = 's' # display mode is set to (s)ilent
	precision = 6 # number of floating points for the round function in 'fx_fitness_eval'
	swim = 'p' # require (p)artial or (f)ull set of features (operators) for each Tree entering the gene_pool
	mode = 's' # pause at the (d)esktop when complete, awaiting further user interaction; or terminate in (s)erver mode
	

#++++++++++++++++++++++++++++++++++++++++++
#   Conduct the GP run                    |
#++++++++++++++++++++++++++++++++++++++++++

gp.fx_karoo_gp(kernel, tree_type, tree_depth_base, tree_depth_max, tree_depth_min, tree_pop_max, gen_max, tourn_size, filename, evolve_repro, evolve_point, evolve_branch, evolve_cross, display, precision, swim, mode)


