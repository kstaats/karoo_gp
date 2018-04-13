# Karoo GP (desktop + server combined)
# Use Genetic Programming for Classification and Symbolic Regression
# by Kai Staats, MSc; see LICENSE.md
# version 2.0

'''
A word to the newbie, expert, and brave--
Even if you are highly experienced in Genetic Programming, it is recommended that you review the 'Karoo User Guide' 
before running this application. While your computer will not burst into flames nor will the sun collapse into a black 
hole if you do not, you will likely find more enjoyment of this particular flavour of GP with a little understanding 
of its intent and design.

Without any arguments, Karoo GP relies entirely upon the scripted settings and the datasets located in karoo_gp/files/.

	$ python karoo_gp_main.py
	
	  (or from iPython)
	
	$ run karoo_gp_main.py


If you include the path to an external dataset, it will auto-load at launch:

	$ python karoo_gp_main.py /[path]/[to_your]/[filename].csv
	

You can include a one or more additional arguments, they will override the default values, as follows:

	-ker [r,c,m]				fitness function: (r)egression, (c)lassification, or (m)atching
	-typ [f,g,r]				Tree type: (f)ull, (g)row, or (r)amped half/half
	-bas [3...10]				maximum Tree depth for the initial population
	-max [3...10]				maximum Tree depth for the entire run
	-min [3...100]			minimum number of nodes
	-pop [10...1000]		maximum population
	-gen [1...100]			number of generations
	-tor [1...100]			number of trees selected for the tournament
	-evr [0.0 ... 1.0]  fraction percent of genetic operator Reproduction
	-evp [0.0 ... 1.0]  fraction percent of genetic operator Point Mutation
	-evb [0.0 ... 1.0]  fraction percent of genetic operator Branch Mutation
	-evc [0.0 ... 1.0]  fraction percent of genetic operator Crossover
	
If you include any of the above flags, then you *must* also include a flag to load an external dataset.

	-fil [filename]			an external dataset


An example is given, as follows:

	$ python karoo_gp_server.py -ker c -typ r -bas 4 -fil /[path]/[to_your]/[filename].csv

'''

import os
import sys; sys.path.append('modules/') # add directory 'modules' to the current path
import argparse
import karoo_gp_base_class; gp = karoo_gp_base_class.Base_GP()

os.system('clear')
print '\n\033[36m\033[1m'
print '\t **   **   ******    *****    ******    ******       ******    ******'
print '\t **  **   **    **  **   **  **    **  **    **     **        **    **'
print '\t ** **    **    **  **   **  **    **  **    **     **        **    **'
print '\t ****     ********  ******   **    **  **    **     **   ***  *******'
print '\t ** **    **    **  ** **    **    **  **    **     **    **  **'
print '\t **  **   **    **  **  **   **    **  **    **     **    **  **'
print '\t **   **  **    **  **   **  **    **  **    **     **    **  **'
print '\t **    ** **    **  **    **  ******    ******       ******   **'
print '\033[0;0m'
print '\t\033[36m Genetic Programming in Python - by Kai Staats, version 1.2\033[0;0m'
print ''


#++++++++++++++++++++++++++++++++++++++++++
#   User Interface Configuation           |
#++++++++++++++++++++++++++++++++++++++++++

if len(sys.argv) < 3: # either no command line argument (1) or the filename (2) was provided

	# menu = ['c','r','m','p',''] # inserted all menus directly into while loops on 2018 05/07
	while True:
		try:
			query = raw_input('\t Select (c)lassification, (r)egression, (m)atching, or (p)lay (default m): ')
			if query not in ['c','r','m','p','']: raise ValueError()
			kernel = query or 'm'; break
		except ValueError: print '\t\033[32m Select from the options given. Try again ...\n\033[0;0m'
		except KeyboardInterrupt: sys.exit()
	
	if kernel == 'p':

		while True:
			try:
				tree_type = raw_input('\t Select (f)ull or (g)row (default g): ')
				if tree_type not in ['f','g','']: raise ValueError()
				tree_type = tree_type or 'f'; break
			except ValueError: print '\t\033[32m Select from the options given. Try again ...\n\033[0;0m'
			except KeyboardInterrupt: sys.exit()
	
		while True:
			try:
				tree_depth_base = raw_input('\t Enter the depth of the Tree (default 1): ')
				if tree_depth_base not in str(range(1,11)) or tree_depth_base == '0': raise ValueError()
				elif tree_depth_base == '': tree_depth_base = 1; break
				tree_depth_base = int(tree_depth_base); break
			except ValueError: print '\t\033[32m Enter a number from 1 including 10. Try again ...\n\033[0;0m'
			except KeyboardInterrupt: sys.exit()
		
		tree_depth_max = tree_depth_base
		tree_depth_min = 3
		tree_pop_max = 1
		generation_max = 1
		tourn_size = 0
		display = 'm'
		#	evolve_repro, evolve_point, evolve_branch, evolve_cross, tourn_size, precision, filename are not required
	
	else: # if any other kernel is selected

		while True:
			try:
				tree_type = raw_input('\t Select (f)ull, (g)row, or (r)amped 50/50 method (default r): ')
				if tree_type not in ['f','g','r','']: raise ValueError()
				tree_type = tree_type or 'r'; break
			except ValueError: print '\t\033[32m Select from the options given. Try again ...\n\033[0;0m'
			except KeyboardInterrupt: sys.exit()
		
		while True:
			try:
				tree_depth_base = raw_input('\t Enter depth of the \033[3minitial\033[0;0m population of Trees (default 3): ')
				if tree_depth_base not in str(range(1,11)) or tree_depth_base == '0': raise ValueError()
				elif tree_depth_base == '': tree_depth_base = 3; break
				tree_depth_base = int(tree_depth_base); break
			except ValueError: print '\t\033[32m Enter a number from 1 including 10. Try again ...\n\033[0;0m'
			except KeyboardInterrupt: sys.exit()
		
		if tree_type == 'f': tree_depth_max = tree_depth_base	
		else: # if type is Full, tree_depth_max is equal to tree_depth_base (initial pop setting) 
			while True:
				try:
					tree_depth_max = raw_input('\t Enter maximum Tree depth (default %i): ' %tree_depth_base)
					if tree_depth_max not in str(range(tree_depth_base,11)): raise ValueError()
					elif tree_depth_max == '': tree_depth_max = tree_depth_base
					tree_depth_max = int(tree_depth_max)
					if tree_depth_max < tree_depth_base: raise ValueError() # if max is set to < min 20170918
					else: break
				except ValueError: print '\t\033[32m Enter a number > or = the initial Tree depth. Try again ...\n\033[0;0m'
				except KeyboardInterrupt: sys.exit()
				
		max_nodes = 2**(tree_depth_base +1) - 1 # auto calc the max number of nodes for the given depth
		
		while True:
			try:
				tree_depth_min = raw_input('\t Enter minimum number of nodes for any given Tree (default 3; max %s): ' %str(max_nodes))
				if tree_depth_min not in str(range(3,max_nodes)) or tree_depth_min == '0' or tree_depth_min == '1' or tree_depth_min == '2': raise ValueError()
				elif tree_depth_min == '': tree_depth_min = 3
				tree_depth_min = int(tree_depth_min); break
			except ValueError: print '\t\033[32m Enter a number from 3 including %s. Try again ...\n\033[0;0m' %str(max_nodes)
			except KeyboardInterrupt: sys.exit()
		
		while True:
			try:
				tree_pop_max = raw_input('\t Enter number of Trees in each population (default 100): ')
				if tree_pop_max not in str(range(1,1001)) or tree_pop_max == '0': raise ValueError()
				elif tree_pop_max == '': tree_pop_max = 100
				tree_pop_max = int(tree_pop_max); break
			except ValueError: print '\t\033[32m Enter a number from 1 including 1000. Try again ...\n\033[0;0m'
			except KeyboardInterrupt: sys.exit()
			
		tourn_size = int(tree_pop_max * 0.07) # default 7% can be changed by selecting Generation, 'ts', and then enter the run.
		
		while True:
			try:
				generation_max = raw_input('\t Enter max number of generations (default 10): ')
				if generation_max not in str(range(1,101)) or generation_max == '0': raise ValueError()
				elif generation_max == '': generation_max = 10
				generation_max = int(generation_max); break
			except ValueError: print '\t\033[32m Enter a number from 1 including 100. Try again ...\n\033[0;0m'
			except KeyboardInterrupt: sys.exit()
		
		while True:
			try:
				display = raw_input('\t Display (i)nteractive, (g)eneration, (m)iminal, (s)ilent, or (d)e(b)ug (default m): ')
				if display not in ['i','g','m','s','db','']: raise ValueError()
				display = display or 'm'; break
			except ValueError: print '\t\033[32m Select from the options given. Try again ...\n\033[0;0m'
			except KeyboardInterrupt: sys.exit()
			
	evolve_repro = int(0.1 * tree_pop_max) # quantity of a population generated through Reproduction
	evolve_point = int(0.0 * tree_pop_max) # quantity of a population generated through Point Mutation
	evolve_branch = int(0.2 * tree_pop_max) # quantity of a population generated through Branch Mutation
	evolve_cross = int(0.7 * tree_pop_max) # quantity of a population generated through Crossover
	filename = '' # not required unless an external file is referenced
	precision = 6 # number of floating points for the round function in 'fx_fitness_eval'
	mode = 'm' # pause when complete, awaiting further user interaction


#++++++++++++++++++++++++++++++++++++++++++
#   Command Line Configuation             |
#++++++++++++++++++++++++++++++++++++++++++

else: # two or more command line argument were provided

	ap = argparse.ArgumentParser(description = 'Karoo GP Server')
	ap.add_argument('-ker', action = 'store', dest = 'kernel', default = 'c', help = '[c,r,m] fitness function: (r)egression, (c)lassification, or (m)atching')
	ap.add_argument('-typ', action = 'store', dest = 'type', default = 'r', help = '[f,g,r] Tree type: (f)ull, (g)row, or (r)amped half/half')
	ap.add_argument('-bas', action = 'store', dest = 'depth_base', default = 4, help = '[3...10] maximum Tree depth for the initial population')
	ap.add_argument('-max', action = 'store', dest = 'depth_max', default = 4, help = '[3...10] maximum Tree depth for the entire run')
	ap.add_argument('-min', action = 'store', dest = 'depth_min', default = 3, help = 'The minimum number of nodes ranges from 3 to 2^(base_depth +1) - 1')
	ap.add_argument('-pop', action = 'store', dest = 'pop_max', default = 100, help = '[10...1000] maximum population')
	ap.add_argument('-gen', action = 'store', dest = 'gen_max', default = 10, help = '[1...100] number of generations')
	ap.add_argument('-tor', action = 'store', dest = 'tor_size', default = 7, help = '[7 for each 100 recommended] tournament size')
	ap.add_argument('-evr', action = 'store', dest = 'evo_r', default = 0.1, help = '[0.0-1.0] fraction of pop generated through Reproduction')
	ap.add_argument('-evp', action = 'store', dest = 'evo_p', default = 0.0, help = '[0.0-1.0] fraction of pop generated through Point Mutation')
	ap.add_argument('-evb', action = 'store', dest = 'evo_b', default = 0.2, help = '[0.0-1.0] fraction of pop generated through Branch Mutation')
	ap.add_argument('-evc', action = 'store', dest = 'evo_c', default = 0.7, help = '[0.0-1.0] fraction of pop generated through Crossover')
	ap.add_argument('-fil', action = 'store', dest = 'filename', default = '', help = '/path/to_your/[data].csv')
	
	args = ap.parse_args()

	# pass the argparse defaults and/or user inputs to the required variables
	kernel = str(args.kernel)
	tree_type = str(args.type)
	tree_depth_base = int(args.depth_base)
	tree_depth_max = int(args.depth_max)
	tree_depth_min = int(args.depth_min)
	tree_pop_max = int(args.pop_max)
	generation_max = int(args.gen_max)
	tourn_size = int(args.tor_size)
	evolve_repro = int(float(args.evo_r) * tree_pop_max) # quantity of each population generated through Reproduction
	evolve_point = int(float(args.evo_p) * tree_pop_max) # quantity of each population generated through Point Mutation
	evolve_branch = int(float(args.evo_b) * tree_pop_max) # quantity of each population generated through Branch Mutation
	evolve_cross = int(float(args.evo_c) * tree_pop_max) # quantity of each population generated through Crossover
	filename = str(args.filename)
	
	display = 's' # display mode is set to (s)ilent
	precision = 6 # number of floating points for the round function in 'fx_fitness_eval'
	mode = 's' # drop back to the command line when complete
	

#++++++++++++++++++++++++++++++++++++++++++
#   Pass all settings to the base_class   |
#++++++++++++++++++++++++++++++++++++++++++
	
gp.fx_karoo_gp(kernel, tree_type, tree_depth_base, tree_depth_max, tree_depth_min, tree_pop_max, generation_max, tourn_size, filename, evolve_repro, evolve_point, evolve_branch, evolve_cross, display, precision, mode)

sys.exit()


