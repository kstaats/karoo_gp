# Karoo GP Server
# Use Genetic Programming for Classification and Symbolic Regression
# by Kai Staats, MSc; see LICENSE.md
# version 1.1

'''
A word to the newbie, expert, and brave--
Even if you are highly experienced in Genetic Programming, it is recommended that you review the 'Karoo User Guide' 
before running this application. While your computer will not burst into flames nor will the sun collapse into a black 
hole if you do not, you will likely find more enjoyment of this particular flavour of GP with a little understanding 
of its intent and design.

KAROO GP SERVER
This is the Karoo GP server application. It can be internally scripted, fully command-line configured, or a combination
of both. If this is your first time using Karoo GP, please run the desktop application karoo_gp_main.py first in order 
that you come to understand the full functionality of this particular Genetic Programming platform.

To launch Karoo GP server:

	$ python karoo_gp_server.py
	
	  (or from iPython)
	
	$ run karoo_gp_server.py


Without any arguments, Karoo GP relies entirely upon the scripted settings and the datasets located in karoo_gp/files/.

If you include the path to an external dataset, it will auto-load at launch:

	$ python karoo_gp_server.py /[path]/[to_your]/[filename].csv
	

You can include a number of additional arguments which override the default values, as follows:

	-ker		[r,c,m]				fitness function: (r)egression, (c)lassification, or (m)atching
	-typ		[f,g,r]				Tree type: (f)ull, (g)row, or (r)amped half/half
	-bas		[3...10]			maximum Tree depth for the initial population
	-max		[3...10]			maximum Tree depth for the entire run
	-min		[3...100]			minimum number of nodes
	-pop		[10...1000]		maximum population
	-gen		[1...100]			number of generations
	-tor		[1...100]			number of trees selected for the tournament
	-fil		[filename]		an external dataset
	
Note that if you include any of the above flags, then you must also include a flag to load an external dataset.

An example is given, as follows:

	$ python karoo_gp_server.py -ker c -typ r -bas 4 -fil /[path]/[to_your]/[filename].csv
	
'''

import sys; sys.path.append('modules/') # to add the directory 'modules' to the current path
import os
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
print '\t\033[36m Genetic Programming in Python - by Kai Staats, version 1.1\033[0;0m'
print ''

ap = argparse.ArgumentParser(description = 'Karoo GP Server')
ap.add_argument('-ker', action = 'store', dest = 'kernel', default = 'c', help = '[c,r,m] fitness function: (r)egression, (c)lassification, or (m)atching')
ap.add_argument('-typ', action = 'store', dest = 'type', default = 'r', help = '[f,g,r] Tree type: (f)ull, (g)row, or (r)amped half/half')
ap.add_argument('-bas', action = 'store', dest = 'depth_base', default = 3, help = '[3...10] maximum Tree depth for the initial population')
ap.add_argument('-max', action = 'store', dest = 'depth_max', default = 5, help = '[3...10] maximum Tree depth for the entire run')
ap.add_argument('-min', action = 'store', dest = 'depth_min', default = 3, help = '[3...100] minimum number of nodes')
ap.add_argument('-pop', action = 'store', dest = 'pop_max', default = 100, help = '[10...1000] maximum population')
ap.add_argument('-gen', action = 'store', dest = 'gen_max', default = 10, help = '[1...100] number of generations')
ap.add_argument('-tor', action = 'store', dest = 'tor_size', default = 7, help = '[1...max pop] tournament size')
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
filename = str(args.filename)

evolve_repro = int(0.1 * tree_pop_max) # quantity of a population generated through Reproduction
evolve_point = int(0.0 * tree_pop_max) # quantity of a population generated through Point Mutation
evolve_branch = int(0.2 * tree_pop_max) # quantity of a population generated through Branch Mutation
evolve_cross = int(0.7 * tree_pop_max) # quantity of a population generated through Crossover

display = 's' # display mode is set to (s)ilent
precision = 6 # the number of floating points for the round function in 'fx_fitness_eval'

# pass all user defined settings to the base_class and launch Karoo GP
gp.fx_karoo_gp(kernel, tree_type, tree_depth_base, tree_depth_max, tree_depth_min, tree_pop_max, generation_max, tourn_size, filename, evolve_repro, evolve_point, evolve_branch, evolve_cross, display, precision, 's')

sys.exit()

