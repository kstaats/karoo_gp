# Karoo GP Main (desktop)
# Use Genetic Programming for Classification and Symbolic Regression
# by Kai Staats, MSc; see LICENSE.md
# version 1.1

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

import sys; sys.path.append('modules/') # add directory 'modules' to the current path
import os
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


#++++++++++++++++++++++++++++++++++++++++++
#   User Defined Configuration            |
#++++++++++++++++++++++++++++++++++++++++++

'''
Karoo GP queries the user for key parameters, some of which may be adjusted during run-time at user invoked pauses. 
See the User Guide for meaning and value of each of the following parameters. The server version of Karoo enables 
all parameters to be configured via command-line arguments.
'''

menu = ['c','r','m','p','']
while True:
	try:
		kernel = raw_input('\t Select (c)lassification, (r)egression, (m)atching, or (p)lay (default m): ')
		if kernel not in menu: raise ValueError()
		kernel = kernel or 'm'; break
	except ValueError: print '\t\033[32m Select from the options given. Try again ...\n\033[0;0m'
	except KeyboardInterrupt: sys.exit()
	
if kernel == 'p':

	menu = ['f','g','']
	while True:
		try:
			tree_type = raw_input('\t Select (f)ull or (g)row method (default f): ')
			if tree_type not in menu: raise ValueError()
			tree_type = tree_type or 'f'; break
		except ValueError: print '\t\033[32m Select from the options given. Try again ...\n\033[0;0m'
		except KeyboardInterrupt: sys.exit()

else:

	menu = ['f','g','r','']
	while True:
		try:
			tree_type = raw_input('\t Select (f)ull, (g)row, or (r)amped 50/50 method (default r): ')
			if tree_type not in menu: raise ValueError()
			tree_type = tree_type or 'r'; break
		except ValueError: print '\t\033[32m Select from the options given. Try again ...\n\033[0;0m'
		except KeyboardInterrupt: sys.exit()
	
menu = range(1,11)
while True:
	try:
		tree_depth_base = raw_input('\t Enter depth of the \033[3minitial\033[0;0m population of Trees (default 3): ')
		if tree_depth_base not in str(menu) or tree_depth_base == '0': raise ValueError()
		elif tree_depth_base == '': tree_depth_base = 3; break
		tree_depth_base = int(tree_depth_base); break
	except ValueError: print '\t\033[32m Enter a number from 1 including 10. Try again ...\n\033[0;0m'
	except KeyboardInterrupt: sys.exit()
	

if kernel == 'p': # if the Play kernel is selected
	tree_depth_max = tree_depth_base
	tree_depth_min = 0
	tree_pop_max = 1
	generation_max = 1
	display = 'm'
#	evolve_repro = evolve_point = evolve_branch = evolve_cross = ''
#	tourn_size = ''
#	precision = ''
#	filename = ''
	
else: # if any other kernel is selected

	if tree_type == 'f': tree_depth_max = tree_depth_base
	else: # if type is Full, the maximum Tree depth for the full run is equal to the initial population
	
		menu = range(tree_depth_base,11)
		while True:
			try:
				tree_depth_max = raw_input('\t Enter maximum Tree depth (default %i): ' %tree_depth_base)
				if tree_depth_max not in str(menu): raise ValueError()
				elif tree_depth_max == '': tree_depth_max = tree_depth_base
				tree_depth_max = int(tree_depth_max)
				if tree_depth_max < tree_depth_base: raise ValueError() # an ugly exception to the norm 20170918
				else: break
			except ValueError: print '\t\033[32m Enter a number >= the initial Tree depth. Try again ...\n\033[0;0m'
			except KeyboardInterrupt: sys.exit()
			
	menu = range(3,101)
	while True:
		try:
			tree_depth_min = raw_input('\t Enter minimum number of nodes for any given Tree (default 3): ')
			if tree_depth_min not in str(menu) or tree_depth_min == '0': raise ValueError()
			elif tree_depth_min == '': tree_depth_min = 3
			tree_depth_min = int(tree_depth_min); break
		except ValueError: print '\t\033[32m Enter a number from 3 to 2^(depth + 1) - 1 including 100. Try again ...\n\033[0;0m'
		except KeyboardInterrupt: sys.exit()
		
	menu = range(10,1001)
	while True:
		try:
			tree_pop_max = raw_input('\t Enter number of Trees in each population (default 100): ')
			if tree_pop_max not in str(menu) or tree_pop_max == '0': raise ValueError()
			elif tree_pop_max == '': tree_pop_max = 100
			tree_pop_max = int(tree_pop_max); break
		except ValueError: print '\t\033[32m Enter a number from 10 including 1000. Try again ...\n\033[0;0m'
		except KeyboardInterrupt: sys.exit()
		
	menu = range(1,101)
	while True:
		try:
			generation_max = raw_input('\t Enter max number of generations (default 10): ')
			if generation_max not in str(menu) or generation_max == '0': raise ValueError()
			elif generation_max == '': generation_max = 10
			generation_max = int(generation_max); break
		except ValueError: print '\t\033[32m Enter a number from 1 including 100. Try again ...\n\033[0;0m'
		except KeyboardInterrupt: sys.exit()
		
	menu = ['i','g','m','s','db','']
	while True:
		try:
			display = raw_input('\t Display (i)nteractive, (g)eneration, (m)iminal, (s)ilent, or (d)e(b)ug (default m): ')
			if display not in menu: raise ValueError()
			display = display or 'm'; break
		except ValueError: print '\t\033[32m Select from the options given. Try again ...\n\033[0;0m'
		except KeyboardInterrupt: sys.exit()
		

# define the ratio between types of mutation, where all sum to 1.0; can be adjusted in 'i'nteractive mode
evolve_repro = int(0.1 * tree_pop_max) # quantity of a population generated through Reproduction
evolve_point = int(0.0 * tree_pop_max) # quantity of a population generated through Point Mutation
evolve_branch = int(0.2 * tree_pop_max) # quantity of a population generated through Branch Mutation
evolve_cross = int(0.7 * tree_pop_max) # quantity of a population generated through Crossover

tourn_size = 7 # qty of individuals entered into each tournament (standard = 7%); can be adjusted in 'i'nteractive mode
precision = 6 # the number of floating points for the round function in 'fx_fitness_eval'; hard coded
filename = '' # not required unless an external file is referenced

# pass all user defined settings to the base_class and launch Karoo GP
gp.fx_karoo_gp(kernel, tree_type, tree_depth_base, tree_depth_max, tree_depth_min, tree_pop_max, generation_max, tourn_size, filename, evolve_repro, evolve_point, evolve_branch, evolve_cross, display, precision, 'm')

print 'You seem to have found your way back to the Desktop. Huh.'
sys.exit()

