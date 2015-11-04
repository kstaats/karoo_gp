# Karoo GP Base Class
# Define the Karoo GP methods and global variables
# by Kai Staats, MSc UCT / AIMS
# Much thanks to Emmanuel Dufourq and Arun Kumar for their support, guidance, and free psychotherapy sessions
# version 0.9

import csv
import os
import sys
import time

import numpy as np
import pprocess as pp
import sklearn.metrics as skm
import sklearn.cross_validation as skcv
import sympy as sp

np.set_printoptions(linewidth = 320) # set the terminal to print 320 characters before line-wrapping in order to view Trees

class Base_GP(object):

	'''
	This Base_BP class contains all methods for Karoo GP.
	
	Method names are differentiated from global variable names (defined below) by the prefix 'fx_'. Following the 'fx_'
	the name contains the category, object, and action: fx_[category]_[object]_[action]. For example,
	'fx_eval_tree_print()' is a method (Python function) denoted by 'fx_', in the category Evaluation, and a 'tree' 
	will 'print' to screen.
	
	The categories (denoted by #+++++++ banners) are as follows:
		'fx_karoo_'		Methods to Run Karoo GP (with the exception of top-level 'karoo_gp' itself)
		'fx_gen_'		Methods to Generate a Tree
		'fx_eval_'		Methods to Evaluate a Tree
		'fx_fitness_'	Methods to Evaluate Tree Fitness
		'fx_evo_'		Methods to Evolve a Population
		'fx_test_'		Methods to Test a Tree
		'fx_tree_'		Methods to Append & Archive
		
	There are no sub-classes at the time of this edit - 2015 09/21
	'''
	
	#++++++++++++++++++++++++++++++++++++++++++
	#   Define Global Variables               |
	#++++++++++++++++++++++++++++++++++++++++++
	
	def __init__(self):
	
		'''
		All Karoo GP global variables are named with the prefix 'gp.' All Karoo GP methods are named with the prefix 
		'gp.fx_'. The 13 variables which begin with 'gp.pop_' are used specifically to define the 13 parameters for 
		each GP as stored in the axis-1 (expanding horizontally on-screen) 'gp.population' Numpy array.
		
		### Variables defined by the user in karoo_gp_main.py (in order of appearence) ###
		'gp.kernel'					fitness function
		'gp.class_method'			select the number of classes (will be automated in future version)
		'tree_type'					Full, Grow, or Ramped 50/50 (local variable)
		'gp.tree_depth_min'			minimum number of nodes
		'tree_depth_max'			maximum number of nodes [nodes = 2^(depth + 1) - 1] (local variable)
		'gp.tree_pop_max'			maximum number of Trees per generation
		'gp.generation_max'			maximum number of generations
		'gp.display'				level of on-screen feedback
		
		'gp.evolve_repro'			% of 1.0 expressed as a decimal value
		'gp.evolve_point'			% of 1.0 expressed as a decimal value
		'gp.evolve_branch'			% of 1.0 expressed as a decimal value
		'gp.evolve_cross'			% of 1.0 expressed as a decimal value
		
		'gp.tourn_size'				the number of Trees chosen for each tournament
		'gp.cores'					user defined or default to 1; can be set to auto-detect number of cores instead
		'gp.precision'				the number of floating points for the round function in 'fx_fitness_eval'
		
		### Variables initiated elsewhere, as used for data management ###		
		'gp.data_train_cols'		number of cols in the TRAINING data (see 'fx_karoo_data_load', below)
		'gp.data_train_rows'		number of rows in the TRAINING data (see 'fx_karoo_data_load', below)
		'data_train_dict'			temporary dictionary which stores the data row-by-row (local variable)
		'gp.data_train_dict_array'	array of dictionaries which stores the TRAINING data, through all generations
		
		'gp.data_test_cols'			number of cols in the TEST data (see 'fx_karoo_data_load', below)
		'gp.data_test_rows'			number of rows in the TEST data (see 'fx_karoo_data_load', below)
		'data_test_dict'			temporary dictionary which stores the data row-by-row (local variable)
		'gp.data_test_dict_array'	array of dictionaries which stores the TEST data for the very end
		
		'gp.functions'				loaded from the associated [functions].csv
		'gp.terminals'				the top row of the associated [data].csv
		
		### Variables initiated elsewhere, as used for evolutionary management ###
		'gp.population_a'			the root generation from which Trees are chosen for mutation and reproduction
		'gp.population_b'			the generation constructed from gp.population_a (recyled)
		'gp.gene_pool'				once-per-generation assessment of trees that meet min and max boundary conditions
		'gp.generation_id'			simple n + 1 increment
		'gp.fitness_type'			set in 'fx_karoo_data_load' as either a minimising or maximising function
		'gp.tree'					axis-1, 13 element Numpy array that defines each Tree, stored in 'gp.population'
		'gp.pop_'					13 elements which define each Tree (see 'fx_gen_tree_initialise' below)
		
		### Fishing nets ###
		You can insert a "fishing net" to search for a specific polynomial expression when you fear the evolutionary
		process or something in the code may not be working. Search for "fishing net" and follow the directions.
		
		### Error checks ###
		You can quickly find all places in which error checks have been inserted by searching for "ERROR!"
		'''
		
		self.algo_raw = 0 # temp store the raw polynomial -- CONSIDER MAKING THIS VARIABLE LOCAL
		self.algo_sym = 0 # temp store the sympified polynomial-- CONSIDER MAKING THIS VARIABLE LOCAL
		self.fittest_dict = {} # temp store all Trees which share the best fitness score
		self.gene_pool = [] # temp store all Tree IDs for use by Tournament
		self.core_count = pp.get_number_of_cores()
		
		return
		
	
	#++++++++++++++++++++++++++++++++++++++++++
	#   Methods to Run Karoo GP               |
	#++++++++++++++++++++++++++++++++++++++++++
	
	def karoo_gp(self, run, tree_type, tree_depth_max):
	
		'''
		This is THE single method that enables the engagement of the entire Karoo GP application. The intent to is to 
		give Karoo GP an SKLearn-like interface for rapid integration into scientific applications. This method enables 
		remotely operated execution and use of scriptable, configuration files.
		
		The file "karoo_gp_server.py" included with the Karoo GP package is an example of a bare-bones configuration 
		and launch file.
		
		You must pass the banner display mode, tree_type, and tree_depth_max, along with configuration of all the 
		variables defined in karoo_gp_server.py, to get the whole thing rolling.
		'''
		
		self.karoo_banner(run)
		
		# construct first generation of Trees
		self.fx_karoo_data_load()
		self.generation_id = 1 # set initial generation ID
		self.population_a = ['Karoo GP by Kai Staats, Generation ' + str(self.generation_id)] # a list which will store all Tree arrays, one generation at a time
		self.fx_karoo_construct(tree_type, tree_depth_max) # construct the first population of Trees
		
		# evaluate first generation of Trees	
		print '\n Evaluate the first generation of Trees ...'	
		self.fx_fitness_gym(self.population_a) # run 'fx_eval', 'fx_fitness', 'fx_fitness_store', and fitness record
		self.fx_tree_archive(self.population_a, 'a') # save the first generation of Trees to disk
		
		# evolve subsequent generations of Trees
		for self.generation_id in range(2, self.generation_max + 1): # loop through 'generation_max'
		
			print '\n Evolve a population of Trees for Generation', self.generation_id, '...'
			self.population_b = ['Karoo GP by Kai Staats, Evolving Generation'] # initialise population_b to host the next generation
			
			self.fx_fitness_gene_pool() # generate the viable gene pool (compares against gp.tree_depth_min)
			self.fx_karoo_reproduce() # method 1 - Reproduction
			self.fx_karoo_point_mutate() # method 2 - Point Mutation
			self.fx_karoo_branch_mutate() # method 3 - Branch Mutation
			self.fx_karoo_crossover_reproduce() # method 4 - Crossover Reproduction
			self.fx_eval_generation() # evaluate all Trees in a single generation
			
			self.population_a = self.fx_evo_pop_copy(self.population_b, ['GP Tree by Kai Staats, Generation ' + str(self.generation_id)])
			
		# "End of line, man!" --CLU
		self.fx_tree_archive(self.population_b, 'f') # save the final generation of Trees to disk
		self.fx_karoo_eol()
		
		return
		
	
	def karoo_banner(self, run):
	
		'''
		This method makes Karoo GP look old-school cool!
		
		While the banner remains the same, it presents a configuration request unique to a 'server' run.
		
		At the time of this writing, the only options are 'server' or 'main' where 'main' defaults to requests for 
		feedback based upon the display mode selected by the user.
		
		See 'fx_karoo_construct' for examples.
		'''
		
		os.system('clear')
		
		print '\n\033[36m\033[1m'
		print '\t **   **   ******    *****    ******    ******       ******   ******'
		print '\t **  **   **    **  **   **  **    **  **    **     **        **   **'
		print '\t ** **    **    **  **   **  **    **  **    **     **        **   **'
		print '\t ****     ********  ******   **    **  **    **     **   ***  ******'
		print '\t ** **    **    **  ** **    **    **  **    **     **    **  **'
		print '\t **  **   **    **  **  **   **    **  **    **     **    **  **'
		print '\t **   **  **    **  **   **  **    **  **    **     **    **  **'
		print '\t **    ** **    **  **    **  ******    ******       ******   **'
		print '\033[0;0m'
		print '\t\033[36m Genetic Programming in Python - by Kai Staats, version 0.9\033[0;0m'
		
		if run == 'server':
			print '\n\t Type \033[1m?\033[0;0m to configure Karoo GP before your run, or \033[1mENTER\033[0;0m to continue.\033[0;0m'
			self.fx_karoo_pause(0)
			
		elif run == 'main': pass
		
		else: pass
		
		return
		
	
	def fx_karoo_data_load(self):
	
		'''
		The data and function .csv files are loaded according to the fitness function kernel selected by the user. An
		alternative dataset may be loaded at launch, by appending a command line argument. The data is then split into 
		both TRAINING and TEST segments in order to validate the success of the GP training run. Datasets less than
		10 rows will not be split, rather copied in full to both TRAINING and TEST as it is assumed you are conducting
		a system validation run, as with the built-in MATCH kernel and associated dataset.
		'''
		
		### 1) load the data file associated with the user selected fitness kernel ###	
		data_dict = {'a':'files/data_ABS.csv', 'b':'files/data_BOOL.csv', 'c':'files/data_CLASSIFY.csv', 'm':'files/data_MATCH.csv', 'p':'files/data_PLAY.csv'}
		func_dict = {'a':'files/functions_ABS.csv', 'b':'files/functions_BOOL.csv', 'c':'files/functions_CLASSIFY.csv', 'm':'files/functions_MATCH.csv', 'p':'files/functions_PLAY.csv'}
		fitt_dict = {'a':'min', 'b':'max', 'c':'max', 'm':'max', 'p':''}
		
		if len(sys.argv) == 1:
			data_x = np.loadtxt(data_dict[self.kernel], skiprows = 1, delimiter = ',', dtype = float); data_x = data_x[:,0:-1]
			data_y = np.loadtxt(data_dict[self.kernel], skiprows = 1, usecols = (-1,), delimiter = ',', dtype = float)
			
		elif len(sys.argv) == 2:
			print '\n\t\033[36m You have opted to load the alternative dataset:', sys.argv[1], '\033[0;0m'
			data_x = np.loadtxt(sys.argv[1], skiprows = 1, delimiter = ',', dtype = float); data_x = data_x[:,0:-1]
			data_y = np.loadtxt(sys.argv[1], skiprows = 1, usecols = (-1,), delimiter = ',', dtype = float)
			
		else: print '\n\t\033[31mERROR! You have assigned too many command line arguments at launch. Try again ...\033[0;0m'; sys.exit()
		
		header = open(data_dict[self.kernel],'r')
		self.terminals = header.readline().split(','); self.terminals[-1] = self.terminals[-1].replace('\n','')
		self.functions = np.loadtxt(func_dict[self.kernel], delimiter=',', skiprows=1, dtype = str)
		self.fitness_type = fitt_dict[self.kernel]
		
		
		### 2) from the dataset, prepare terminals, TRAINING, and TEST data ###
		if len(data_x) < 11: # for small datasets we will not split them into TRAINING and TEST components
			data_train = np.c_[data_x, data_y]
			data_test = np.c_[data_x, data_y]
			
		else: # if larger than 10, we run the data through the SciKit Learn random split
			x_train, x_test, y_train, y_test = skcv.train_test_split(data_x, data_y, test_size = 0.2)
			data_x, data_y = [], [] # clear from memory
			
			data_train = np.c_[x_train, y_train] # recombine the features with the solutions
			x_train, y_train = [], [] # clear from memory
			
			data_test = np.c_[x_test, y_test] # recombine the features with the solutions
			x_test, y_test = [], [] # clear from memory
			
		self.data_train_cols = len(data_train[0,:])
		self.data_train_rows = len(data_train[:,0])
		self.data_test_cols = len(data_test[0,:])
		self.data_test_rows = len(data_test[:,0])
		
		
		### 3) copy TRAINING data into an array (rows) of dictionaries (columns) ###
		data_train_dict = {}
		self.data_train_dict_array = np.array([])
		
		for row in range(0, self.data_train_rows): # increment through each row of data
			for col in range(0, self.data_train_cols): # increment through each column
				data_train_dict.update( {self.terminals[col]:data_train[row,col]} ) # to be unpacked in 'fx_fitness_eval'
				
			self.data_train_dict_array = np.append(self.data_train_dict_array, data_train_dict.copy())
		
		data_train = [] # clear from memory
			
		
		### 4) copy TEST data into an array (rows) of dictionaries (columns) ###
		data_test_dict = {}
		self.data_test_dict_array = np.array([])
		
		for row in range(0, self.data_test_rows): # increment through each row of data
			for col in range(0, self.data_test_cols): # increment through each column
				data_test_dict.update( {self.terminals[col]:data_test[row,col]} ) # to be unpacked in 'fx_fitness_eval'
				
			self.data_test_dict_array = np.append(self.data_test_dict_array, data_test_dict.copy())
		
		data_test = [] # clear from memory
			
		
		### 5) initialise all .csv files ###
		self.filename = {} # a dictionary to hold .csv filenames
		
		self.filename.update( {'a':'files/population_a.csv'} )
		target = open(self.filename['a'], 'w') # initialise the .csv file for population 'a' (foundation)
		target.close()
		
		self.filename.update( {'b':'files/population_b.csv'} )
		target = open(self.filename['b'], 'w') # initialise the .csv file for population 'b' (evolving)
		target.close()
		
		self.filename.update( {'f':'files/population_f.csv'} )
		target = open(self.filename['f'], 'w') # initialise the .csv file for the final population (used to test)
		target.close()
		
		self.filename.update( {'s':'files/population_s.csv'} )
		# do NOT initialise this .csv file, as it is retained for loading a previous run
		
		return
		
	
	def fx_karoo_data_recover(self, population):
	
		'''
		[need to write]
		'''
		
		with open(population, 'rb') as csv_file:
			target = csv.reader(csv_file, delimiter=',')
			n = 0 # track row count
			
			for row in target:
			
				n = n + 1
				if n == 1: pass # skip first empty row
				
				elif n == 2:
					self.population_a = [row] # write header to population_a
					
				else:
					if row == []:
						self.tree = np.array([[]]) # initialise tree array
						
					else:
						if self.tree.shape[1] == 0:
							self.tree = np.append(self.tree, [row], axis = 1) # append first row to Tree
							
						else:
							self.tree = np.append(self.tree, [row], axis = 0) # append subsequent rows to Tree
							
					if self.tree.shape[0] == 13:
						self.population_a.append(self.tree) # append complete Tree to population list
						
		print self.population_a
		
		return
		
	
	def fx_karoo_construct(self, tree_type, tree_depth_max):
		
		'''
		As used by the method 'fx_karoo_gp', this method constructs the initial population based upon the user-defined 
		Tree type and quantity. As "ramped half/half" is an industry standard, it was hard-coded into this method. But 
		the ratio of Full to Grow Trees may be easily modified, below.
		'''
		
		if self.display == 'i' or self.display == 'g':
			print '\n\t Type \033[1m?\033[0;0m at any (pause) to review your options, or \033[1mENTER\033[0;0m to continue.\033[0;0m'
			self.fx_karoo_pause(0)
			
		if self.display == 's':
			print '\n\t Type \033[1m?\033[0;0m to configure Karoo GP before your run, or \033[1mENTER\033[0;0m to continue.\033[0;0m'
			self.fx_karoo_pause(0)
			
		if tree_type == 'r': # Ramped 50/50
			for TREE_ID in range(1, int(self.tree_pop_max / 2) + 1):
				self.fx_gen_tree_build(TREE_ID, 'f', tree_depth_max) # build 1/2 of the 1st generation of Trees as Full
				self.fx_tree_append(self.tree) # append each Tree in the first generation to the list 'gp.population_a'
				
			for TREE_ID in range(int(self.tree_pop_max / 2) + 1, self.tree_pop_max + 1):
				self.fx_gen_tree_build(TREE_ID, 'g', tree_depth_max) # build 2/2 of the 1st generation of Trees as Grow
				self.fx_tree_append(self.tree)
				
		else: # Full or Grow
			for TREE_ID in range(1, self.tree_pop_max + 1):
				self.fx_gen_tree_build(TREE_ID, tree_type, tree_depth_max) # build the 1st generation of Trees
				self.fx_tree_append(self.tree)
				
		return
		
	
	def fx_karoo_reproduce(self):
	
		'''
		Through tournament selection, a single tree from the prior generation is copied without mutation to the next 
		generation. This is analogous to a member of the prior generation directly entering the gene pool of the 
		subsequent (younger) generation.
		'''
		
		if self.display != 's':
			if self.display == 'i': print ''
			print '  Perform', self.evolve_repro, 'Reproductions ...'
			if self.display == 'i': self.fx_karoo_pause(0)
			
		for n in range(self.evolve_repro): # quantity of Trees to be copied without mutation
			tourn_winner = self.fx_fitness_tournament(self.tourn_size) # perform tournament selection for each reproduction
			tourn_winner = self.fx_evo_fitness_wipe(tourn_winner) # remove fitness data
			self.population_b.append(tourn_winner) # append array to next generation population of Trees
			if self.display == 'i': print '\t\033[36m has been copied to the next Generation\033[0;0m'
			
		return
		
	
	def fx_karoo_point_mutate(self):
	
		'''
		Through tournament selection, a copy of a tree from the prior generation mutates before being added to the 
		next generation. In the biological world, this may be analogous to asexual reproduction, that is, a copy of 
		an individual with a minor mutation. In this method, a single point is selected for mutation while maintaining 
		function nodes as functions (operands) and terminal nodes as terminals (variables). The size and shape of the 
		tree will remain identical.
		'''
		
		if self.display != 's':
			if self.display == 'i': print ''
			print '  Perform', self.evolve_point, 'Point Mutations ...'
			if self.display == 'i': self.fx_karoo_pause(0)
			
		for n in range(self.evolve_point): # quantity of Trees to be generated through mutation
			tourn_winner = self.fx_fitness_tournament(self.tourn_size) # perform tournament selection for each mutation
			tourn_winner, node = self.fx_evo_point_mutate(tourn_winner) # perform point mutation; return single point for record keeping
			self.population_b.append(tourn_winner) # append array to next generation population of Trees
			
		return
		
	
	def fx_karoo_branch_mutate(self):
	
		'''
		Through tournament selection, a copy of a tree from the prior generation mutates before being added to the 
		next generation. In the biological world, this may be analogous to asexual reproduction, that is, a copy of an 
		individual but with a potentially substantial mutation. Unlike Point Mutation, in this method an entire branch 
		is selected. If the evolutionary run is designated as Full, the size and shape of the tree will remain 
		identical, each node mutated sequentially, where functions remain functions and terminals remain terminals. If 
		the evolutionary run is designated as Grow or Ramped Half/Half, the size and shape of the tree may grow 
		smaller or larger, but it may not exceed the maximum depth defined by the user.
		'''
		
		if self.display != 's':
			if self.display == 'i': print ''
			print '  Perform', self.evolve_branch, 'Full or Grow Mutations ...'
			if self.display == 'i': self.fx_karoo_pause(0)
			
		for n in range(self.evolve_branch): # quantity of Trees to be generated through mutation
			tourn_winner = self.fx_fitness_tournament(self.tourn_size) # perform tournament selection for each mutation
			branch = self.fx_evo_branch_select(tourn_winner) # select point of mutation and all nodes beneath
			
			if tourn_winner[1][1] == 'f': # perform Full method mutation on 'tourn_winner'
				tourn_winner = self.fx_evo_full_mutate(tourn_winner, branch)
				
			elif tourn_winner[1][1] == 'g': # perform Grow method mutation on 'tourn_winner'
				tourn_winner = self.fx_evo_grow_mutate(tourn_winner, branch)
				
			self.population_b.append(tourn_winner) # append array to next generation population of Trees
			
		return
		
	
	def fx_karoo_crossover_reproduce(self):
	
		'''
		Through tournament selection, two trees are selected as parents to produce a single offspring (future 
		versions of Karoo GP will produce 2 offspring per set of parent trees). Within each parent tree a branch is
		selected. Parent A is copied, with its selected branch deleted. Parent B's branch is then copied to the former 
		location of Parent A's branch, and inserted (grafted). The size and shape of the child tree may be smaller or 
		larger than either of the parents, but may not exceed the maximum depth defined by the user.
		
		This process combines genetic code from two trees, both of which were chosen by the tournament process as 
		having a higher fitness than the average population. Therefore, there is a chance their offspring will provide 
		an improvement in total fitness.
		
		In most GP applications, Crossover Reproduction is the most commonly applied evolutionary operator (60-70%).
		
		For those who like to watch, select 'db' (debug mode) at the launch of Karoo GP or at any (pause).
		'''
		
		if self.display != 's':
			if self.display == 'i': print ''
			print '  Perform', self.evolve_cross, 'Crossover Reproductions ...'
			if self.display == 'i': self.fx_karoo_pause(0)
			
		for n in range(self.evolve_cross): # quantity of Trees to be generated through crossover
		
			parent_a = self.fx_fitness_tournament(self.tourn_size) # perform tournament selection for 'parent_a'
			branch_a = self.fx_evo_branch_select(parent_a) # select branch within 'parent_a' to crossover to 'parent_b'
			
			parent_b = self.fx_fitness_tournament(self.tourn_size) # perform tournament selection for 'parent_b'
			branch_b = self.fx_evo_branch_select(parent_b) # select branch within 'parent_b' to receive crossover
			
			# child_1 = self.fx_evo_crossover(parent_a, branch_a, parent_b, branch_b) # perform Crossover Reproduction
			# self.population_b.append(child_1) # append the child of Crossover Reproduction to next generation population of Trees
			
			child_2 = self.fx_evo_crossover(parent_b, branch_b, parent_a, branch_a) # perform Crossover Reproduction
			self.population_b.append(child_2) # append the child of Crossover Reproduction to next generation population of Trees
			
		return
		
	
	def fx_karoo_pause(self, eol):
	
		'''
		Pause the visual output to screen until the user selects an option, as outlined below.
		'''
				
		pause = raw_input('\n\t\033[36m (pause) \033[0;0m')
		
		if pause == '?' or pause == 'help':
			print '\n\t\033[36mSelect one of the following options:\033[0;0m'
			
			print '\t\033[36m\033[1m i\033[0;0m\t interactive display mode'
			print '\t\033[36m\033[1m m\033[0;0m\t minimal display mode'
			print '\t\033[36m\033[1m g\033[0;0m\t generation display mode'
			print '\t\033[36m\033[1m s\033[0;0m\t silent (server) display mode'
			print '\t\033[36m\033[1m db\033[0;0m\t debug display mode'
			print '\t\033[36m\033[1m t\033[0;0m\t timer display mode'
			print ''
			print '\t\033[36m\033[1m ts\033[0;0m\t adjust the tournament size'
			print '\t\033[36m\033[1m min\033[0;0m\t adjust the minimum number of nodes (minimum boundary)'
			# print '\t\033[36m\033[1m max\033[0;0m\t adjust the maximum tree depth (maximum boundary)'
			print '\t\033[36m\033[1m b\033[0;0m\t adjust the balance of genetic operators (sum to 100%)'
			print '\t\033[36m\033[1m c\033[0;0m\t adjust the number of engaged CPU cores'
			print ''
			print '\t\033[36m\033[1m id\033[0;0m\t display the generation ID'
			print '\t\033[36m\033[1m l\033[0;0m\t list all Trees with the best fitness score'
			print '\t\033[36m\033[1m p\033[0;0m\t print a Tree to screen'
			print ''
			print '\t\033[36m\033[1m a\033[0;0m\t evaluate a Tree for Accuracy (TRAINING)'
			print '\t\033[36m\033[1m test\033[0;0m\t evaluate a Tree for Precision & Recall (TEST)'
			print ''
			print '\t\033[36m\033[1m cont\033[0;0m\t continue evolution, starting with the current population'
			print '\t\033[36m\033[1m load\033[0;0m\t load population_s to replace population_a'
			print '\t\033[36m\033[1m w\033[0;0m\t write the evolving population_b to disk'
			print '\t\033[36m\033[1m q\033[0;0m\t quit Karoo GP without saving population_b'
			print ''
			
			if eol == 0: print '\t\033[36m\033[1m ENTER\033[0;0m to continue ...'
			
			self.fx_karoo_pause(0)
			
		elif pause == 'i': self.display = 'i'; print '\t Interactive mode engaged (for control freaks)'; self.fx_karoo_pause(0)
		elif pause == 'm': self.display = 'm'; print '\t Minimal mode engaged (for recovering control freaks)'; self.fx_karoo_pause(0)
		elif pause == 'g': self.display = 'g'; print '\t Generation mode engaged (for GP gurus)'; self.fx_karoo_pause(0)
		elif pause == 's': self.display = 's'; print '\t Server mode engaged (for zen masters)'; self.fx_karoo_pause(0)
		elif pause == 'db': self.display = 'db'; print '\t Debug mode engaged (for vouyers)'; self.fx_karoo_pause(0)
		elif pause == 't': self.display = 't'; print '\t Timer mode engaged (for managers)'; self.fx_karoo_pause(0)			
		
		elif pause == 'ts': # adjust the tournament size
			n = range(1, self.tree_pop_max + 1) # set to total population size only for the sake of experimentation
			while True:
				try:
					print '\n\t The current tournament size is:', self.tourn_size
					query = int(raw_input('\t Adjust the tournament size (suggested 10): '))
					if query not in (n): raise ValueError()
					self.tourn_size = query; break
				except ValueError: print '\n\t\033[32m Enter a number from 1 including', str(self.tree_pop_max) + ".", 'Try again ...\033[0;0m'
			self.fx_karoo_pause(0)
				
		elif pause == 'min': # adjust the minimum boundary condition
			n = range(3, 1001) # we must have at least 3 nodes, as in: x * y; 1000 is an arbitrary number
			while True:
				try:
					print '\n\t The current minimum number of nodes is:', self.tree_depth_min
					query = int(raw_input('\t Adjust the minimum number of nodes for any given Tree (min 3): '))
					if query not in (n): raise ValueError()					
					self.tree_depth_min = query; break
				except ValueError: print '\n\t\033[32m Enter a number from 3 including 1000. Try again ...\033[0;0m'
			self.fx_karoo_pause(0)
			
		### cannot be made live until a new global variable is set for tree_depth_max and then fully tested ###
		
		# elif pause == 'max': # adjust the maximum boundary condition
			# n = range(3, 11) # we must have a depth of at least 3; a depth 10 carries up to 2047 nodes
			# while True:
				# try:
					# print '\n\t The current maximum Tree depth is:', tree_depth_max
					# query = int(raw_input('\n\t Adjust the maximum Tree depth (min 3, max 10): '))
					# if query not in (n): raise ValueError()					
					# tree_depth_max = query; break
				# except ValueError: print '\n\t\033[32m Enter a number from 3 including 10. Try again ...\033[0;0m'; self.fx_karoo_pause(0)
			
		elif pause == 'b': # adjust the balance of genetic operators'
			n = range(0, 101) # 0 to 100% expresssed as an integer
			
			print '\n\t The current balance of genetic operators is:'
			print '\t\t Reproduction:', self.evolve_repro
			print '\t\t Point Mutation:', self.evolve_point
			print '\t\t Branch Mutation:', self.evolve_branch
			print '\t\t Cross Over Reproduction:', self.evolve_cross, '\n'

			while True:
				try:
					query = int(raw_input('\t Enter percentage (0-100) for Reproduction: '))
					if query not in (n): raise ValueError()
					self.evolve_repro = int(float(query) / 100 * self.tree_pop_max); break
				except ValueError: print '\n\t\033[32m Enter a number from 0 including 100. Try again ...\033[0;0m'
				
			while True:
				try:
					query = int(raw_input('\t Enter percentage (0-100) for Point Mutation: '))
					if query not in (n): raise ValueError()
					self.evolve_point = int(float(query) / 100 * self.tree_pop_max); break
				except ValueError: print '\n\t\033[32m Enter a number from 0 including 100. Try again ...\033[0;0m'
				
			while True:
				try:
					query = int(raw_input('\t Enter percentage (0-100) for Branch Mutation: '))
					if query not in (n): raise ValueError()
					self.evolve_branch = int(float(query) / 100 * self.tree_pop_max); break
				except ValueError: print '\n\t\033[32m Enter a number from 0 including 100. Try again ...\033[0;0m'
				
			while True:
				try:
					query = int(raw_input('\t Enter percentage (0-100) for Cross Over Reproduction: '))
					if query not in (n): raise ValueError()
					self.evolve_cross = int(float(query) / 100 * self.tree_pop_max); break
				except ValueError: print '\n\t\033[32m Enter a number from 0 including 100. Try again ...\033[0;0m'
				
			print '\n\t The revised balance of genetic operators is:'
			print '\t\t Reproduction:', self.evolve_repro
			print '\t\t Point Mutation:', self.evolve_point
			print '\t\t Branch Mutation:', self.evolve_branch
			print '\t\t Cross Over Reproduction:', self.evolve_cross
			
			if self.evolve_repro + self.evolve_point + self.evolve_branch + self.evolve_cross != 100:
				print '\n\t The sum of the above does not equal 100%. Try again ...'
			self.fx_karoo_pause(0)
			
		elif pause == 'c': # adjust the number of engaged CPU cores
			n = range(1, self.core_count + 1) # assuming any run above 24 cores will simply use the maximum number
			while True:
				try:
					print '\n\t The current number of engaged CPU cores is:', self.cores
					query = int(raw_input('\n\t Adjust the number of CPU cores (min 1): '))
					if query not in (n): raise ValueError()
					self.cores = int(query); break
				except ValueError: print '\n\t\033[32m Enter a number from 1 including', str(self.core_count) + ".", 'Try again ...\033[0;0m'
			self.fx_karoo_pause(0)
			
		elif pause == 'id': print '\n\t The current generation is:', self.generation_id; self.fx_karoo_pause(0)
			
		elif pause == 'l': # display dictionary of Trees with the best fitness score
			print '\n\t The leading Trees and their associated expressions are:'
			for n in range(len(self.fittest_dict)):
				print '\t  ', self.fittest_dict.keys()[n], ':', self.fittest_dict.values()[n]
			self.fx_karoo_pause(0)
			
		elif pause == 'p': # display a Tree; need to add a SymPy graphical print option
			n = range(1, self.tree_pop_max + 1)
			while True:
				try:
					query = raw_input('\n\t Select a Tree to print ([ENTER] to exit): ')
					if query not in str(n) and query not in '': raise ValueError()
					elif query == '0' or query == '': break
					
					if len(self.population_a) > 1: self.fx_eval_tree_print(self.population_a[int(query)]); break
					else: break
				except ValueError: print '\n\t\033[32m Enter a number from 1 including', str(self.tree_pop_max) + ".", 'Try again ...\033[0;0m'
			self.fx_karoo_pause(0)
			
		elif pause == 'a': # evaluate Accuracy against the TRAINING data
			n = range(1, self.tree_pop_max + 1)
			while True:
				try:
					query = raw_input('\n\t Select a Tree for Accuracy in TRAINING ([ENTER] to exit): ')
					if query not in str(n) and query not in '': raise ValueError()
					elif query == '0' or query == '': break
					
					if len(self.population_a) > 1: self.fx_eval_accuracy(int(query)); break
					else: break
				except ValueError: print '\n\t\033[32m Enter a number from 1 including', str(self.tree_pop_max) + ".", 'Try again ...\033[0;0m'
			self.fx_karoo_pause(0)
			
		elif pause == 'test': # evaluate a Tree against the TEST data
			n = range(1, self.tree_pop_max + 1)
			while True:
				try:
					query = raw_input('\n\t Select a Tree for Precision & Recall in TEST ([ENTER] to exit): ')
					if query not in str(n) and query not in '': raise ValueError()
					elif query == '0' or query == '': break
					
					if len(self.population_a) > 1:
						if self.kernel == 'a': self.fx_test_abs(int(query)); break
						elif self.kernel == 'm': self.fx_test_match(int(query)); break
						elif self.kernel == 'c': self.fx_test_classify(int(query)); break
					else: break
				except ValueError: print '\n\t\033[32m Enter a number from 1 including', str(self.tree_pop_max) + ".", 'Try again ...\033[0;0m'
			self.fx_karoo_pause(0)
						
		elif pause == 'cont': # continue evolution, starting with the current population
			n = range(1, 101)
			while True:
				try:
					query = raw_input('\n\t How many more generations would you like to add? (1-100): ')
					if query not in str(n) and query not in '': raise ValueError()
					elif query == '0' or query == '': break
					
					self.generation_max = self.generation_max + int(query)
					next_gen_start = self.generation_id + 1
					self.fx_karoo_continue(next_gen_start) # continue evolving, starting with the last population
				except ValueError: print '\n\t\033[32m Enter a number from 1 including 100. Try again ...\033[0;0m'
			self.fx_karoo_pause(0)
			
		elif pause == 'load': # load population_s to replace population_a
			while True:
				try:
					query = raw_input('\n\t Overwrite the current population with population_s? ')
					if query not in ['y','n']: raise ValueError()
					
					if query == 'y': self.fx_karoo_data_recover(self.filename['s']); break
					elif query == 'n': break
				except ValueError: print '\n\t\033[32m Enter (y)es or (n)o. Try again ...\033[0;0m'
			self.fx_karoo_pause(0)
			
		elif pause == 'w': # write the evolving population_b to disk
			if self.generation_id > 1:
				self.fx_tree_archive(self.population_b, 'b')
				print '\t All current members of the evolving population_b saved to .csv'; self.fx_karoo_pause(0)
				
			else: print '\t The evolving population_b does not yet exist'; self.fx_karoo_pause(0)
			
		elif pause == 'q': # quit the script without saving the evolving population_b
			sys.exit()
			
		elif pause != '': self.fx_karoo_pause(0)
		
		if eol == 1: self.fx_karoo_pause(1) # catch all other entries so as to not accidentally exit
		
		return
		
	
	def fx_karoo_continue(self, next_gen_start):
	
		'''
		This method enables the launch of another full run of Karoo GP, but starting with a seed generation
		instead of with a randomly generated first population.
		'''
		
		for self.generation_id in range(next_gen_start, self.generation_max + 1): # evolve additional generations of Trees
		
			print '\n Evolve a population of Trees for Generation', self.generation_id, '...'
			self.population_b = ['Karoo GP by Kai Staats, Evolving Generation'] # initialise population_b to host the next generation
			
			self.fx_fitness_gene_pool() # generate the viable gene pool (compares against gp.tree_depth_min)
			self.fx_karoo_reproduce() # method 1 - Reproduction
			self.fx_karoo_point_mutate() # method 2 - Point Mutation
			self.fx_karoo_branch_mutate() # method 3 - Branch Mutation
			self.fx_karoo_crossover_reproduce() # method 4 - Crossover Reproduction
			self.fx_eval_generation() # evaluate all Trees in a single generation
			
			self.population_a = self.fx_evo_pop_copy(self.population_b, ['GP Tree by Kai Staats, Generation ' + str(self.generation_id)])
			
		# "End of line, man!" --CLU
		target = open(self.filename['f'], 'w') # reset the .csv file for the final population
		target.close()
		
		self.fx_tree_archive(self.population_b, 'f') # save the final generation of Trees to disk
		self.fx_karoo_eol()
		
		return
		
	
	def fx_karoo_eol(self):
		
		'''
		The very last method to run in Karoo GP.
		'''
		
		print '\n\033[3m "It is not the strongest of the species that survive, nor the most intelligent,\033[0;0m'
		print '\033[3m  but the one most responsive to change."\033[0;0m --Charles Darwin'
		print ''
		print '\033[3m Congrats!\033[0;0m Your multi-generational Karoo GP run is complete.\n'
		print '\033[36m Type \033[1m?\033[0;0m\033[36m to review your options or \033[1mENTER\033[0;0m\033[36m to exit.\033[0;0m\n'
		self.fx_karoo_pause(1)
		
		return
		
	
	#++++++++++++++++++++++++++++++++++++++++++
	#   Methods to Generate a Tree            |
	#++++++++++++++++++++++++++++++++++++++++++
	
	def fx_gen_tree_initialise(self, TREE_ID, tree_type, tree_depth_max):

		'''
		Assign 13 global variables to the array 'tree'.
		
		Build the array 'tree' with 13 rows and initally, just 1 column of labels. This array will
		grow as each new node is appended.
		
		The values of this array are stored as string characters. Numbers will be forced to integers
		at the point of execution.
		
		Requires 'TREE_ID', 'tree_type', and 'tree_depth_max'
		'''
		
		self.pop_TREE_ID = TREE_ID 					# pos 0: unique identifier for each tree
		self.pop_tree_type = tree_type				# pos 1: defined in 'User Input' as (f)ull, (g)row, or (r)amped 50/50
		self.pop_tree_depth_max = tree_depth_max 	# pos 2: defined in 'User Input' as the maximum allowable depth of any given tree
		self.pop_NODE_ID = 1 						# pos 3: unique identifier for each node; this is the INDEX KEY to this array
		self.pop_node_depth = 0 					# pos 4: depth of node when committed to the array
		self.pop_node_type = '' 					# pos 5: root, function, or terminal
		self.pop_node_label = '' 					# pos 6: operand [+, -, *, ...] or terminal [a, b, c, ...]
		self.pop_node_parent = '' 					# pos 7: parent node
		self.pop_node_arity = '' 					# pos 8: number of nodes attached to each non-terminal node
		self.pop_node_c1 = '' 						# pos 9: child node 1
		self.pop_node_c2 = '' 						# pos 10: child node 2
		self.pop_node_c3 = '' 						# pos 11: child node 3 (assumed max of 3 with boolean operator 'if')
		self.pop_fitness = ''						# pos 12: fitness value following Tree evaluation
		
		self.tree = np.array([ ['TREE_ID'],['tree_type'],['tree_depth_max'],['NODE_ID'],['node_depth'],['node_type'],['node_label'],['node_parent'],['node_arity'],['node_c1'],['node_c2'],['node_c3'],['fitness'] ])
		
		return
				
	
	### Root Node ###
	
	def fx_gen_root_node_build(self):
	
		'''
		Build the Root node
		'''
				
		self.fx_gen_function_select() # select the operand for root
		
		if self.pop_node_arity == 1: # 1 child
			self.pop_node_c1 = 2
			self.pop_node_c2 = ''
			self.pop_node_c3 = ''
			
		elif self.pop_node_arity == 2: # 2 children
			self.pop_node_c1 = 2
			self.pop_node_c2 = 3
			self.pop_node_c3 = ''
			
		elif self.pop_node_arity == 3: # 3 children
			self.pop_node_c1 = 2
			self.pop_node_c2 = 3
			self.pop_node_c3 = 4

		self.pop_node_type = 'root'
			
		self.fx_gen_node_commit()

		return
		
	
	### Function Nodes ###
	
	def fx_gen_function_node_build(self):
	
		'''
		Build the Function nodes
		'''
		
		for i in range(1, self.pop_tree_depth_max): # increment depth, from 1 through 1 shy of 'tree_depth_max'
		
			self.pop_node_depth = i # increment 'node_depth'
			
			parent_arity_sum = 0
			prior_sibling_arity = 0 # reset for 'c_buffer' in 'children_link'
			prior_siblings = 0 # reset for 'c_buffer' in 'children_link'
			
			for j in range(1, len(self.tree[3])): # increment through all nodes (exclude 0) in array 'tree'
			
				if int(self.tree[4][j]) == self.pop_node_depth-1: # find parent nodes which reside at the prior depth
					parent_arity_sum = parent_arity_sum + int(self.tree[8][j]) # sum arities of all parent nodes at the prior depth
					
					# (do *not* merge these 2 "j" loops or it gets all kinds of messed up)
										
			for j in range(1, len(self.tree[3])): # increment through all nodes (exclude 0) in array 'tree'
			
				if int(self.tree[4][j]) == self.pop_node_depth-1: # find parent nodes which reside at the prior depth

					for k in range(1, int(self.tree[8][j]) + 1): # increment through each degree of arity for each parent node
						self.pop_node_parent = int(self.tree[3][j]) # set the parent 'NODE_ID' ...
						prior_sibling_arity = self.fx_gen_function_gen(parent_arity_sum, prior_sibling_arity, prior_siblings) # ... generate a Function ndoe
						prior_siblings = prior_siblings + 1 # sum sibling nodes (current depth) who will spawn their own children (cousins? :)
												
		return
		
	
	def fx_gen_function_gen(self, parent_arity_sum, prior_sibling_arity, prior_siblings):
	
		'''
		Generate a single Function node
		'''
		
		if self.pop_tree_type == 'f': # user defined as (f)ull
			self.fx_gen_function_select() # retrieve a function
			self.fx_gen_child_link(parent_arity_sum, prior_sibling_arity, prior_siblings) # establish links to children
			
		elif self.pop_tree_type == 'g': # user defined as (g)row
			rnd = np.random.randint(2)
			
			if rnd == 0: # randomly selected as Function
				self.fx_gen_function_select() # retrieve a function
				self.fx_gen_child_link(parent_arity_sum, prior_sibling_arity, prior_siblings) # establish links to children
				
			elif rnd == 1: # randomly selected as Terminal
				self.fx_gen_terminal_select() # retrieve a terminal
				self.pop_node_c1 = ''
				self.pop_node_c2 = ''
				self.pop_node_c3 = ''
				
		self.fx_gen_node_commit() # commit new node to array
		prior_sibling_arity = prior_sibling_arity + self.pop_node_arity # sum the arity of prior siblings
		
		return prior_sibling_arity
		
	
	def fx_gen_function_select(self):
	
		'''
		Define a single Function (operand extracted from the associated functions.csv)
		'''
		
		self.pop_node_type = 'func'
		rnd = np.random.randint(0, len(self.functions[:,0])) # call the previously loaded .csv which contains all operands
		self.pop_node_label = self.functions[rnd][0]
		self.pop_node_arity = int(self.functions[rnd][1])
		
		return
		
	
	### Terminal Nodes ###
	
	def fx_gen_terminal_node_build(self):
	
		'''
		Build the Terminal nodes
		'''
			
		self.pop_node_depth = self.pop_tree_depth_max # set the final node_depth (same as 'pop_node_depth' + 1)
		
		for j in range(1, len(self.tree[3]) ): # increment through all nodes (exclude 0) in array 'tree'
		
			if int(self.tree[4][j]) == self.pop_node_depth-1: # find parent nodes which reside at the prior depth
			
				for k in range(1,(int(self.tree[8][j]) + 1)): # increment through each degree of arity for each parent node
					self.pop_node_parent = int(self.tree[3][j]) # set the parent 'NODE_ID'  ...
					self.fx_gen_terminal_gen() # ... generate a Terminal node
					
		return
		
	
	def fx_gen_terminal_gen(self):
	
		'''
		Generate a single Terminal node
		'''
		
		self.fx_gen_terminal_select() # retrieve a terminal
		self.pop_node_c1 = ''
		self.pop_node_c2 = ''
		self.pop_node_c3 = ''
	
		self.fx_gen_node_commit() # commit new node to array
	
		return
		
	
	def fx_gen_terminal_select(self):
	
		'''
		Define a single Terminal (variable extracted from the top row of the associated TRAINING data)
		'''
				
		self.pop_node_type = 'term'
		rnd = np.random.randint(0, len(self.terminals) - 1) # call the previously loaded .csv which contains all terminals
		self.pop_node_label = self.terminals[rnd]
		self.pop_node_arity = 0
		
		return
		
	
	### The Lovely Children ###
			
	def fx_gen_child_link(self, parent_arity_sum, prior_sibling_arity, prior_siblings):
	
		'''
		Link each parent node to its children
		'''
		
		c_buffer = 0
		
		for n in range(1, len(self.tree[3]) ): # increment through all nodes (exclude 0) in array 'tree'
		
			if int(self.tree[4][n]) == self.pop_node_depth-1: # find all nodes that reside at the prior (parent) 'node_depth'
			
				c_buffer = self.pop_NODE_ID + (parent_arity_sum + prior_sibling_arity - prior_siblings) # One algo to rule the world!
				
				if self.pop_node_arity == 0: # terminal in a Grow Tree
					self.pop_node_c1 = ''
					self.pop_node_c2 = ''
					self.pop_node_c3 = ''
					
				elif self.pop_node_arity == 1: # 1 child
					self.pop_node_c1 = c_buffer
					self.pop_node_c2 = ''
					self.pop_node_c3 = ''
					
				elif self.pop_node_arity == 2: # 2 children
					self.pop_node_c1 = c_buffer
					self.pop_node_c2 = c_buffer + 1
					self.pop_node_c3 = ''
					
				elif self.pop_node_arity == 3: # 3 children
					self.pop_node_c1 = c_buffer
					self.pop_node_c2 = c_buffer + 1
					self.pop_node_c3 = c_buffer + 2
					
				else:
					print '\n\t\033[31mERROR! Something has gone very wrong with the children!\033[0;0m'
					print '\t pop_node_arity =', self.pop_node_arity; self.fx_karoo_pause(0)
					
		return
		
	
	def fx_gen_node_commit(self):
	
		'''
		Commit the values of a new node (root, function, or terminal) to the array 'tree'
		'''
		
		self.tree = np.append(self.tree, [ [self.pop_TREE_ID],[self.pop_tree_type],[self.pop_tree_depth_max],[self.pop_NODE_ID],[self.pop_node_depth],[self.pop_node_type],[self.pop_node_label],[self.pop_node_parent],[self.pop_node_arity],[self.pop_node_c1],[self.pop_node_c2],[self.pop_node_c3],[self.pop_fitness] ], 1)
		
		self.pop_NODE_ID = self.pop_NODE_ID + 1
		
		return
		
	
	def fx_gen_tree_build(self, TREE_ID, tree_type, tree_depth_max):
	
		'''
		This method combines 4 sub-methods into a single method for ease of deployment. It is designed to executed 
		within a for loop such that an entire population is built. However, it may also be run from the command line, 
		passing a single TREE_ID to the method.
		
		'tree_type' is either (f)ull or (g)row. Note, however, that when the user selects 'ramped 50/50' at launch, it 
		is still (f) or (g) which are passed to this method.
		'''
		
		self.fx_gen_tree_initialise(TREE_ID, tree_type, tree_depth_max) # initialise a new Tree
		self.fx_gen_root_node_build() # build the Root node
		self.fx_gen_function_node_build() # build the Function nodes
		self.fx_gen_terminal_node_build() # build the Terminal nodes
		
		return # each Tree is written to 'gp.tree'
		
	
	#++++++++++++++++++++++++++++++++++++++++++
	#   Methods to Evaluate a Tree            |
	#++++++++++++++++++++++++++++++++++++++++++		
	
	def fx_eval_poly(self, tree):
	
		'''
		Generate the polynomial (both raw and sympified)
		'''
		
		self.algo_raw = self.fx_eval_label(tree, 1) # pass the root 'node_id', then flatten the tree to a string
		self.algo_sym = sp.sympify(self.algo_raw) # string converted to a functional polynomial (the coolest line in the script! :)
						
		return
		
	
	def fx_eval_label(self, tree, node_id):
	
		'''
		Evaluate all or part of a Tree and return a raw polynomial ('algo_raw').
		
		In the main code, this method is called once per Tree, but may be called at any time to prepare a polynomial 
		for any full or partial (branch) tree contained in 'population'.
		
		Pass the starting node for recursion via the local variable 'node_id' where the local variable 'tree' is a 
		copy of the Tree you desire to evaluate.
		'''
		
		if tree[8, node_id] == '0': # arity of 0 for the pattern '[term]'
			return '(' + tree[6, node_id] + ')' # 'node_label' (function or terminal)
			
		else:
			if tree[8, node_id] == '1': # arity of 1 for the pattern '[func] [term]'
				return self.fx_eval_label(tree, tree[9, node_id]) + tree[6, node_id]
				
			elif tree[8, node_id] == '2': # arity of 2 for the pattern '[func] [term] [func]'
				return self.fx_eval_label(tree, tree[9, node_id]) + tree[6, node_id] + self.fx_eval_label(tree, tree[10, node_id])
				
			elif tree[8, node_id] == '3': # arity of 3 for the explicit pattern 'if [term] then [term] else [term]'
				return tree[6, node_id] + self.fx_eval_label(tree, tree[9, node_id]) + ' then ' + self.fx_eval_label(tree, tree[10, node_id]) + ' else ' + self.fx_eval_label(tree, tree[11, node_id])
						
	
	def fx_eval_id(self, tree, node_id):
	
		'''
		Evaluate all or part of a Tree and return a list of all 'NODE_ID'.
	
		This method generates a list of all 'NODE_ID's from the given Node and below. It is used primarily to generate 
		'branch' for the multi-generatioal mutation of Trees.
	
		Pass the starting node for recursion via the local variable 'node_id' where the local variable 'tree' is a copy 
		of the Tree you desire to evaluate.
	
		'''
		
		if tree[8, node_id] == '0': # arity of 0 for the pattern '[NODE_ID]'
			return tree[3, node_id] # 'NODE_ID'
			
		else:
			if tree[8, node_id] == '1': # arity of 1 for the pattern '[NODE_ID], [NODE_ID]'
				return tree[3, node_id] + ', ' + self.fx_eval_id(tree, tree[9, node_id])
				
			elif tree[8, node_id] == '2': # arity of 2 for the pattern '[NODE_ID], [NODE_ID], [NODE_ID]'
				return tree[3, node_id] + ', ' + self.fx_eval_id(tree, tree[9, node_id]) + ', ' + self.fx_eval_id(tree, tree[10, node_id])
				
			elif tree[8, node_id] == '3': # arity of 3 for the pattern '[NODE_ID], [NODE_ID], [NODE_ID], [NODE_ID]'
				return tree[3, node_id] + ', ' + self.fx_eval_id(tree, tree[9, node_id]) + ', ' + self.fx_eval_id(tree, tree[10, node_id]) + ', ' + self.fx_eval_id(tree, tree[11, node_id])
				

	def fx_eval_tree_print(self, tree):

		'''
		Display all or part of a Tree on-screen.
		
		This method displays all sequential node_ids from 'start' node through bottom, within the given tree.
		'''
		start = 1 # can pass 'start' to this method, to print only a sub-section of the Tree
		
		ind = ''
		
		print '\n\033[1m\033[36m Tree ID', int(tree[0][1]), '\033[0;0m'
		
		for depth in range(int(tree[4][start]), int(tree[2][1]) + 1): # increment through all Tree depths
			print '\n', ind,'\033[36m Tree Depth:', depth, 'of', tree[2][1], '\033[0;0m'
			
			for node in range(start, len(tree[3])): # increment through all nodes (redundant, I know)
				if int(tree[4][node]) == depth:
					print ''
					print ind,'\033[1m\033[36m NODE:', tree[3][node], '\033[0;0m'
					print ind,'  type:', tree[5][node]
					print ind,'  label:', tree[6][node], '\tparent node:', tree[7][node]
					print ind,'  arity:', tree[8][node], '\tchild node(s):', tree[9][node], tree[10][node], tree[11][node]
					
			ind = ind + '\t'
			
		print ''
		self.fx_eval_poly(tree) # generate the raw and sympified equation for the entire Tree
		print '\t\033[36mTree', tree[0][1], 'yields (raw):', self.algo_raw, '\033[0;0m'
		print '\t\033[36mTree', tree[0][1], 'yields (sym):\033[1m', self.algo_sym, '\033[0;0m'
		
		return
		
	
	def fx_eval_branch_print(self, tree, start):
	
		'''
		Display a Tree branch on-screen.
		
		This method displays all sequential node_ids from 'start' node through bottom, within the given branch.
		'''
		
		branch = np.array([]) # the array is necessary in order to len(branch) when 'branch' has only one element
		branch_eval = self.fx_eval_id(tree, start) # generate tuple of given 'branch'		
		branch_symp = sp.sympify(branch_eval) # convert string from tuple to list
		branch = np.append(branch, branch_symp) # append list to array
		ind = ''
		
		for depth in range(int(tree[4][start]), int(tree[2][1]) + 1): # increment through all Tree depths
			print '\n', ind,'\033[36m Tree Depth:', depth, 'of', tree[2][1], '\033[0;0m'
			
			for n in range(0, len(branch)): # increment through all nodes listed in the branch
				node = branch[n]
				
				if int(tree[4][node]) == depth:
					print ''
					print ind,'\033[1m\033[36m NODE:', node, '\033[0;0m'
					print ind,'  type:', tree[5][node]
					print ind,'  label:', tree[6][node], '\tparent node:', tree[7][node]
					print ind,'  arity:', tree[8][node], '\tchild node(s):', tree[9][node], tree[10][node], tree[11][node]
					
			ind = ind + '\t'
					
		print ''
		self.fx_eval_poly(tree) # generate the raw and sympified equation for the entire Tree
		print '\t\033[36mTree', tree[0][1], 'yields (raw):', self.algo_raw, '\033[0;0m'
		print '\t\033[36mTree', tree[0][1], 'yields (sym):\033[1m', self.algo_sym, '\033[0;0m'
		
		return
		
	
	def fx_eval_accuracy(self, tree_id):
	
		'''
		Evaluate Accuracy of a single Tree during training.
		
		This method compares the stored, total fitness score for all rows of a single Tree to the total number of rows 
		in the associated dataset.
		
		For this method to provide meaningful output, the fitness function must be maximising and the desired solution 
		an exact Match. This method will not provide meaningful output for a minimisation (Absolute Diff) nor 
		Classification function.
		'''
		
		print '\n\t Tree', tree_id, 'has an accuracy of:', float(self.population_a[tree_id][12][1]) / self.data_train_dict_array.shape[0] * 100
		
		return
		
	
	def fx_eval_generation(self):
	
		'''
		Karoo GP evaluates each subsequent generation of Trees. This process flattens each GP Tree into a standard 
		equation by means of a recursive algorithm and subsequent processing by the Sympy library. Sympy simultaneously 
		evaluates the Tree for its results, returns null for divide by zero, reorganises and then rewrites the 
		expression in its simplest form.
		'''
		
		if self.display != 's':
			if self.display == 'i': print ''
			print '\n Evaluate all Trees in Generation', self.generation_id
			if self.display == 'i': self.fx_karoo_pause(0)
			
		self.fx_evo_tree_renum(self.population_b) # population renumber
		self.fx_fitness_gym(self.population_b) # run 'fx_eval', 'fx_fitness', 'fx_fitness_store', and fitness record
		self.fx_tree_archive(self.population_b, 'a') # archive the current, evolved generation of Trees
		
		if self.display != 's':
			print '\n Copy gp.population_b to gp.population_a\n'
			
		return
		
	
	#++++++++++++++++++++++++++++++++++++++++++
	#   Methods to Evaluate Tree Fitness      |
	#++++++++++++++++++++++++++++++++++++++++++
	
	def fx_fitness_gym(self, population):
	
		'''
		This method combines 3 methods into one: 'fx_eval', 'fx_fitness', 'fx_fitness_store', and then displays the 
		results to the user. It's a hard-core, all-out GP workout!
		
		Part 1 evaluates each polynomial against the data, line for line. This is the most time consuming and CPU 
		engaging of the entire Genetic Program.
		
		Part 2 evaluates every Tree in each generation to determine which have the best, overall fitness score. This 
		could be the highest or lowest depending upon if the fitness function is maximising (higher is better) or 
		minimising (lower is better). The total fitness score is then saved with each Tree in the external .csv file.
		
		Part 3 compares the fitness of each Tree to the prior best fit in order to track those that improve with each
		comparison. For matching functions, all the Trees will have the same fitness value, but they may present more 
		than one solution. For minimisation and maximisation functions, the final Tree should present the best overall 
		fitness for that generation. It is important to note that Part 3 does *not* in any way influence the Tournament 
		Selection which is a stand-alone process.
		'''
		
		fitness_best = 0
		self.fittest_dict = {}
		time_sum = 0
		
		for tree_id in range(1, len(population)):
		
			### PART 1 - EXTRACT POLYNOMIAL FROM EACH TREE ###
			self.fx_eval_poly(population[tree_id]) # extract the Polynomial
			if self.display not in ('s','t'): print '\t\033[36mTree', population[tree_id][0][1], 'yields (sym):\033[1m', self.algo_sym, '\033[0;0m'
			
			
			### PART 2 - EVALUATE TREE FITNESS AND STORE ###
			fitness = 0
			
			if self.display == 't': start = time.time() # start the clock for 'timer' mode
			
			if self.cores == 1: # employ only one CPU core and bypass 'pprocess' to avoid overhead
				for row in range(0, self.data_train_rows): # increment through all rows in the TRAINING data
					fitness = fitness + self.fx_fitness_eval(row) # evaluate Tree Fitness
					
			else: # employ multiple CPU cores using 'pprocess'
				results = pp.Map(limit = self.cores)
				parallel_function = results.manage(pp.MakeParallel(self.fx_fitness_eval))
				for row in range(0, self.data_train_rows): # increment through all rows in TRAINING data
					parallel_function(row) # evaluate Tree Fitness
					
				fitness = sum(results[:]) # 'pprocess' returns the fitness scores in a single dump
				
			
			if self.display == 'i':
				print '\t \033[36m with fitness sum:\033[1m', fitness, '\033[0;0m\n'
				
			if self.display == 't':
				print '\t \033[36m The fitness_gym has an ellapsed time of \033[0;0m\033[31m%f\033[0;0m' % (time.time() - start), '\033[36mon', self.cores, 'cores\033[0;0m'
				time_sum = time_sum + (time.time() - start)
				
			self.fx_fitness_store(population[tree_id], fitness) # store Fitness with each Tree
			
			
			### PART 3 - COMPARE TREE FITNESS FOR DISPLAY ###
			if self.kernel == 'a': # display best fit Trees for the ABSOLUTE DIFFERENCE kernel
				if fitness_best == 0: # first time through
					fitness_best = fitness
					
				if fitness <= fitness_best: # find the Tree with Minimum fitness score
					fitness_best = fitness # set best fitness score
					self.fittest_dict.update({tree_id:self.algo_sym}) # add to dictionary
					
			elif self.kernel == 'b': # display best fit Trees for the BOOLEAN kernel
				if fitness == self.data_train_rows: # find the Tree with a perfect match for all data rows
					fitness_best = fitness # set best fitness score
					self.fittest_dict.update({tree_id:self.algo_sym}) # add to dictionary
					
			elif self.kernel == 'c': # display best fit Trees for the CLASSIFY kernel
				if fitness >= fitness_best: # find the Tree with Maximum fitness score
					fitness_best = fitness # set best fitness score
					self.fittest_dict.update({tree_id:self.algo_sym}) # add to dictionary
					
			elif self.kernel == 'm': # display best fit Trees for the MATCH kernel
				if fitness == self.data_train_rows: # find the Tree with a perfect match for all data rows
					fitness_best = fitness # set best fitness score
					self.fittest_dict.update({tree_id:self.algo_sym}) # add to dictionary
					
			# elif self.kernel == '[other]': # display best fit Trees for the [OTHER] kernel
				# if fitness >= fitness_best: # find the Tree with [Maximum or Minimum] fitness score
					# fitness_best = fitness # set best fitness score
					# self.fittest_dict.update({tree_id:self.algo_sym}) # add to dictionary
														
		if self.display == 't': print '\n\t \033[36m The sum of all Tree timings is \033[0;0m\033[31m%f\033[0;0m' % time_sum, '\033[0;0m'; self.fx_karoo_pause(0)
		
		print '\n\033[36m ', len(self.fittest_dict.keys()), 'trees\033[1m', np.sort(self.fittest_dict.keys()), '\033[0;0m\033[36moffer the highest fitness scores.\033[0;0m'
		if self.display == 'g': self.fx_karoo_pause(0)
		
		return
		
	
	def fx_fitness_eval(self, row):
	
		'''
		Evaluate the fitness of the Tree.
		
		This method uses the 'sympified' (SymPy) polynomial ('algo_sym') created in 'fx_eval_poly' and the data set 
		loaded at run-time to evaluate the fitness of the selected kernel. The output is returned as the global 
		variable 'fitness'.
		
		[need to write more]
		'''
		
		# We need to extract the variables from the polynomial. However, these variables are no longer correlated
		# to the original variables listed across the top of each column of data.csv, so we must re-assign their 
		# respective values for each subsequent row in the data .csv, for each Tree's unique polynomial.
		
		data_train_dict = self.data_train_dict_array[row] # re-assign (unpack) a temp dictionary to each row of data
		
		if str(self.algo_sym.subs(data_train_dict)) == 'zoo': # divide by zero demands we avoid use of the 'float' function
			result = self.algo_sym.subs(data_train_dict) # print 'divide by zero', result; self.fx_karoo_pause(0)
			
		else:
			result = float(self.algo_sym.subs(data_train_dict)) # process the polynomial to produce the result
			result = round(result, self.precision) # force 'result' and 'solution' to the same number of floating points
			
		solution = float(data_train_dict['s']) # extract the desired solution from the data
		solution = round(solution, self.precision) # force 'result' and 'solution' to the same number of floating points
		
		# if str(self.algo_sym) == 'a + b/c': # a temp fishing net to catch a specific result
			# print 'algo_sym', self.algo_sym
			# print 'result', result, 'solution', solution
			# self.fx_karoo_pause(0)
			
		if self.kernel == 'a': # ABSOLUTE DIFFERENCE kernel
			fitness = self.fx_fitness_function_abs_diff(row, result, solution)
			
		elif self.kernel == 'b': # BOOLEAN kernel
			fitness = self.fx_fitness_function_bool(row, result, solution)
			
		elif self.kernel == 'c': # CLASSIFY kernel
			fitness = self.fx_fitness_function_classify(row, result, solution)
			
		elif self.kernel == 'm': # MATCH kernel
			fitness = self.fx_fitness_function_match(row, result, solution)
			
		# elif: # self.fx_kernel == '[other]': # place-holder for a new kernel
			# self.fx_kernel_[other](row, result, solution)
			
		return fitness
		
	
	def fx_fitness_function_abs_diff(self, row, result, solution): # the ABSOLUTE DIFFERENCE kernel
	
		'''
		A Symbolic Regression kernel used within the 'fitness_eval' function.
		
		This is a minimisation function which seeks a result which is closest to the solution.
		
		[result is close to solution]
		
		[need to write more]
		'''
		
		fitness = abs(result - solution) # this is a Minimisation function which seeks the smallest fitness
		
		if self.display == 'i': print '\t\033[36m data row', row, 'yields:', result, '\033[0;0m'
			
		return fitness
		
	
	def fx_fitness_function_bool(self, row, result, solution):
	
		'''
		A Boolean kernel used within the 'fitness_eval' function.
		
		This is a maximization function which seeks an exact solution (a perfect match).
		
		[need to write more]
		'''
		
		if result == solution:
			fitness = 1 # improve the fitness score by 1
			
			if self.display == 'i': print '\t\033[36m data row', row, '\033[0;0m\033[36myields:\033[1m', result, '\033[0;0m' # bold font face
			
		else:
			fitness = 0 # do not adjust the fitness score
			
			if self.display == 'i': print '\t\033[36m data row', row, 'yields:', result, '\033[0;0m' # standard font face
			
		return fitness
		
	
	def fx_fitness_function_classify(self, row, result, solution):
	
		'''
		This multiclass classifer compares each row of a given Tree to the known solution, comparing estimated values 
		(labels) generated by Karoo GP against the correct labels. This method is able to work with any number of class 
		labels, from 2 to n. The first label bin includes -inf. The last label bin includes +inf. Those in between are 
		by default confined to the spacing of 1.0 each, as defined by:
		
			(solution - 1) < result <= solution
			
		The skew adjusts the boundaries of the bins such that they fall on both the negative and positive sides of the 
		origin. At the time of this writing, an odd number of class labels will generate an extra bin on the positive 
		side of origin as it has not yet been determined the effect of enabling the middle bin to include both a 
		negative and positive space.
		
		Commented in the code is another 
		'''
		
		# tested 2015 10/18
		
		skew = (self.class_labels / 2) - 1 # '-1' keeps a binary classification splitting over the origin
		# skew = 0 # for code testing
		
		if solution == 0 and result <= 0 - skew: # check for first class
			fitness = 1
			if self.display == 'i': print '\t\033[36m data row', row, 'yields class label:\033[1m', int(solution), 'as', result, '<=', int(0 - skew), '\033[0;0m'
			
		elif solution == self.class_labels - 1 and result > (solution - 1) - skew: # check for last class
			fitness = 1
			if self.display == 'i': print '\t\033[36m data row', row, 'yields class label:\033[1m', int(solution), 'as', result, '>', int(solution - skew), '\033[0;0m'
			
		elif (solution - 1) - skew < result <= solution - skew: # check for class bins between first and last
			fitness = 1
			if self.display == 'i': print '\t\033[36m data row', row, 'yields class label:\033[1m', int(solution), 'as', int(solution - 1 - skew), '<', result, '<=', int(solution - skew), '\033[0;0m'
			
		else: # no class match
			fitness = 0
			if self.display == 'i': print '\t\033[36m data row', row, 'yields: no match \033[0;0m'
			
		return fitness
		
	
	def fx_fitness_function_match(self, row, result, solution):
	
		'''
		A Symbolic Regression kernel used within the 'fitness_eval' function.
		
		This is a maximization function which seeks an exact solution (a perfect match).
		
		result = solution
		
		[need to write more]
		'''
		
		if result == solution:
			fitness = 1 # improve the fitness score by 1
				
			if self.display == 'i': print '\t\033[36m data row', row, '\033[0;0m\033[36myields:\033[1m', result, '\033[0;0m' # bold font face
			
		else:
			fitness = 0 # do not adjust the fitness score
			
			if self.display == 'i': print '\t\033[36m data row', row, 'yields:', result, '\033[0;0m' # standard font face
			
		return fitness
		
		
	# def fx_fitness_function_[other](self, row, result, solution): # the [OTHER] kernel

		'''
		[other] kernel used within the 'fitness_eval' function
		
		This is a [minimisation or maximization] function which [insert description].
				
		[insert new fitness function kernel here]
		
		return fitness
		'''
		
	
	def fx_fitness_store(self, tree, fitness):
	
		fitness = float(fitness)
		fitness = round(fitness, self.precision)
		# print '\t\033[36m with fitness', fitness, '\033[0;0m'
		
		tree[12][1] = fitness # store the fitness with each tree
		# tree[12][2] = result # store the result of the executed polynomial
		# tree[12][3] = solution # store the desired solution
		
		if len(tree[3]) > 4: # if the Tree array is wide enough ...
			tree[12][4] = len(str(self.algo_raw)) # store the length of the SymPyfied algo (for Tournament selection)
			
		return
		
	
	def fx_fitness_tournament(self, tourn_size):
	
		'''
		Select one Tree by means of a Tournament in which 'tourn_size' contenders are randomly selected and then 
		compared for their respective fitness (as determined in 'fx_fitness_gym'). The tournament is engaged for each 
		of the four types of inter-generational evolution: reproduction, point mutation, branch (full and grow)
		mutation, and crossover (sexual) reproduction.
		
		The original Tournament Selection drew directly from the foundation generation (gp.generation_a). However, 
		with the introduction of a minimum boundary condition as defined by the user ('gp.tree_depth_min'), 
		'gp.gene_pool' provides only from those Trees which meet all criteria.
		
		With upper (max depth) and lower (min nodes) boundary conditions invoked, one may enjoy interesting results. 
		Stronger boundary conditions (a reduced gap between the min and max number of nodes) typically forces more 
		creative solutions, but also runs the risk of elitism, even total population die-off where a healthy 
		population once existed.
		'''
		
		tourn_test = 0
		# short_test = 0 # an incomplete parsimony test (seeking shortest solution)
		
		if self.display == 'i': print '\n\tEnter the tournament ...'
		
		for n in range(tourn_size):
			# tree_id = np.random.randint(1, self.tree_pop_max + 1) # former method of selection from the unfiltered population
			rnd = np.random.randint(len(self.gene_pool)) # select one tree at random from the gene pool
			tree_id = int(self.gene_pool[rnd])
			
			fitness = float(self.population_a[tree_id][12][1]) # extract the fitness from the array
			fitness = round(fitness, self.precision) # force 'result' and 'solution' to the same number of floating points
			
			if self.fitness_type == 'max': # if the fitness function is Maximising
			
				# first time through, 'tourn_test' will be initialised below
				
				if fitness > tourn_test: # if the current Tree's 'fitness' is greater than the priors'
					if self.display == 'i': print '\t\033[36m Tree', tree_id, 'has fitness', fitness, '>', tourn_test, 'and leads\033[0;0m'
					tourn_lead = tree_id # set 'TREE_ID' for the new leader
					tourn_test = fitness # set 'fitness' of the new leader
					# short_test = int(self.population_a[tree_id][12][4]) # set len(algo_raw) of new leader
					
				elif fitness == tourn_test: # if the current Tree's 'fitness' is equal to the priors'
					if self.display == 'i': print '\t\033[36m Tree', tree_id, 'has fitness', fitness, '=', tourn_test, 'and leads\033[0;0m'
					tourn_lead = tree_id # in case there is no variance in this tournament
					# tourn_test remains unchanged
					
					# if int(self.population_a[tree_id][12][4]) < short_test:
						# short_test = int(self.population_a[tree_id][12][4]) # set len(algo_raw) of new leader
						# print '\t\033[36m with improved parsimony score of:\033[1m', short_test, '\033[0;0m'
						
				elif fitness < tourn_test: # if the current Tree's 'fitness' is less than the priors'
					if self.display == 'i': print '\t\033[36m Tree', tree_id, 'has fitness', fitness, '<', tourn_test, 'and is ignored\033[0;0m'
					# tourn_lead remains unchanged
					# tourn_test remains unchanged
					
				else:
					print '\n\t\033[31mERROR! Maximising fx_fitness_tournament is all kinds of messed up!\033[0;0m'
					print '\t fitness =', fitness, 'and tourn_test =', tourn_test; self.fx_karoo_pause(0)
				
			
			elif self.fitness_type == 'min': # if the fitness function is Minimising
			
				if tourn_test == 0: # first time through, 'tourn_test' is given a baseline value
					tourn_test = fitness
					
				if fitness < tourn_test: # if the current Tree's 'fitness' is less than the priors'
					if self.display == 'i': print '\t\033[36m Tree', tree_id, 'has fitness', fitness, '<', tourn_test, 'and leads\033[0;0m'
					tourn_lead = tree_id # set 'TREE_ID' for the new leader
					tourn_test = fitness # set 'fitness' of the new leader
					
				elif fitness == tourn_test: # if the current Tree's 'fitness' is equal to the priors'
					if self.display == 'i': print '\t\033[36m Tree', tree_id, 'has fitness', fitness, '=', tourn_test, 'and leads\033[0;0m'
					tourn_lead = tree_id # in case there is no variance in this tournament
					# tourn_test remains unchanged
					
				elif fitness > tourn_test: # if the current Tree's 'fitness' is greater than the priors'
					if self.display == 'i': print '\t\033[36m Tree', tree_id, 'has fitness', fitness, '>', tourn_test, 'and is ignored\033[0;0m'
					# tourn_lead remains unchanged
					# tourn_test remains unchanged
					
				else:
					print '\n\t\033[31mERROR! Minimising fx_fitness_tournament is all kinds of messed up!\033[0;0m'
					print '\t fitness =', fitness, 'and tourn_test =', tourn_test; self.fx_karoo_pause(0)
		
		tourn_winner = np.copy(self.population_a[tourn_lead]) # copy full tree so as to not inadvertantly modify the original tree
		
		if self.display == 'i': print '\n\t\033[36mThe winner of the tournament is Tree:\033[1m', tourn_winner[0][1], '\033[0;0m'
		
		return tourn_winner
		
	
	def fx_fitness_gene_pool(self):
	
		'''		
		With the introduction of the minimum boundary condition (gp.tree_depth_min), the means by which the lower node 
		count is enforced is through the creation of a gene pool from those Trees which contain equal or greater nodes 
		to the user defined limit.
		
		What's more, the gene pool also keeps the solution from defaulting to a simple t/t as with the Kepler problem.
		Howevr, the ramifications of this further limitation on the evolutionary process has not been full studied.
		
		This method is automatically invoked with every Tournament Selection ('fx_fitness_tournament').
		
		At this point in time, the gene pool does *not* limit the number of times any given Tree may be selected for 
		mutation or reproduction.
		'''
		
		self.gene_pool = []
		if self.display == 'i': print '\n Prepare a viable gene pool ...'; self.fx_karoo_pause(0)
		
		for tree_id in range(1, len(self.population_a)):
		
			self.fx_eval_poly(self.population_a[tree_id]) # extract the Polynomial
			
			if len(self.population_a[tree_id][3])-1 >= self.tree_depth_min and self.algo_sym != 1: # if Tree meets the min node count and > 1
				self.gene_pool.append(self.population_a[tree_id][0][1])
				
				if self.display == 'i': print '\t\033[36m Tree', tree_id, 'has >=', self.tree_depth_min, 'nodes and is added to the gene pool\033[0;0m'
				
		if len(self.gene_pool) > 0 and self.display == 'i': print '\n\t The total population of the gene pool is', len(self.gene_pool); self.fx_karoo_pause(0)

		elif len(self.gene_pool) <= 0:
			self.generation_id = self.generation_id - 1 # catch the hidden increment of the 'generation_id'
			self.generation_max = self.generation_id # catch the unused "cont" values in the 'fx_karoo_pause' method
			print '\n\t There are no Trees in the gene pool. Adjust the minimum nodes to a lower value!' #; self.fx_karoo_pause(0)
			
	
	#++++++++++++++++++++++++++++++++++++++++++
	#   Methods to Evolve a Population        |
	#++++++++++++++++++++++++++++++++++++++++++
	
	def fx_evo_point_mutate(self, tree):
	
		'''
		Mutate a single point in any Tree (Grow or Full).
		'''
		
		node = np.random.randint(1, len(tree[3])) # randomly select a point in the Tree (including root)
		if self.display == 'i': print '\t\033[36m with', tree[5][node], 'node\033[1m', tree[3][node], '\033[0;0m\033[36mchosen for mutation\n\033[0;0m'		
		
		if tree[5][node] == 'root':
			rnd = np.random.randint(0, len(self.functions[:,0])) # call the previously loaded .csv which contains all operands
			tree[6][node] = self.functions[rnd][0] # replace function (operand)
			
		if tree[5][node] == 'func':
			rnd = np.random.randint(0, len(self.functions[:,0])) # call the previously loaded .csv which contains all operands
			tree[6][node] = self.functions[rnd][0] # replace function (operand)
			
		if tree[5][node] == 'term':
			rnd = np.random.randint(0, len(self.terminals) - 1) # call the previously loaded .csv which contains all terminals
			tree[6][node] = self.terminals[rnd] # replace terminal (variable)

		tree = self.fx_evo_fitness_wipe(tree) # remove fitness data

		return tree, node # 'node' is returned only to be assigned to the 'tourn_trees' record keeping
		
	
	def fx_evo_full_mutate(self, tree, branch):
	
		'''
		Mutate a branch of a Full method Tree.
		
		The full mutate method is straight-forward. A branch was selected and passed to this method. As the size and 
		shape of the Tree must remain identical, each node is mutated sequentially, where functions remain functions 
		and terminals remain terminals.
		'''
		
		for n in range(len(branch)):
		
			# 'root' is not made available for Full mutation as this would build an entirely new Tree
			
			if tree[5][branch[n]] == 'func':
				# if self.display == 'i': print '\t\033[36m from\033[1m', tree[5][branch[n]], '\033[0;0m\033[36mto\033[1m func \033[0;0m'
				rnd = np.random.randint(0, len(self.functions[:,0])) # call the previously loaded .csv which contains all operands
				tree[6][branch[n]] = self.functions[rnd][0] # replace function (operand)
				
			if tree[5][branch[n]] == 'term':
				# if self.display == 'i': print '\t\033[36m from\033[1m', tree[5][branch[n]], '\033[0;0m\033[36mto\033[1m term \033[0;0m'
				rnd = np.random.randint(0, len(self.terminals) - 1) # call the previously loaded .csv which contains all terminals
				tree[6][branch[n]] = self.terminals[rnd] # replace terminal (variable)
				
		tree = self.fx_evo_fitness_wipe(tree) # remove fitness data
		
		return tree
		
	
	def fx_evo_grow_mutate(self, tree, branch):
	
		'''
		Mutate a branch of a Grow method Tree.
		
		A branch is selected within a given tree. If the top of that branch is a terminal which does not reside at 
		'tree_depth_max', then it may either remain a terminal (in which case a new value is randomly assigned) or it 
		may mutate into a function. If it becomes a function, a new branch (mini-tree) is generated to be appended to 
		that terminal's current location. The same is true for function-to-function mutation. If however a function 
		mutates into a terminal, then the entire branch beneath the function is deleted from the array.
		
		If the point of mutation ('branch_top') resides at 'tree_depth_max', we do not need to grow a new tree. As the 
		methods for building trees always assume root (node 0) to be a function, this would force our tree beyond its 
		maximum depth. To avoid pain and suffering, we intercept any Grow method, 'branch_depth' = 0 (maximum depth) 
		mutation and replace it with another randomly chosen terminal.
		'''
		
		branch_depth = int(tree[2][1]) - int(tree[4][branch[0]]) # 'tree_depth_max' - depth at 'branch_top' to set max potential size of new branch
		
		if branch_depth < 0:
			print '\n\t\033[31mERROR! Captain, this is not logical!\033[0;0m'
			print '\t branch_depth =', branch_depth; self.fx_karoo_pause(0)
			
		elif branch_depth == 0: # check if we are at 'tree_depth_max' (per the notes above), then mutate term to term
		
			# if self.display == 'i': print '\t\033[36m max depth mutate\033[1m', branch[0], '\033[0;0m\033[36mfrom \033[1mterm\033[0;0m \033[36mto \033[1mterm\033[0;0m\n'
			rnd = np.random.randint(0, len(self.terminals) - 1) # call the previously loaded .csv which contains all terminals
			tree[6][branch[0]] = self.terminals[rnd] # replace terminal (variable)
			
			
		else: # now we are working with a branch >= depth 1 (min 3 nodes) within 'tourn_winner'
		
			# type_mod = 'func' # force to 'func' or 'term' and comment the next 3 lines for test runs and debug
			rnd = np.random.randint(2)
			if rnd == 0: type_mod = 'func' # randomly selected as Function
			elif rnd == 1: type_mod = 'term' # randomly selected as Terminal
			
			if type_mod == 'term': # mutate 'branch_top' to a terminal and delete all nodes beneath (no subsequent nodes are added to this branch)
			
				branch_top = int(branch[0])
				
				# if self.display == 'i': print '\t\033[36m branch node\033[1m', tree[3][branch_top], '\033[0;0m\033[36mmutates from\033[1m', tree[5][branch_top], '\033[0;0m\033[36mto\033[1m term \n\033[0;0m'
				if self.display == 'db': print '\n *** New Branch for Grow - Terminal Mutation *** \n This is the unaltered tourn_winner:\n', tree
				
				rnd = np.random.randint(0, len(self.terminals) - 1) # call the previously loaded .csv which contains all terminals
				tree[5][branch_top] = 'term' # replace type ('func' to 'term' or 'term' to 'term')
				tree[6][branch_top] = self.terminals[rnd] # replace label
				
				tree = np.delete(tree, branch[1:], axis = 1) # delete all nodes beneath point of mutation ('branch_top')
				
				tree = self.fx_evo_node_arity_fix(tree) # fix all node arities
				tree = self.fx_evo_child_link_fix(tree) # fix all child links
				tree = self.fx_evo_node_renum(tree) # renumber all 'NODE_ID's
				
				if self.display == 'db': print '\n This is tourn_winner after terminal', branch_top, 'mutation, branch deletion, and updates:\n', tree; self.fx_karoo_pause(0)
				
			
			if type_mod == 'func': # mutate 'branch_top' to a function (a new 'gp.tree' will be copied, node by node, into 'tourn_winner')
			
				branch_top = int(branch[0])
				
				# if self.display == 'i': print '\t\033[36m branch node\033[1m', tree[3][branch_top], '\033[0;0m\033[36mmutates from\033[1m', tree[5][branch_top], '\033[0;0m\033[36mto\033[1m func \n\033[0;0m'
				if self.display == 'db': print '\n *** New Branch for Grow - Function Mutation *** \n This is the unaltered tourn_winner:\n', tree
				
				branch_depth = int(tree[2][1]) - int(tree[4][branch_top]) # max potential size of 'tree' to insert into array			
				self.fx_gen_tree_build('mutant', self.pop_tree_type, branch_depth) # build new tree ('gp.tree') with a maximum depth which matches 'branch'
				
				if self.display == 'db': print '\n This is the new tree to be inserted at node', branch_top, 'in tourn_winner:\n', self.tree; self.fx_karoo_pause(0)
				
				tree = self.fx_evo_branch_top_copy(tree, branch) # copy root of new 'gp.tree' to point of mutation ('branch_top') in 'tree' ('tourn_winner')
				tree = self.fx_evo_branch_body_copy(tree) # copy remaining nodes in new 'gp.tree' to 'tree' ('tourn_winner')
				
		tree = self.fx_evo_fitness_wipe(tree) # wipe fitness data
		
		return tree
		
	
	def fx_evo_crossover(self, parent_x, branch_x, parent_y, branch_y):
	
		'''
		Through tournament selection, two trees are selected as parents for a single offspring. Within each tree a 
		branch is selected and copied. One tree's branch is then grafted onto a copy of the other parent tree. The 
		resulting, new tree is moved into the new generation.
		
		Currently, each pair of parent Trees produces only one offspring.
		
		This method may be called twice to produce a second children per pair of parent Trees. However, 'parent_a'
		will be passed to 'parent_x' and 'parent_b' to 'parent_y' for the first child, and then 'parent_b' to 
		'parent_x' and 'parent_a' to 'parent_y' (and their branches) for the second child accordingly.
		
		Future versions will handle this automatically.
		
		In applications of GP, Crossover Reproduction is the most commonly applied evolutionary operator.
		'''
		
		crossover = int(branch_x[0]) # a pointer to the top of the branch in 'parent_x'
		branch_top = int(branch_y[0]) # a pointer to the top of the branch in 'parent_y'
		
		# As the 'fx_evo_branch_select' method recursively chases a branch from top to bottom,
		# a branch of one node must be a terminal. Therefore a branch of len 1 may be immediately 
		# applied to Crossover without the hassle of generating a new, stand-alone tree.
		
		if len(branch_x) == 1: # if the branch from 'parent_x' contains only one node (terminal)
		
			if self.display == 'i': print '\t\033[36m terminal crossover from \033[1mparent', parent_x[0][1], '\033[0;0m\033[36mto \033[1mparent', parent_y[0][1], '\033[0;0m\033[36mat node\033[1m', branch_top, '\033[0;0m'
			
			if self.display == 'db':
				print '\nFrom parent_y:\n', parent_y
				print '\n ... we will remove nodes', branch_y, 'and replace node', branch_top, 'with a terminal from branch_x'; self.fx_karoo_pause(0)
				
			parent_y[5][branch_top] = 'term' # replace type
			parent_y[6][branch_top] = parent_x[6][crossover] # replace label
			parent_y[8][branch_top] = 0 # set terminal arity
			
			parent_y = np.delete(parent_y, branch_y[1:], axis = 1) # delete all nodes beneath point of mutation ('branch_top')
			parent_y = self.fx_evo_child_link_fix(parent_y) # fix all child links
			parent_y = self.fx_evo_node_renum(parent_y) # renumber all 'NODE_ID's
			
			if self.display == 'db': print parent_y; self.fx_karoo_pause(0)
				
		else: # we are working with a branch from 'parent_x' >= depth 1 (min 3 nodes)
		
			if self.display == 'i': print '\t\033[36m branch crossover from \033[1mparent', parent_x[0][1], '\033[0;0m\033[36mto \033[1mparent', parent_y[0][1], '\033[0;0m\033[36mat node\033[1m', branch_top, '\033[0;0m'
			
			# self.fx_gen_tree_build('test', 'f', 2) # to use for debug, disable the next 'self.tree ...' line
			self.tree = self.fx_evo_branch_copy(parent_x, branch_x) # generate stand-alone 'gp.tree' with properties of 'branch_x'
			
			if self.display == 'db':
				print '\nFrom parent_x:\n', parent_x
				print '\n ... we extract branch_x', branch_x, 'as a new tree:\n', self.tree; self.fx_karoo_pause(0)
				
			if self.display == 'db':
				print ' ... to be inserted into parent_y:\n', parent_y
				print '\n ... in place of branch_y:', branch_y; self.fx_karoo_pause(0)
				
			parent_y = self.fx_evo_branch_top_copy(parent_y, branch_y) # copy root of 'branch_y' ('gp.tree') to 'parent_y'
			parent_y = self.fx_evo_branch_body_copy(parent_y) # copy remaining nodes in 'branch_y' ('gp.tree') to 'parent_y'
			parent_y = self.fx_evo_tree_prune(parent_y, int(parent_y[2][1])) # prune to the user defined maximum depth
			
		parent_y = self.fx_evo_fitness_wipe(parent_y) # wipe fitness data
		
		return parent_y
		
	
	def fx_evo_branch_select(self, tree):
	
		'''
		Select all nodes in the 'tourn_winner' Tree at and below the randomly selected starting point.
		
		While Grow mutation uses this method to select a region of the 'tourn_winner' to delete, Crossover mutation 
		uses this method to select a region of the 'tourn_winner' which is then converted to a stand-alone tree. As 
		such, it is imperative that the nodes be in the correct order, else all kinds of bad things happen.
		'''
		
		branch = np.array([]) # the array is necessary in order to len(branch) when 'branch' has only one element
		branch_top = np.random.randint(2, len(tree[3])) # randomly select a non-root node
		branch_eval = self.fx_eval_id(tree, branch_top) # generate tuple of 'branch_top' and subseqent nodes
		branch_symp = sp.sympify(branch_eval) # convert string into something useful
		branch = np.append(branch, branch_symp)
		
		branch = np.sort(branch) # sort nodes in branch for Crossover Reproduction.
		
		if self.display == 'i': print '\t \033[36mwith nodes\033[1m', branch, '\033[0;0m\033[36mchosen for mutation\033[0;0m'
		
		return branch
		
	
	def fx_evo_branch_top_copy(self, tree, branch):
	
		'''
		Copy the point of mutation ('branch_top') from 'gp.tree' to 'tree'.
		
		This method works with 3 inputs: local 'tree' is being modified; local 'branch' is a section of 'tree' which 
		will be removed; and global 'gp.tree' (recycling from initial population generation) is the new tree to be 
		copied into 'tree', replacing 'branch'.
		
		This is used in both Grow Mutation and Crossover Reproduction.
		'''
		
		branch_top = int(branch[0])
		
		tree[5][branch_top] = 'func' # update type ('func' to 'term' or 'term' to 'term'); this modifies gp.tree[5[1] from 'root' to 'func'
		tree[6][branch_top] = self.tree[6][1] # copy node_label from new tree
		tree[8][branch_top] = self.tree[8][1] # copy node_arity from new tree
		
		tree = np.delete(tree, branch[1:], axis = 1) # delete all nodes beneath point of mutation ('branch_top')
		
		c_buffer = self.fx_evo_c_buffer(tree, branch_top) # generate c_buffer for point of mutation ('branch_top')
		tree = self.fx_evo_child_insert(tree, branch_top, c_buffer) # insert new nodes
		tree = self.fx_evo_node_renum(tree) # renumber all 'NODE_ID's
		
		return tree
		
	
	def fx_evo_branch_body_copy(self, tree):
	
		'''
		Copy the body of 'gp.tree' to 'tree', one node at a time.
		
		This method works with 3 inputs: local 'tree' is being modified; local 'branch' is a section of 'tree' which 
		will be removed; and global 'gp.tree' (recycling from initial population generation) is the new tree to be 
		copied into 'tree', replacing 'branch'.
		
		This is used in both Grow and Crossover Reproduction.
		'''
				
		node_count = 2 # set node count for 'gp.tree' to 2 as the new root has already replaced 'branch_top' in 'fx_evo_branch_top_copy'
		
		while node_count < len(self.tree[3]): # increment through all nodes in the new tree ('gp.tree'), starting with node 2
		
			for j in range(1, len(tree[3])): # increment through all nodes in tourn_winner ('tree')
			
				if self.display == 'db': print '\tScanning tourn_winner node_id:', j
				
				if tree[5][j] == '':					
					tree[5][j] = self.tree[5][node_count] # copy 'node_type' from branch to tree
					tree[6][j] = self.tree[6][node_count] # copy 'node_label' from branch to tree
					tree[8][j] = self.tree[8][node_count] # copy 'node_arity' from branch to tree
					
					if tree[5][j] == 'term':
						tree = self.fx_evo_child_link_fix(tree) # fix all child links
						tree = self.fx_evo_node_renum(tree) # renumber all 'NODE_ID's
						
					if tree[5][j] == 'func':
						c_buffer = self.fx_evo_c_buffer(tree, j) # generate 'c_buffer' for point of mutation ('branch_top')
						tree = self.fx_evo_child_insert(tree, j, c_buffer) # insert new nodes
						tree = self.fx_evo_child_link_fix(tree) # fix all child links
						tree = self.fx_evo_node_renum(tree) # renumber all 'NODE_ID's
						
					if self.display == 'db':
						print '\t inserted new tree node', node_count, 'of', len(self.tree[3])-1
						print '\n This is tourn_winner after the new nodes are inserted and updated:\n', tree; self.fx_karoo_pause(0)
						
					node_count = node_count + 1 # exit loop when 'node_count' reaches the number of columns in the array 'gp.tree'
							
		return tree
		
	
	def fx_evo_branch_copy(self, tree, branch):
	
		'''
		This method prepares a stand-alone tree as a copy of the given branch.
		
		This method is used with Crossover Reproduction.
		'''
		
		new_tree = np.array([ ['TREE_ID'],['tree_type'],['tree_depth_max'],['NODE_ID'],['node_depth'],['node_type'],['node_label'],['node_parent'],['node_arity'],['node_c1'],['node_c2'],['node_c3'],['fitness'] ])
		
		# tested 2015 06/08
		for n in range(len(branch)):
		
			node = branch[n]
			branch_top = int(branch[0])
			
			TREE_ID = 'copy'
			tree_type = tree[1][1]
			tree_depth_max = int(tree[4][branch[-1]]) - int(tree[4][branch[0]]) # subtract depth of the first node from the last in 'branch'
			NODE_ID = tree[3][node]
			node_depth = int(tree[4][node]) - int(tree[4][branch_top]) # subtract the depth of 'branch_top' from the current node depth
			node_type = tree[5][node]
			node_label = tree[6][node]
			node_parent = '' # updated by 'fx_evo_parent_link_fix', below
			node_arity = tree[8][node]
			node_c1 = '' # updated by 'fx_evo_child_link_fix', below
			node_c2 = ''
			node_c3 = ''
			fitness = ''
			
			new_tree = np.append(new_tree, [ [TREE_ID],[tree_type],[tree_depth_max],[NODE_ID],[node_depth],[node_type],[node_label],[node_parent],[node_arity],[node_c1],[node_c2],[node_c3],[fitness] ], 1)
			
		new_tree = self.fx_evo_node_renum(new_tree)
		new_tree = self.fx_evo_child_link_fix(new_tree)
		new_tree = self.fx_evo_parent_link_fix(new_tree)
		new_tree = self.fx_tree_clean(new_tree)
		
		return new_tree
		
	
	def fx_evo_c_buffer(self, tree, node):
	
		'''
		This method serves the very important function of determining the links from parent to child for any given 
		node. The single, simple formula [parent_arity_sum + prior_sibling_arity - prior_siblings] perfectly determines 
		the correct position of the child node, already in place or to be inserted, no matter the depth nor complexity 
		of the tree.
		
		This method is currently called from the evolution methods, but will soon (I hope) be called from the first 
		generation Tree generation methods (above) such that the same method may be used repeatedly.
		'''
		
		parent_arity_sum = 0
		prior_sibling_arity = 0
		prior_siblings = 0
		
		for n in range(1, len(tree[3])): # increment through all nodes (exclude 0) in array 'tree'
		
			if int(tree[4][n]) == int(tree[4][node])-1: # find parent nodes at the prior depth
				if tree[8][n] != '': parent_arity_sum = parent_arity_sum + int(tree[8][n]) # sum arities of all parent nodes at the prior depth
				
			if int(tree[4][n]) == int(tree[4][node]) and int(tree[3][n]) < int(tree[3][node]): # find prior siblings at the current depth
				if tree[8][n] != '': prior_sibling_arity = prior_sibling_arity + int(tree[8][n]) # sum prior sibling arity
				prior_siblings = prior_siblings + 1 # sum quantity of prior siblings
				
		c_buffer = node + (parent_arity_sum + prior_sibling_arity - prior_siblings) # One algo to rule the world!
		
		return c_buffer
		
	
	def fx_evo_child_link(self, tree, node, c_buffer):
	
		'''
		Link each parent node to its children.
		'''
		
		if int(tree[3][node]) == 1: c_buffer = c_buffer + 1 # if root (node 1) is passed through this method
		
		if tree[8][node] != '':
		
			if int(tree[8][node]) == 0: # if arity = 0
				tree[9][node] = ''
				tree[10][node] = ''
				tree[11][node] = ''
				
			elif int(tree[8][node]) == 1: # if arity = 1
				tree[9][node] = c_buffer
				tree[10][node] = ''
				tree[11][node] = ''
				
			elif int(tree[8][node]) == 2: # if arity = 2
				tree[9][node] = c_buffer
				tree[10][node] = c_buffer + 1
				tree[11][node] = ''
				
			elif int(tree[8][node]) == 3: # if arity = 3
				tree[9][node] = c_buffer
				tree[10][node] = c_buffer + 1
				tree[11][node] = c_buffer + 2
				
			else:
				print '\n\t\033[31mERROR! Something has gone very wrong with the mutant children!\033[0;0m'
				print '\t node', node, 'arity:', tree[8][node]; self.fx_karoo_pause(0)
				
		return tree
		
	
	def fx_evo_child_link_fix(self, tree):
	
		'''
		In a given Tree, fix 'node_c1', 'node_c2', 'node_c3' for all nodes.
		
		This is required anytime the size of the array 'gp.tree' has been modified, as with both Grow and Full mutation.
		'''
		
		# tested 2015 06/04
		for node in range(1, len(tree[3])):
		
			c_buffer = self.fx_evo_c_buffer(tree, node) # generate c_buffer for each node
			tree = self.fx_evo_child_link(tree, node, c_buffer) # update child links for each node
			
		return tree
		
	
	def fx_evo_child_insert(self, tree, node, c_buffer):
	
		'''
		Insert child nodes.
		'''
		
		if int(tree[8][node]) == 0: # if arity = 0
			print '\n\t\033[31mERROR! Arity = 0 in fx_evo_child_insert --not good!\033[0;0m'; self.fx_karoo_pause(0)
		
		elif int(tree[8][node]) == 1: # if arity = 1
			tree = np.insert(tree, c_buffer, '', axis=1) # insert node for 'node_c1'
			tree[3][c_buffer] = c_buffer # node ID
			tree[4][c_buffer] = int(tree[4][node]) + 1 # node_depth
			tree[7][c_buffer] = int(tree[3][node]) # parent ID
			
		elif int(tree[8][node]) == 2: # if arity = 2
			tree = np.insert(tree, c_buffer, '', axis=1) # insert node for 'node_c1'
			tree[3][c_buffer] = c_buffer # node ID
			tree[4][c_buffer] = int(tree[4][node]) + 1 # node_depth
			tree[7][c_buffer] = int(tree[3][node]) # parent ID
			
			tree = np.insert(tree, c_buffer + 1, '', axis=1) # insert node for 'node_c2'
			tree[3][c_buffer + 1] = c_buffer + 1 # node ID
			tree[4][c_buffer + 1] = int(tree[4][node]) + 1 # node_depth
			tree[7][c_buffer + 1] = int(tree[3][node]) # parent ID
			
		elif int(tree[8][node]) == 3: # if arity = 3
			tree = np.insert(tree, c_buffer, '', axis=1) # insert node for 'node_c1'
			tree[3][c_buffer] = c_buffer # node ID
			tree[4][c_buffer] = int(tree[4][node]) + 1 # node_depth
			tree[7][c_buffer] = int(tree[3][node]) # parent ID
			
			tree = np.insert(tree, c_buffer + 1, '', axis=1) # insert node for 'node_c2'
			tree[3][c_buffer + 1] = c_buffer + 1 # node ID
			tree[4][c_buffer + 1] = int(tree[4][node]) + 1 # node_depth
			tree[7][c_buffer + 1] = int(tree[3][node]) # parent ID
			
			tree = np.insert(tree, c_buffer + 2, '', axis=1) # insert node for 'node_c3'
			tree[3][c_buffer + 2] = c_buffer + 2 # node ID
			tree[4][c_buffer + 2] = int(tree[4][node]) + 1 # node_depth
			tree[7][c_buffer + 2] = int(tree[3][node]) # parent ID
			
		else: print '\n\t\033[31mERROR! Arity > 3 in fx_evo_child_insert --even more not good!\033[0;0m'; self.fx_karoo_pause(0)
			
		return tree
		
	
	def fx_evo_parent_link_fix(self, tree):
	
		'''
		In a given Tree, fix 'parent_id' for all nodes.
		
		This is automatically handled in all mutations except with Crossover due to the need to copy branches 'a' and 
		'b' to their own trees before inserting them into copies of	the parents.
		
		Technically speaking, the 'node_parent' value is not used by any methods. The parent ID can be completely out 
		of whack and the polynomial expression will work perfectly. This is maintained for the sole purpose of granting 
		the user a friendly, makes-sense interface which can be read in both directions.
		'''
		
		### THIS METHOD MAY NOT BE NEEDED AS SORTING 'branch' SEEMS TO HAVE FIXED 'parent_id' ###
		
		# tested 2015 06/05
		for node in range(1, len(tree[3])):
		
			if tree[9][node] != '':
				child = int(tree[9][node])
				tree[7][child] = node
				
			if tree[10][node] != '':
				child = int(tree[10][node])
				tree[7][child] = node
				
			if tree[11][node] != '':
				child = int(tree[11][node])
				tree[7][child] = node
				
		return tree
		
	
	def fx_evo_node_arity_fix(self, tree):
	
		'''
		In a given Tree, fix 'node_arity' for all nodes labeled 'term' but with arity 2.
		
		This is required after a function has been replaced by a terminal, as may occur with both Grow mutation and 
		Crossover reproduction.
		'''
		
		# tested 2015 05/31
		for n in range(1, len(tree[3])): # increment through all nodes (exclude 0) in array 'tree'
		
			if tree[5][n] == 'term': # check for discrepency
				tree[8][n] = '0' # set arity to 0
				tree[9][n] = '' # wipe 'node_c1'
				tree[10][n] = '' # wipe 'node_c2'
				tree[11][n] = '' # wipe 'node_c3'
				
		return tree
		
		
	def fx_evo_node_renum(self, tree):
	
		'''
		Renumber all 'NODE_ID' in a given tree.
		
		This is required after a new generation is evolved as the NODE_ID numbers are carried forward from the previous 
		generation but are no longer in order.
		'''
		
		for n in range(1, len(tree[3])):
		
			tree[3][n] = n  # renumber all Trees in given population
			
		return tree
		
	
	def fx_evo_fitness_wipe(self, tree):
	
		'''
		Remove all fitness data from a given tree.
		
		This is required after a new generation is evolved as the fitness of the same tree prior to its mutation will 
		no longer apply.
		'''
		
		tree[12][1:] = '' # remove all 'fitness' data
		
		return tree
		
	
	def fx_evo_tree_prune(self, tree, depth):
	
		'''
		This method reduces the depth of a given branch.
				
		This method is used with Crossover Reproduction. However, the input value 'branch' can be a partial tree 
		(branch) or a full tree, and it will operate correctly. The input value 'depth' becomes the new maximum depth.
		'''
		
		nodes = []
		
		# tested 2015 06/08
		for n in range(1, len(tree[3])):
		
			if int(tree[4][n]) == depth and tree[5][n] == 'func':
				rnd = np.random.randint(0, len(self.terminals) - 1) # call the previously loaded .csv which contains all terminals
				tree[5][n] = 'term' # mutate type 'func' to 'term'
				tree[6][n] = self.terminals[rnd] # replace label
				
			elif int(tree[4][n]) > depth:
				nodes.append(n)
				
		tree = np.delete(tree, nodes, axis = 1) # delete nodes whose depth is greater than 'depth'
		tree = self.fx_evo_node_arity_fix(tree) # fix all node arities
		
		return tree
		
		
	def fx_evo_tree_renum(self, population):
	
		'''
		Renumber all 'TREE_ID' in a given population.
		
		This is required after a new generation is evolved as the TREE_ID numbers are carried forward from the previous 
		generation but are no longer in order.
		'''
		
		for tree_id in range(1, len(population)):
		
			population[tree_id][0][1] = tree_id  # renumber all Trees in given population
			
		return population
		
	
	def fx_evo_pop_copy(self, pop_a, title):
	
		'''
		Copy one population to another.
		
		Simply copying a list of arrays generates a pointer to the original list. Therefore we must append each array 
		to a new, empty array and then build a list of those new arrays.
		'''
		
		pop_b = [title] # an empty list stores a copy of the prior generation
	
		for tree in range(1, len(pop_a)): # increment through each Tree in the current population

			tree_copy = np.copy(pop_a[tree]) # copy each array in the current population
			pop_b.append(tree_copy) # add each copied Tree to the new population list
			
		return pop_b
		
		
	#++++++++++++++++++++++++++++++++++++++++++
	#   Methods to Test a Tree                |
	#++++++++++++++++++++++++++++++++++++++++++
	
	def fx_test_abs(self, tree_id):
	
		'''
		[need to write]
		'''
		
		self.fx_eval_poly(self.population_a[tree_id]) # generate the raw and sympified equation for the given Tree
		print '\n\t\033[36mTree', tree_id, 'yields (raw):', self.algo_raw, '\033[0;0m'
		print '\t\033[36mTree', tree_id, 'yields (sym):\033[1m', self.algo_sym, '\033[0;0m\n'
		
		for row in range(0, self.data_test_rows): # test against data_test_dict
			data_test_dict = self.data_test_dict_array[row] # re-assign (unpack) a temp dictionary to each row of data
			
			if str(self.algo_sym.subs(data_test_dict)) == 'zoo': # divide by zero demands we avoid use of the 'float' function
				result = self.algo_sym.subs(data_test_dict) # print 'divide by zero', result; self.fx_karoo_pause(0)
				
			else:
				result = float(self.algo_sym.subs(data_test_dict)) # process the polynomial to produce the result
				result = round(result, self.precision) # force 'result' and 'solution' to the same number of floating points
				
			solution = float(data_test_dict['s']) # extract the desired solution from the data
			solution = round(solution, self.precision) # force 'result' and 'solution' to the same number of floating points
			
			# fitness = abs(result - solution) # this is a Minimisation function (seeking smallest fitness)
			print '\t\033[36m data row', row, 'yields:', result, '\033[0;0m'
		
		# measure the total or average difference between result and solution across all rows ???
		
		print '\n\t (this test is not yet complete)'
		
		return
		
	
	def fx_test_match(self, tree_id):
	
		'''
		[need to write]
		'''
		
		self.fx_eval_poly(self.population_a[tree_id]) # generate the raw and sympified equation for the given Tree
		print '\n\t\033[36mTree', tree_id, 'yields (raw):', self.algo_raw, '\033[0;0m'
		print '\t\033[36mTree', tree_id, 'yields (sym):\033[1m', self.algo_sym, '\033[0;0m\n'
		
		for row in range(0, self.data_test_rows): # test against data_test_dict
			data_test_dict = self.data_test_dict_array[row] # re-assign (unpack) a temp dictionary to each row of data
			
			if str(self.algo_sym.subs(data_test_dict)) == 'zoo': # divide by zero demands we avoid use of the 'float' function
				result = self.algo_sym.subs(data_test_dict) # print 'divide by zero', result; self.fx_karoo_pause(0)
				
			else:
				result = float(self.algo_sym.subs(data_test_dict)) # process the polynomial to produce the result
				result = round(result, self.precision) # force 'result' and 'solution' to the same number of floating points
			
			solution = float(data_test_dict['s']) # extract the desired solution from the data
			solution = round(solution, self.precision) # force 'result' and 'solution' to the same number of floating points
			
			if result == solution:
				fitness = 1 # improve the fitness score by 1		
				print '\t\033[36m data row', row, '\033[0;0m\033[36myields:\033[1m', result, '\033[0;0m'
				
			else:
				fitness = 0 # do not adjust the fitness score
				print '\t\033[36m data row', row, 'yields:', result, '\033[0;0m'
				
		print '\n\t Tree', tree_id, 'has an accuracy of:', float(self.population_a[tree_id][12][1]) / self.data_test_dict_array.shape[0] * 100
		
		# IS THAT ALL ???
		
		return
		
	
	def fx_test_classify(self, tree_id):
	
		'''
		Conduct class Precision / Recall on selected Trees against the TEST data.
		
		(see fx_fitness_function_classify for more information)

		From scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
			Precision (P) = true_pos / true_pos + false_pos
			Recall (R) = true_pos / true_pos + false_neg
			harmonic mean of Precision and Recall (F1) = 2(P x R) / (P + R)
			
		From scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
			y_pred = result, the estimated target values (labels) generated by Karoo GP
			y_true = solution, the correct target values (labels) associated with the data
		'''
		
		# tested 2015 10/18
		
		y_pred = []
		y_true = []
		
		self.fx_eval_poly(self.population_a[tree_id]) # generate the raw and sympified equation for the given Tree
		print '\n\t\033[36mTree', tree_id, 'yields (raw):', self.algo_raw, '\033[0;0m'
		print '\t\033[36mTree', tree_id, 'yields (sym):\033[1m', self.algo_sym, '\033[0;0m\n'
		
		skew = (self.class_labels / 2) - 1 # '-1' keeps a binary classification splitting over the origin
		# skew = 0 # for code testing
		
		for row in range(0, self.data_test_rows): # test against data_test_dict
			data_test_dict = self.data_test_dict_array[row] # re-assign (unpack) a temp dictionary to each row of data
			
			if str(self.algo_sym.subs(data_test_dict)) == 'zoo': # divide by zero demands we avoid use of the 'float' function
				result = self.algo_sym.subs(data_test_dict) # print 'divide by zero', result; self.fx_karoo_pause(0)
				
			else:
				result = float(self.algo_sym.subs(data_test_dict)) # process the polynomial to produce the result
				result = round(result, self.precision) # force 'result' to the set number of floating points
				
			label_pred = '' # we can remove this and the associated "if label_pred == ''" (below) once thoroughly tested - 2015 10/19
			label_true = int(data_test_dict['s'])
			
			if result <= 0 - skew: # test for the first class
				label_pred = 0
				print '\t\033[36m data row', row, 'predicts class:\033[1m', label_pred, '(', label_true, ') as', result, '<=', 0 - skew, '\033[0;0m'
								
			elif result > (self.class_labels - 2) - skew: # test for last class (the right-most bin
				label_pred = self.class_labels - 1
				print '\t\033[36m data row', row, 'predicts class:\033[1m', label_pred, '(', label_true, ') as', result, '>', (self.class_labels - 2) - skew, '\033[0;0m'
				
			else:
				for class_label in range(1, self.class_labels - 1): # increment through all class labels, skipping first and last
				
					if (class_label - 1) - skew < result <= class_label - skew: # test for classes between first and last
						label_pred = class_label
						print '\t\033[36m data row', row, 'predicts class:\033[1m', label_pred, '(', label_true, ') as', (class_label - 1) - skew, '<', result, '<=', class_label - skew, '\033[0;0m'
						
					else: pass # print '\t\033[36m data row', row, 'predicts no class with label', class_label, '(', label_true, ') and result', result, '\033[0;0m'
					
			if label_pred == '': print '\n\t\033[31mERROR! I am sorry to report that tree', tree_id, 'failed to generate a class label prediction.\033[0;0m'; self.fx_karoo_pause(0)
			
			y_pred.append(label_pred)
			y_true.append(label_true)
			
		print '\nClassification report for Tree: ', tree_id
		print skm.classification_report(y_true, y_pred)
		
		print '\nConfusion matrix for Tree: ', tree_id
		print skm.confusion_matrix(y_true, y_pred)
		
		return
		
	
	def fx_test_normalize(self, array):
	
		'''
		This method refits each data point within the given array to within 0 through 1, where 0 is the minimum and 1 
		is the maximum value.
		
		The formula employed was derived from the following website:
		stn.spotfire.com/spotfire_client_help/norm/norm_normalizing_columns.htm 
		'''
		
		norm = []
		array_norm = []
		array_min = np.min(array)
		array_max = np.max(array)
		
		for col in range(1, len(array) + 1):
			norm = float((array[col - 1] - array_min) / (array_max - array_min))			
			array_norm = np.append(array_norm, norm)
			
		return array_norm
		
	
	def fx_test_plot(self, tree):
	
		'''
		# [need to build]
		'''
		
		return
		
	
	#++++++++++++++++++++++++++++++++++++++++++
	#   Methods to Append & Archive           |
	#++++++++++++++++++++++++++++++++++++++++++
	
	def fx_tree_clean(self, tree):
	
		'''
		Clean the Tree array
		'''
		
		tree[0][2:] = '' # A little clean-up to make things look pretty :)
		tree[1][2:] = '' # Ignore the man behind the curtain!
		tree[2][2:] = '' # Yes, I am a bit OCD ... but you *know* you appreciate clean data.
		
		return tree
		
	
	def fx_tree_append(self, tree):
	
		'''
		Append Tree array to the foundation Population 
		'''
		
		self.fx_tree_clean(tree) # clean 'tree' prior to storing
		self.population_a.append(tree) # append 'tree' to population list
		
		return
		
	
	def fx_tree_archive(self, population, key):
	
		'''
		Save Population list to disk
		'''
		
		with open(self.filename[key], 'a') as csv_file:
			target = csv.writer(csv_file, delimiter=',')
			if self.generation_id != 1: target.writerows(['']) # empty row before each generation
			target.writerows([['Karoo GP by Kai Staats', 'Generation:', str(self.generation_id)]])
			# need to add date / time to file header
			
			for tree in range(1, len(population)):
				target.writerows(['']) # empty row before each Tree
				for row in range(0, 13): # increment through each row in the array Tree
					target.writerows([population[tree][row]])
					
		return
		
		
