# Karoo GP Base Class
# Define the methods and global variables used by Karoo GP
# by Kai Staats, MSc; see LICENSE.md
# Thanks to Emmanuel Dufourq and Arun Kumar for support during 2014-15 devel; TensorFlow support provided by Iurii Milovanov
# version 1.0.5

'''
A NOTE TO THE NEWBIE, EXPERT, AND BRAVE
Even if you are highly experienced in Genetic Programming, it is recommended that you review the 'Karoo User Guide' before running 
this application. While your computer will not burst into flames nor will the sun collapse into a black hole if you do not, you will 
likely find more enjoyment of this particular flavour of GP with a little understanding of its intent and design.
'''

import sys
import os
import csv
import time

import numpy as np
import sklearn.metrics as skm
import sklearn.model_selection as skcv

from sympy import sympify
from datetime import datetime
from collections import OrderedDict

# TensorFlow-related imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
import ast
import operator as op
operators = {ast.Add: tf.add, # e.g., a + b
             ast.Sub: tf.subtract, # e.g., a - b
             ast.Mult: tf.multiply, # e.g., a * b
             ast.Div: tf.divide, # e.g., a / b
             ast.Pow: tf.pow, # e.g., a ** 2
             ast.USub: tf.negative, # e.g., -a
             ast.And: tf.logical_and, # e.g., a and b
             ast.Or: tf.logical_or, # e.g., a or b
             ast.Not: tf.logical_not, # e.g., not a
             ast.Eq: tf.equal, # e.g., a == b
             ast.NotEq: tf.not_equal, # e.g., a != b
             ast.Lt: tf.less, # e.g., a < b
             ast.LtE: tf.less_equal, # e.g., a <= b
             ast.Gt: tf.greater, # e.g., a > b
             ast.GtE: tf.greater_equal, # e.g., a >= 1
            'abs': tf.abs, # e.g., abs(a)
            'sign': tf.sign, # e.g., sign(a)
            'square': tf.square, # e.g., square(a)
            'sqrt': tf.sqrt, # e.g., sqrt(a)
            'pow': tf.pow, # e.g., pow(a, b)
            'log': tf.log, # e.g., log(a)
            'log1p': tf.log1p, # e.g., log1p(a)
            'cos': tf.cos, # e.g., cos(a)
            'sin': tf.sin, # e.g., sin(a)
            'tan': tf.tan, # e.g., tan(a)
            'acos': tf.acos, # e.g., acos(a)
            'asin': tf.asin, # e.g., asin(a)
            'atan': tf.atan, # e.g., atan(a)
            }

np.set_printoptions(linewidth = 320) # set the terminal to print 320 characters before line-wrapping in order to view Trees


class Base_GP(object):

	'''
	This Base_BP class contains all methods for Karoo GP.
	
	Method names are differentiated from global variable names (defined below) by the prefix 'fx_' followed by an object
	and action, as in 'fx_display_tree()', with a few expections, such as 'fx_fitness_gene_pool'.
	
	The categories (denoted by +++ banners +++) are as follows:
		'karoo_gp'						A single method which conducts an entire run. Employed only by karoo_gp_server.py
		'fx_karoo_'						Methods to Run Karoo GP
		'fx_gen_'							Methods to Generate a Tree
		'fx_eval_'						Methods to Evaluate a Tree
		'fx_fitness_'					Methods to Train and Test a Tree for Fitness
		'fx_evolve_'					Methods to Evolve a Population
		'fx_display_'					Methods to Display a Tree
		'fx_archive_'					Methods to Archive
		
	There are no sub-classes at the time of this edit - 2015 09/21
	'''
	
	#++++++++++++++++++++++++++++++++++++++++++
	#   Define Global Variables               |
	#++++++++++++++++++++++++++++++++++++++++++
	
	def __init__(self):
	
		'''
		All Karoo GP global variables are named with the prefix 'gp.' The 13 variables which begin with 'gp.pop_' are 
		specifically employed to define the 13 parameters for each tree as stored in the axis-1 (expand horizontally) 
		'gp.population' Numpy array.
		
		### Global and local variables defined by the user in karoo_gp_main.py (in order of appearence) ###
		'gp.kernel'							fitness function
		'gp.class_method'				select the number of classes (will be automated in future version)
		'tree_type'							Full, Grow, or Ramped 50/50 (local variable)
		'gp.tree_depth_min'			minimum number of nodes
		'tree_depth_base'				maximum Tree depth for the initial population, where nodes = 2^(depth + 1) - 1
		'gp.tree_depth_max'			maximum Tree depth for the entire run; introduces potential bloat
		'gp.tree_pop_max'				maximum number of Trees per generation
		'gp.generation_max'			maximum number of generations
		'gp.display'						level of on-screen feedback
		
		'gp.evolve_repro'				quantity of a population generated through Reproduction
		'gp.evolve_point'				quantity of a population generated through Point Mutation
		'gp.evolve_branch'			quantity of a population generated through Branch Mutation
		'gp.evolve_cross'				quantity of a population generated through Crossover
		
		'gp.tourn_size'					the number of Trees chosen for each tournament
		'gp.precision'					the number of floating points for all applications of the round function
		
		### Global variables used for data management ###
		'gp.data_train'					store train data for processing in TF
		'gp.data_test'					store test data for processing in TF
		'gp.tf_device'					set TF computation backend device (CPU or GPU)
		'gp.tf_device_log'			employed for TensorFlow debugging
		
		'gp.data_train_cols'		number of cols in the TRAINING data (see 'fx_karoo_data_load', below)
		'gp.data_train_rows'		number of rows in the TRAINING data (see 'fx_karoo_data_load', below)
		'gp.data_test_cols'			number of cols in the TEST data (see 'fx_karoo_data_load', below)
		'gp.data_test_rows'			number of rows in the TEST data (see 'fx_karoo_data_load', below)
		
		'gp.functions'					user defined functions (operators) from the associated files/[functions].csv
		'gp.terminals'					user defined variables (operands) from the top row of the associated [data].csv
		'gp.coeff'							user defined coefficients (NOT YET IN USE)
		'gp.fitness_type'				fitness type
		'gp.datetime'						date-time stamp of when the unique directory is created
		'gp.path'								full path to the unique directory created with each run
		'gp.dataset'						local path and dataset filename
		
		### Global variables initiated and/or used by Sympy ###
		'gp.algo_raw'						a Sympy string which represents a flattened tree
		'gp.algo_sym'						a Sympy executable version of algo_raw
		'gp.fittest_dict'				a dictionary of the most fit trees, compiled during fitness function execution
		
		### Variables used for evolutionary management ###
		'gp.population_a'				the root generation from which Trees are chosen for mutation and reproduction
		'gp.population_b'				the generation constructed from gp.population_a (recyled)
		'gp.gene_pool'					once-per-generation assessment of trees that meet min and max boundary conditions
		'gp.generation_id'			simple n + 1 increment
		'gp.fitness_type'				set in 'fx_karoo_data_load' as either a minimising or maximising function
		'gp.tree'								axis-1, 13 element Numpy array that defines each Tree, stored in 'gp.population'
		'gp.pop_*'							13 elements which define each Tree (see 'fx_gen_tree_initialise' below)
		
		### Fishing nets ###
		You can insert a "fishing net" to search for a specific expression when you fear the evolutionary process or 
		something in the code may not be working. Search for "fishing net" and follow the directions.
		
		### Error checks ###
		You can quickly find all places in which error checks have been inserted by searching for "ERROR!"
		'''
		
		self.algo_raw = [] # temp store the raw expression -- CONSIDER MAKING THIS VARIABLE LOCAL
		self.algo_sym = [] # temp store the sympified expression-- CONSIDER MAKING THIS VARIABLE LOCAL
		self.fittest_dict = {} # temp store all Trees which share the best fitness score
		self.gene_pool = [] # temp store all Tree IDs for use by Tournament
		self.class_labels = 0 # temp set a variable which will be assigned the number of class labels (data_y)
		
		return
		
	
	#++++++++++++++++++++++++++++++++++++++++++
	#   Methods to Run Karoo GP               |
	#++++++++++++++++++++++++++++++++++++++++++
	
	def karoo_gp(self, tree_type, tree_depth_base, filename):
	
		'''
		This method enables the engagement of the entire Karoo GP application. It is used exclusively by the server script
		karoo_gp_server.py (not by the desktop script karoo_gp_main.py). Instead of returning the user to the pause menu,
		this script terminates at the command-line, providing support for bash and chron job execution.
		
		Arguments required: tree_type, tree_depth_base, filename
		'''
		
		self.karoo_banner()
		start = time.time() # start the clock for the timer
		
		# construct first generation of Trees
		self.fx_karoo_data_load(tree_type, tree_depth_base, filename)
		self.generation_id = 1 # set initial generation ID
		self.population_a = ['Karoo GP by Kai Staats, Generation ' + str(self.generation_id)] # list to store all Tree arrays, one generation at a time
		self.fx_karoo_construct(tree_type, tree_depth_base) # construct the first population of Trees
		
		# evaluate first generation of Trees	
		print('\n Evaluate the first generation of Trees ...')	
		self.fx_fitness_gym(self.population_a) # generate expression, evaluate fitness, compare fitness
		self.fx_archive_tree_write(self.population_a, 'a') # save the first generation of Trees to disk
		
		# evolve subsequent generations of Trees
		for self.generation_id in range(2, self.generation_max + 1): # loop through 'generation_max'
		
			print('\n Evolve a population of Trees for Generation', self.generation_id, '...')
			self.population_b = ['Karoo GP by Kai Staats, Evolving Generation'] # initialise population_b to host the next generation
			
			self.fx_fitness_gene_pool() # generate the viable gene pool (compares against gp.tree_depth_min)
			self.fx_karoo_reproduce() # method 1 - Reproduction
			self.fx_karoo_point_mutate() # method 2 - Point Mutation
			self.fx_karoo_branch_mutate() # method 3 - Branch Mutation
			self.fx_karoo_crossover() # method 4 - Crossover
			self.fx_eval_generation() # evaluate all Trees in a single generation
			
			self.population_a = self.fx_evolve_pop_copy(self.population_b, ['Karoo GP by Kai Staats, Generation ' + str(self.generation_id)])
	
		# "End of line, man!" --CLU
		print('\n \033[36m Karoo GP has an ellapsed time of \033[0;0m\033[31m%f\033[0;0m' % (time.time() - start), '\033[0;0m')
		self.fx_archive_tree_write(self.population_b, 'f') # save the final generation of Trees to disk
		self.fx_archive_params_write('Server') # save run-time parameters to disk
		
		print('\n \033[3m Congrats!\033[0;0m Your multi-generational Karoo GP run is complete.\n')
		sys.exit() # return Karoo GP to the command line to support bash and chron job execution
		
		# return
		
	
	def karoo_banner(self):
	
		'''
		This method makes Karoo GP look old-school cool!
		
		Arguments required: none
		'''
		
		os.system('clear')
		
		print('\n\033[36m\033[1m')
		print('\t **   **   ******    *****    ******    ******       ******   ******')
		print('\t **  **   **    **  **   **  **    **  **    **     **        **   **')
		print('\t ** **    **    **  **   **  **    **  **    **     **        **   **')
		print('\t ****     ********  ******   **    **  **    **     **   ***  ******')
		print('\t ** **    **    **  ** **    **    **  **    **     **    **  **')
		print('\t **  **   **    **  **  **   **    **  **    **     **    **  **')
		print('\t **   **  **    **  **   **  **    **  **    **     **    **  **')
		print('\t **    ** **    **  **    **  ******    ******       ******   **')
		print('\033[0;0m')
		print('\t\033[36m Genetic Programming in Python - by Kai Staats, version 1.0\033[0;0m')
				
		return
		
	
	def fx_karoo_data_load(self, tree_type, tree_depth_base, filename):
	
		'''
		The data and function .csv files are loaded according to the fitness function kernel selected by the user. An
		alternative dataset may be loaded at launch, by appending a command line argument. The data is then split into 
		both TRAINING and TEST segments in order to validate the success of the GP training run. Datasets less than
		10 rows will not be split, rather copied in full to both TRAINING and TEST as it is assumed you are conducting
		a system validation run, as with the built-in MATCH kernel and associated dataset.
		
		Arguments required: tree_type, tree_depth_base, filename (of the dataset)
		'''
		
		### 1) load the associated data set, operators, operands, fitness type, and coefficients ###
		
		full_path = os.path.realpath(__file__); cwd = os.path.dirname(full_path) # Good idea Marco :)
		# cwd = os.getcwd()
		
		data_dict = {'c':cwd + '/files/data_CLASSIFY.csv', 'r':cwd + '/files/data_REGRESS.csv', 'm':cwd + '/files/data_MATCH.csv', 'p':cwd + '/files/data_PLAY.csv'}
		
		if len(sys.argv) == 1: # load data from the default karoo_gp/files/ directory
			data_x = np.loadtxt(data_dict[self.kernel], skiprows = 1, delimiter = ',', dtype = float); data_x = data_x[:,0:-1] # load all but the right-most column
			data_y = np.loadtxt(data_dict[self.kernel], skiprows = 1, usecols = (-1,), delimiter = ',', dtype = float) # load only right-most column (class labels)
			header = open(data_dict[self.kernel],'r')
			self.dataset = data_dict[self.kernel]
			
		elif len(sys.argv) == 2: # load an external data file
			data_x = np.loadtxt(sys.argv[1], skiprows = 1, delimiter = ',', dtype = float); data_x = data_x[:,0:-1] # load all but the right-most column
			data_y = np.loadtxt(sys.argv[1], skiprows = 1, usecols = (-1,), delimiter = ',', dtype = float) # load only right-most column (class labels)
			header = open(sys.argv[1],'r')
			self.dataset = sys.argv[1]
			
		elif len(sys.argv) > 2: # receive filename and additional flags from karoo_gp_server.py via argparse
			data_x = np.loadtxt(filename, skiprows = 1, delimiter = ',', dtype = float); data_x = data_x[:,0:-1] # load all but the right-most column
			data_y = np.loadtxt(filename, skiprows = 1, usecols = (-1,), delimiter = ',', dtype = float) # load only right-most column (class labels)
			header = open(filename,'r')
			self.dataset = filename
			
		fitt_dict = {'c':'max', 'r':'min', 'm':'max', 'p':''}
		self.fitness_type = fitt_dict[self.kernel] # load fitness type
		
		func_dict = {'c':cwd + '/files/operators_CLASSIFY.csv', 'r':cwd + '/files/operators_REGRESS.csv', 'm':cwd + '/files/operators_MATCH.csv', 'p':cwd + '/files/operators_PLAY.csv'}
		self.functions = np.loadtxt(func_dict[self.kernel], delimiter=',', skiprows=1, dtype = str) # load the user defined functions (operators)
		self.terminals = header.readline().split(','); self.terminals[-1] = self.terminals[-1].replace('\n','') # load the user defined terminals (operands)
		self.class_labels = len(np.unique(data_y)) # load the user defined labels for classification or solutions for regression
		#self.coeff = np.loadtxt(cwd + '/files/coefficients.csv', delimiter=',', skiprows=1, dtype = str) # load the user defined coefficients - NOT USED YET
		
		
		### 2) from the dataset, extract TRAINING and TEST data ###
		
		if len(data_x) < 11: # for small datasets we will not split them into TRAINING and TEST components
			data_train = np.c_[data_x, data_y]
			data_test = np.c_[data_x, data_y]
			
		else: # if larger than 10, we run the data through the SciKit Learn's 'random split' function
			x_train, x_test, y_train, y_test = skcv.train_test_split(data_x, data_y, test_size = 0.2) # 80/20 TRAIN/TEST split
			data_x, data_y = [], [] # clear from memory
			
			data_train = np.c_[x_train, y_train] # recombine each row of data with its associated label (right column)
			x_train, y_train = [], [] # clear from memory
			
			data_test = np.c_[x_test, y_test] # recombine each row of data with its associated label (right column)
			x_test, y_test = [], [] # clear from memory
			
		self.data_train_cols = len(data_train[0,:]) # qty count
		self.data_train_rows = len(data_train[:,0]) # qty count
		self.data_test_cols = len(data_test[0,:]) # qty count
		self.data_test_rows = len(data_test[:,0]) # qty count
		
		
		### 3) load TRAINING and TEST data for TensorFlow processing - tested 2017 02/02
		
		self.data_train = data_train # Store train data for processing in TF
		self.data_test = data_test # Store test data for processing in TF
		self.tf_device = "/gpu:0" # Set TF computation backend device (CPU or GPU)
		self.tf_device_log = False # TF device usage logging (for debugging)
		
		
		### 4) create a unique directory and initialise all .csv files ###
		
		# self.datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
		self.datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
		self.path = os.path.join(cwd, 'runs/', filename.split('.')[0] + '_' + self.datetime) # generate a unique directory name
		# self.path = os.path.join(cwd, 'runs/', self.datetime) # generate a unique directory name
		if not os.path.isdir(self.path): os.makedirs(self.path) # make a unique directory
		
		self.filename = {} # a dictionary to hold .csv filenames
		
		self.filename.update( {'a':self.path + '/population_a.csv'} )
		target = open(self.filename['a'], 'w') # initialise the .csv file for population 'a' (foundation)
		target.close()
		
		self.filename.update( {'b':self.path + '/population_b.csv'} )
		target = open(self.filename['b'], 'w') # initialise the .csv file for population 'b' (evolving)
		target.close()
		
		self.filename.update( {'f':self.path + '/population_f.csv'} )
		target = open(self.filename['f'], 'w') # initialise the .csv file for the final population (test)
		target.close()
		
		self.filename.update( {'s':self.path + '/population_s.csv'} )
		# do NOT initialise this .csv file, as it is retained for loading a previous run (recover)
		
		return
		
	
	def fx_karoo_data_recover(self, population):
	
		'''
		This method is used to load a saved population of Trees, as invoked through the (pause) menu where population_s 
		replaces population_a in the /[path]/karoo_gp/runs/ directory.
		
		Arguments required: population size
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
						self.tree = np.array([[]]) # initialise Tree array
						
					else:
						if self.tree.shape[1] == 0:
							self.tree = np.append(self.tree, [row], axis = 1) # append first row to Tree
							
						else:
							self.tree = np.append(self.tree, [row], axis = 0) # append subsequent rows to Tree
							
					if self.tree.shape[0] == 13:
						self.population_a.append(self.tree) # append complete Tree to population list
						
		print(self.population_a)
		
		return
		
	
	def fx_karoo_construct(self, tree_type, tree_depth_base):
		
		'''
		As used by the method 'karoo_gp', this method constructs the initial population based upon the user-defined 
		Tree type and initial, maximum Tree depth ('tree_depth_base'). "Ramped half/half" was defined by John Koza as 
		a means of building maximum diversity in the initial population. There are equal numbers of Full and Grow 
		methods trees, and an equal spread of Trees across depths 1 to 'tree_depth_base'.
		
		Arguments required: tree_type, tree_depth_base
		'''
		
		if self.display == 'i' or self.display == 'g':
			print('\n\t Type \033[1m?\033[0;0m at any (pause) to review your options, or \033[1mENTER\033[0;0m to continue.\033[0;0m')
			self.fx_karoo_pause(0)
						
		if tree_type == 'r': # Ramped 50/50
			
			TREE_ID = 1
			for n in range(1, int((self.tree_pop_max // 2) / tree_depth_base) + 1): # split the population into equal parts
				for depth in range(1, tree_depth_base + 1): # build 2 Trees ats each depth
					self.fx_gen_tree_build(TREE_ID, 'f', depth) # build a Full Tree
					self.fx_archive_tree_append(self.tree) # append Tree to the list 'gp.population_a'
					TREE_ID = TREE_ID + 1
					
					self.fx_gen_tree_build(TREE_ID, 'g', depth) # build a Grow Tree
					self.fx_archive_tree_append(self.tree) # append Tree to the list 'gp.population_a'
					TREE_ID = TREE_ID + 1
						
			if TREE_ID < self.tree_pop_max: # eg: split 100 by 2*3 and it will produce only 96 Trees ...
				for n in range(self.tree_pop_max - TREE_ID + 1): # ... so we complete the run
					self.fx_gen_tree_build(TREE_ID, 'g', tree_depth_base)
					self.fx_archive_tree_append(self.tree)
					TREE_ID = TREE_ID + 1
					
			else: pass
									
		else: # Full or Grow
			for TREE_ID in range(1, self.tree_pop_max + 1):
				self.fx_gen_tree_build(TREE_ID, tree_type, tree_depth_base) # build the 1st generation of Trees
				self.fx_archive_tree_append(self.tree)
				
		return
		
	
	def fx_karoo_reproduce(self):
	
		'''
		Through tournament selection, a single Tree from the prior generation is copied without mutation to the next 
		generation. This is analogous to a member of the prior generation directly entering the gene pool of the 
		subsequent (younger) generation.
		
		Arguments required: none
		'''
		
		if self.display != 's':
			if self.display == 'i': print('')
			print('Perform', self.evolve_repro, 'Reproductions ...')
			if self.display == 'i': self.fx_karoo_pause(0)
			
		for n in range(self.evolve_repro): # quantity of Trees to be copied without mutation
			tourn_winner = self.fx_fitness_tournament(self.tourn_size) # perform tournament selection for each reproduction
			tourn_winner = self.fx_evolve_fitness_wipe(tourn_winner) # wipe fitness data
			self.population_b.append(tourn_winner) # append array to next generation population of Trees
			
		return
		
	
	def fx_karoo_point_mutate(self):
	
		'''
		Through tournament selection, a copy of a Tree from the prior generation mutates before being added to the 
		next generation. In this method, a single point is selected for mutation while maintaining function nodes as 
		functions (operators) and terminal nodes as terminals (variables). The size and shape of the Tree will remain 
		identical.
		
		Arguments required: none
		'''
		
		if self.display != 's':
			if self.display == 'i': print('')
			print('Perform', self.evolve_point, 'Point Mutations ...')
			if self.display == 'i': self.fx_karoo_pause(0)
			
		for n in range(self.evolve_point): # quantity of Trees to be generated through mutation
			tourn_winner = self.fx_fitness_tournament(self.tourn_size) # perform tournament selection for each mutation
			tourn_winner, node = self.fx_evolve_point_mutate(tourn_winner) # perform point mutation; return single point for record keeping
			self.population_b.append(tourn_winner) # append array to next generation population of Trees
						
		return
		
	
	def fx_karoo_branch_mutate(self):
	
		'''
		Through tournament selection, a copy of a Tree from the prior generation mutates before being added to the 
		next generation. Unlike Point Mutation, in this method an entire branch is selected. If the evolutionary run is 
		designated as Full, the size and shape of the Tree will remain identical, each node mutated sequentially, where 
		functions remain functions and terminals remain terminals. If the evolutionary run is designated as Grow or 
		Ramped Half/Half, the size and shape of the Tree may grow smaller or larger, but it may not exceed
		tree_depth_max as defined by the user.
		
		Arguments required: none
		'''
		
		if self.display != 's':
			if self.display == 'i': print('')
			print('Perform', self.evolve_branch, 'Full or Grow Mutations ...')
			if self.display == 'i': self.fx_karoo_pause(0)
			
		for n in range(self.evolve_branch): # quantity of Trees to be generated through mutation
			tourn_winner = self.fx_fitness_tournament(self.tourn_size) # perform tournament selection for each mutation
			branch = self.fx_evolve_branch_select(tourn_winner) # select point of mutation and all nodes beneath
			
			# TEST & DEBUG: comment the top or bottom to force all Full or all Grow methods
			
			if tourn_winner[1][1] == 'f': # perform Full method mutation on 'tourn_winner'
				tourn_winner = self.fx_evolve_full_mutate(tourn_winner, branch)
				
			elif tourn_winner[1][1] == 'g': # perform Grow method mutation on 'tourn_winner'
				tourn_winner = self.fx_evolve_grow_mutate(tourn_winner, branch)
				
			self.population_b.append(tourn_winner) # append array to next generation population of Trees
			
		return
		
	
	def fx_karoo_crossover(self):
	
		'''
		Through tournament selection, two trees are selected as parents to produce two offspring. Within each parent 
		Tree a branch is selected. Parent A is copied, with its selected branch deleted. Parent B's branch is then 
		copied to the former location of Parent A's branch and inserted (grafted). The size and shape of the child 
		Tree may be smaller or larger than either of the parents, but may not exceed 'tree_depth_max' as defined 
		by the user.
		
		This process combines genetic code from two parent Trees, both of which were chosen by the tournament process 
		as having a higher fitness than the average population. Therefore, there is a chance their offspring will 
		provide an improvement in total fitness. In most GP applications, Crossover is the most commonly applied 
		evolutionary operator (~70-80%).
		
		For those who like to watch, select 'db' (debug mode) at the launch of Karoo GP or at any (pause).
		
		Arguments required: none
		'''
		
		if self.display != 's':
			if self.display == 'i': print('')
			print('Perform', self.evolve_cross, 'Crossovers ...')
			if self.display == 'i': self.fx_karoo_pause(0)
			
		for n in range(self.evolve_cross // 2): # quantity of Trees to be generated through Crossover, accounting for 2 children each
		
			parent_a = self.fx_fitness_tournament(self.tourn_size) # perform tournament selection for 'parent_a'
			branch_a = self.fx_evolve_branch_select(parent_a) # select branch within 'parent_a', to copy to 'parent_b' and receive a branch from 'parent_b'
			
			parent_b = self.fx_fitness_tournament(self.tourn_size) # perform tournament selection for 'parent_b'
			branch_b = self.fx_evolve_branch_select(parent_b) # select branch within 'parent_b', to copy to 'parent_a' and receive a branch from 'parent_a'
			
			parent_c = np.copy(parent_a); branch_c = np.copy(branch_a) # else the Crossover mods affect the parent Trees, due to how Python manages '='
			parent_d = np.copy(parent_b); branch_d = np.copy(branch_b) # else the Crossover mods affect the parent Trees, due to how Python manages '='
			
			offspring_1 = self.fx_evolve_crossover(parent_a, branch_a, parent_b, branch_b) # perform Crossover
			self.population_b.append(offspring_1) # append the 1st child to next generation of Trees
			
			offspring_2 = self.fx_evolve_crossover(parent_d, branch_d, parent_c, branch_c) # perform Crossover
			self.population_b.append(offspring_2) # append the 2nd child to next generation of Trees
			
		return
		
	
	def fx_karoo_pause(self, eol):
	
		'''
		Pause the program execution and output to screen until the user selects a valid option. The "eol" parameter 
		instructs this method to display a different screen for run-time or end-of-line, and to dive back into the
		current run, or do nothing, accordingly.
		
		Arguments required: eol
		'''
		
		options = ['?','help','i','m','g','s','db','ts','min','max','bal','id','pop','dir','l','p','t','cont','load','w','q','']
		
		while True:
			try:
				pause = input('\n\t\033[36m (pause) \033[0;0m')
				if pause not in options: raise ValueError()
				if pause == '':
					if eol == 1: self.fx_karoo_pause(1) # return to pause menu as the GP run is complete
					else: break # drop back into the current GP run
					
				if pause == '?' or pause == 'help':
					print('\n\t\033[36mSelect one of the following options:\033[0;0m')
					print('\t\033[36m\033[1m i \t\033[0;0m Interactive display mode')
					print('\t\033[36m\033[1m m \t\033[0;0m Minimal display mode')
					print('\t\033[36m\033[1m g \t\033[0;0m Generation display mode')
					print('\t\033[36m\033[1m s \t\033[0;0m Silent display mode')
					print('\t\033[36m\033[1m db \t\033[0;0m De-Bug display mode')
					print('')
					print('\t\033[36m\033[1m ts \t\033[0;0m adjust the tournament size')
					print('\t\033[36m\033[1m min \t\033[0;0m adjust the minimum number of nodes')
					# print '\t\033[36m\033[1m max \t\033[0;0m adjust the maximum Tree depth'
					print('\t\033[36m\033[1m bal \t\033[0;0m adjust the balance of genetic operators')
					print('')
					print('\t\033[36m\033[1m l \t\033[0;0m list Trees with leading fitness scores')
					print('\t\033[36m\033[1m t \t\033[0;0m evaluate a single Tree against the test data')
					print('')
					print('\t\033[36m\033[1m p \t\033[0;0m print a single Tree to screen')
					print('\t\033[36m\033[1m id \t\033[0;0m display the current generation ID')
					print('\t\033[36m\033[1m pop \t\033[0;0m list all Trees in current population')
					print('\t\033[36m\033[1m dir \t\033[0;0m display current working directory')
					print('')
					print('\t\033[36m\033[1m cont \t\033[0;0m continue evolution, starting with the current population')
					print('\t\033[36m\033[1m load \t\033[0;0m load population_s (seed) to replace population_a (current)')
					print('\t\033[36m\033[1m w \t\033[0;0m write the evolving population_b to disk')
					print('\t\033[36m\033[1m q \t\033[0;0m quit Karoo GP without saving population_b')
					print('')
					
					if eol == 1: print('\t\033[0;0m Remember to archive your final population before your next run!')
					else: print('\t\033[36m\033[1m ENTER\033[0;0m to continue ...')
					
				elif pause == 'i': self.display = 'i'; print('\t Interactive display mode engaged (for control freaks)')
				elif pause == 'm': self.display = 'm'; print('\t Minimal display mode engaged (for recovering control freaks)')
				elif pause == 'g': self.display = 'g'; print('\t Generation display mode engaged (for GP gurus)')
				elif pause == 's': self.display = 's'; print('\t Silent display mode engaged (for zen masters)')
				elif pause == 'db': self.display = 'db'; print('\t De-Bug display mode engaged (for vouyers)')
				
				
				elif pause == 'ts': # adjust the tournament size
					menu = list(range(2,self.tree_pop_max + 1)) # set to total population size only for the sake of experimentation
					while True:
						try:
							print('\n\t The current tournament size is:', self.tourn_size)
							query = int(input('\t Adjust the tournament size (suggest 10): '))
							if query not in menu: raise ValueError()
							self.tourn_size = query; break
						except ValueError: print('\n\t\033[32m Enter a number from 2 including', str(self.tree_pop_max) + ".", 'Try again ...\033[0;0m')
						
				
				elif pause == 'min': # adjust the global, minimum number of nodes per Tree
					menu = list(range(3,1001)) # we must have at least 3 nodes, as in: x * y; 1000 is an arbitrary number
					while True:
						try:
							print('\n\t The current minimum number of nodes is:', self.tree_depth_min)
							query = int(input('\t Adjust the minimum number of nodes for all Trees (min 3): '))
							if query not in menu: raise ValueError()
							self.tree_depth_min = query; break
						except ValueError: print('\n\t\033[32m Enter a number from 3 including 1000. Try again ...\033[0;0m')
						
				
				# NEED TO ADD: adjustable tree_depth_max
				#elif pause == 'max': # adjust the global, adjusted maximum Tree depth
				#
				#	menu = range(1,11)
				#	while True:
				#		try:
				#			print '\n\t The current \033[3madjusted\033[0;0m maximum Tree depth is:', self.tree_depth_max
				#			query = int(raw_input('\n\t Adjust the global maximum Tree depth to (1 ... 10): '))
				#			if query not in menu: raise ValueError()
				#			if query < self.tree_depth_max:
				#				print '\n\t\033[32m This value is less than the current value.\033[0;0m'
				#				conf = raw_input('\n\t Are you ok with this? (y/n) ')
				#				if conf == 'n': break
				#		except ValueError: print '\n\t\033[32m Enter a number from 1 including 10. Try again ...\033[0;0m'
						
				
				elif pause == 'bal': # adjust the balance of genetic operators'					
					print('\n\t The current balance of genetic operators is:')
					print('\t\t Reproduction:', self.evolve_repro); tmp_repro = self.evolve_repro
					print('\t\t Point Mutation:', self.evolve_point); tmp_point = self.evolve_point
					print('\t\t Branch Mutation:', self.evolve_branch); tmp_branch = self.evolve_branch
					print('\t\t Crossover:', self.evolve_cross, '\n'); tmp_cross = self.evolve_cross
					
					menu = list(range(0,1000)) # 0 to 1000 expresssed as an integer
					
					while True:
						try:
							query = input('\t Enter quantity of Trees to be generated by Reproduction: ')
							if query not in str(menu): raise ValueError()
							elif query == '': break
							tmp_repro = int(float(query)); break
						except ValueError: print('\n\t\033[32m Enter a number from 0 including 1000. Try again ...\033[0;0m')
						
					while True:
						try:
							query = input('\t Enter quantity of Trees to be generated by Point Mutation: ')
							if query not in str(menu): raise ValueError()
							elif query == '': break
							tmp_point = int(float(query)); break
						except ValueError: print('\n\t\033[32m Enter a number from 0 including 1000. Try again ...\033[0;0m')
						
					while True:
						try:
							query = input('\t Enter quantity of Trees to be generated by Branch Mutation: ')
							if query not in str(menu): raise ValueError()
							elif query == '': break
							tmp_branch = int(float(query)); break
						except ValueError: print('\n\t\033[32m Enter a number from 0 including 1000. Try again ...\033[0;0m')
						
					while True:
						try:
							query = input('\t Enter quantity of Trees to be generated by Crossover: ')
							if query not in str(menu): raise ValueError()
							elif query == '': break
							tmp_cross = int(float(query)); break
						except ValueError: print('\n\t\033[32m Enter a number from 0 including 1000. Try again ...\033[0;0m')
						
					if tmp_repro + tmp_point + tmp_branch + tmp_cross != self.tree_pop_max: print('\n\t The sum of the above does not equal', self.tree_pop_max, 'Try again ...')
					else:
						print('\n\t The revised balance of genetic operators is:')
						self.evolve_repro = tmp_repro; print('\t\t Reproduction:', self.evolve_repro)
						self.evolve_point = tmp_point; print('\t\t Point Mutation:', self.evolve_point)
						self.evolve_branch = tmp_branch; print('\t\t Branch Mutation:', self.evolve_branch)
						self.evolve_cross = tmp_cross; print('\t\t Crossover:', self.evolve_cross)
						
						
				elif pause == 'l': # display dictionary of Trees with the best fitness score
					print('\n\t The leading Trees and their associated expressions are:')
					for n in sorted(self.fittest_dict): print('\t ', n, ':', self.fittest_dict[n])
					
										
				elif pause == 't':  # evaluate a Tree against the TEST data
					if self.generation_id > 1:
						menu = list(range(1, len(self.population_b)))
						while True:
							try:
								query = input('\n\t Select a Tree in population_b to test: ')
								if query not in str(menu) or query == '0': raise ValueError()
								elif query == '': break
								
								self.fx_eval_poly(self.population_b[int(query)]) # generate the raw and sympified equation for the given Tree using SymPy
								
								# get simplified expression and process it by TF - tested 2017 02/02
								expr = str(self.algo_sym) # might change this to algo_raw for more correct expression evaluation
								result = self.fx_fitness_eval(expr, self.data_test, get_labels=True)
								
								print('\n\t\033[36mTree', query, 'yields (raw):', self.algo_raw, '\033[0;0m')
								print('\t\033[36mTree', query, 'yields (sym):\033[1m', self.algo_sym, '\033[0;0m\n')
								
								# test user selected Trees using TF - tested 2017 02/02
								if self.kernel == 'c': self.fx_fitness_test_classify(result); break
								elif self.kernel == 'r': self.fx_fitness_test_regress(result); break
								elif self.kernel == 'm': self.fx_fitness_test_match(result); break
								# elif self.kernel == '[other]': self.fx_fitness_test_[other](result); break
								
							except ValueError: print('\n\t\033[32m Enter a number from 1 including', str(len(self.population_b) - 1) + ".", 'Try again ...\033[0;0m')
							
					else: print('\n\t\033[32m Karoo GP does not enable evaluation of the foundation population. Be patient ...\033[0;0m')
					
				
				elif pause == 'p': # print a Tree to screen -- NEED TO ADD: SymPy graphical print option
					if self.generation_id == 1:
						menu = list(range(1,len(self.population_a)))
						while True:
							try:
								query = input('\n\t Select a Tree to print: ')
								if query not in str(menu) or query == '0': raise ValueError()
								elif query == '': break
								self.fx_display_tree(self.population_a[int(query)]); break
							except ValueError: print('\n\t\033[32m Enter a number from 1 including', str(len(self.population_a) - 1) + ".", 'Try again ...\033[0;0m')
							
					elif self.generation_id > 1:
						menu = list(range(1,len(self.population_b)))
						while True:
							try:
								query = input('\n\t Select a Tree to print: ')
								if query not in str(menu) or query == '0': raise ValueError()
								elif query == '': break
								self.fx_display_tree(self.population_b[int(query)]); break
							except ValueError: print('\n\t\033[32m Enter a number from 1 including', str(len(self.population_b) - 1) + ".", 'Try again ...\033[0;0m')
							
					else: print('\n\t\033[36m There is nor forest for which to see the Trees.\033[0;0m')
					
				
				elif pause == 'id': print('\n\t The current generation is:', self.generation_id)
				
				
				elif pause == 'pop': # list Trees in the current population
					print('')
					if self.generation_id == 1:
						for tree_id in range(1, len(self.population_a)):
							self.fx_eval_poly(self.population_a[tree_id]) # extract the expression
							print('\t\033[36m Tree', self.population_a[tree_id][0][1], 'yields (sym):\033[1m', self.algo_sym, '\033[0;0m')
							
					elif self.generation_id > 1:
						for tree_id in range(1, len(self.population_b)):
							self.fx_eval_poly(self.population_b[tree_id]) # extract the expression
							print('\t\033[36m Tree', self.population_b[tree_id][0][1], 'yields (sym):\033[1m', self.algo_sym, '\033[0;0m')
							
					else: print('\n\t\033[36m There is nor forest for which to see the Trees.\033[0;0m')
					
					
				elif pause == 'dir': print('\n\t The current working directory is:', self.path)
				
									
				elif pause == 'cont': # continue evolution, starting with the current population
					menu = list(range(1,101))
					while True:
						try:
							query = input('\n\t How many more generations would you like to add? (1-100): ')
							if query not in str(menu) or query == '0': raise ValueError()
							elif query == '': break
							self.generation_max = self.generation_max + int(query)
							next_gen_start = self.generation_id + 1
							self.fx_karoo_continue(next_gen_start) # continue evolving, starting with the last population
						except ValueError: print('\n\t\033[32m Enter a number from 1 including 100. Try again ...\033[0;0m')
					
				
				elif pause == 'load': # load population_s to replace population_a
					while True:
						try:
							query = input('\n\t Overwrite the current population with population_s? ')
							if query not in ['y','n']: raise ValueError()
							if query == 'y': self.fx_karoo_data_recover(self.filename['s']); break
							elif query == 'n': break
						except ValueError: print('\n\t\033[32m Enter (y)es or (n)o. Try again ...\033[0;0m')
						
				
				elif pause == 'w': # write the evolving population_b to disk
					if self.generation_id > 1:
						self.fx_archive_tree_write(self.population_b, 'b')
						print('\t\033[36m All current members of the evolving population_b saved to .csv\033[0;0m')
						
					else: print('\n\t\033[36m The evolving population_b does not yet exist\033[0;0m')
					
				
				elif pause == 'q':
					if eol == 0: # if the GP run is not at the final generation
						query = input('\n\t \033[32mThe current population_b will be lost!\033[0;0m\n\n\t Are you certain you want to quit? (y/n)')
						if query == 'y':
							self.fx_archive_params_write('Desktop') # save run-time parameters to disk
							sys.exit() # quit the script without saving population_b
						else: break
						
					else: # if the GP run is complete
						query = input('\n\t Are you certain you want to quit? (y/n)')
						if query == 'y':
							print('\n\t \033[32mYour Trees and runtime parameters are archived in karoo_gp/runs/\033[0;0m')
							self.fx_archive_params_write('Desktop') # save run-time parameters to disk
							sys.exit()
						else: self.fx_karoo_pause(1)
						
			except ValueError: print('\t\033[32m Select from the options given. Try again ...\033[0;0m')
			except KeyboardInterrupt: print('\n\t\033[32m Enter q to quit\033[0;0m')
			
		return
		
	
	def fx_karoo_continue(self, next_gen_start):
	
		'''
		This method enables the launch of another full run of Karoo GP, but starting with a seed generation
		instead of with a randomly generated first population. This can be used at the end of a standard run to
		continue the evoluationary process, or after having recovered a set of trees from a prior run.
		
		Arguments required: next_gen_start
		'''
		
		for self.generation_id in range(next_gen_start, self.generation_max + 1): # evolve additional generations of Trees
		
			print('\n Evolve a population of Trees for Generation', self.generation_id, '...')
			self.population_b = ['Karoo GP by Kai Staats, Evolving Generation'] # initialise population_b to host the next generation
			
			self.fx_fitness_gene_pool() # generate the viable gene pool (compares against gp.tree_depth_min)
			self.fx_karoo_reproduce() # method 1 - Reproduction
			self.fx_karoo_point_mutate() # method 2 - Point Mutation
			self.fx_karoo_branch_mutate() # method 3 - Branch Mutation
			self.fx_karoo_crossover() # method 4 - Crossover
			self.fx_eval_generation() # evaluate all Trees in a single generation
			
			self.population_a = self.fx_evolve_pop_copy(self.population_b, ['Karoo GP by Kai Staats, Generation ' + str(self.generation_id)])
			
		# "End of line, man!" --CLU
		target = open(self.filename['f'], 'w') # reset the .csv file for the final population
		target.close()
		
		self.fx_archive_tree_write(self.population_b, 'f') # save the final generation of Trees to disk
		self.fx_karoo_eol()
		
		return
		
	
	def fx_karoo_eol(self):
		
		'''
		The very last method to run in Karoo GP.
		
		Arguments required: none
		'''
		
		print('\n\033[3m "It is not the strongest of the species that survive, nor the most intelligent,\033[0;0m')
		print('\033[3m  but the one most responsive to change."\033[0;0m --Charles Darwin')
		print('')
		print('\033[3m Congrats!\033[0;0m Your multi-generational Karoo GP run is complete.\n')
		print('\033[36m Type \033[1m?\033[0;0m\033[36m to review your options or \033[1mq\033[0;0m\033[36m to quit.\033[0;0m\n')
		self.fx_karoo_pause(1)
		
		return
		
	
	#++++++++++++++++++++++++++++++++++++++++++
	#   Methods to Generate a new Tree        |
	#++++++++++++++++++++++++++++++++++++++++++
	
	def fx_gen_tree_initialise(self, TREE_ID, tree_type, tree_depth_base):

		'''
		Assign 13 global variables to the array 'tree'.
		
		Build the array 'tree' with 13 rows and initally, just 1 column of labels. This array will grow as each new 
		node is appended. The values of this array are stored as string characters. Numbers will be forced to integers
		at the point of execution.
		
		This method is called by 'fx_gen_tree_build'.
		
		Arguments required: TREE_ID, tree_type, tree_depth_base
		'''
		
		self.pop_TREE_ID = TREE_ID 			# pos 0: a unique identifier for each tree
		self.pop_tree_type = tree_type	# pos 1: a global constant based upon the initial user setting
		self.pop_tree_depth_base = tree_depth_base	# pos 2: a global variable which conveys 'tree_depth_base' as unique to each new Tree
		self.pop_NODE_ID = 1 						# pos 3: unique identifier for each node; this is the INDEX KEY to this array
		self.pop_node_depth = 0 				# pos 4: depth of each node when committed to the array
		self.pop_node_type = '' 				# pos 5: root, function, or terminal
		self.pop_node_label = '' 				# pos 6: operator [+, -, *, ...] or terminal [a, b, c, ...]
		self.pop_node_parent = '' 			# pos 7: parent node
		self.pop_node_arity = '' 				# pos 8: number of nodes attached to each non-terminal node
		self.pop_node_c1 = '' 					# pos 9: child node 1
		self.pop_node_c2 = '' 					# pos 10: child node 2
		self.pop_node_c3 = '' 					# pos 11: child node 3 (assumed max of 3 with boolean operator 'if')
		self.pop_fitness = ''						# pos 12: fitness score following Tree evaluation
		
		self.tree = np.array([ ['TREE_ID'],['tree_type'],['tree_depth_base'],['NODE_ID'],['node_depth'],['node_type'],['node_label'],['node_parent'],['node_arity'],['node_c1'],['node_c2'],['node_c3'],['fitness'] ])
		
		return
				
	
	### Root Node ###
	
	def fx_gen_root_node_build(self):
	
		'''
		Build the Root node for the initial population.
		
		This method is called by 'fx_gen_tree_build'.
		
		Arguments required: none
		'''
				
		self.fx_gen_function_select() # select the operator for root
		
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

		else: print('\n\t\033[31m ERROR! In fx_gen_root_node_build: pop_node_arity =', self.pop_node_arity, '\033[0;0m'); self.fx_karoo_pause(0)

		self.pop_node_type = 'root'
			
		self.fx_gen_node_commit()

		return
		
	
	### Function Nodes ###
	
	def fx_gen_function_node_build(self):
	
		'''
		Build the Function nodes for the intial population.
		
		This method is called by 'fx_gen_tree_build'.
		
		Arguments required: none
		'''
		
		for i in range(1, self.pop_tree_depth_base): # increment depth, from 1 through 'tree_depth_base' - 1
		
			self.pop_node_depth = i # increment 'node_depth'
			
			parent_arity_sum = 0
			prior_sibling_arity = 0 # reset for 'c_buffer' in 'children_link'
			prior_siblings = 0 # reset for 'c_buffer' in 'children_link'
			
			for j in range(1, len(self.tree[3])): # increment through all nodes (exclude 0) in array 'tree'
			
				if int(self.tree[4][j]) == self.pop_node_depth - 1: # find parent nodes which reside at the prior depth
					parent_arity_sum = parent_arity_sum + int(self.tree[8][j]) # sum arities of all parent nodes at the prior depth
					
					# (do *not* merge these 2 "j" loops or it gets all kinds of messed up)
										
			for j in range(1, len(self.tree[3])): # increment through all nodes (exclude 0) in array 'tree'
			
				if int(self.tree[4][j]) == self.pop_node_depth - 1: # find parent nodes which reside at the prior depth

					for k in range(1, int(self.tree[8][j]) + 1): # increment through each degree of arity for each parent node
						self.pop_node_parent = int(self.tree[3][j]) # set the parent 'NODE_ID' ...
						prior_sibling_arity = self.fx_gen_function_gen(parent_arity_sum, prior_sibling_arity, prior_siblings) # ... generate a Function ndoe
						prior_siblings = prior_siblings + 1 # sum sibling nodes (current depth) who will spawn their own children (cousins? :)
												
		return
		
	
	def fx_gen_function_gen(self, parent_arity_sum, prior_sibling_arity, prior_siblings):
	
		'''
		Generate a single Function node for the initial population.
		
		This method is called by 'fx_gen_function_node_build'.
		
		Arguments required: parent_arity_sum, prior_sibling_arity, prior_siblings
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
		Define a single Function (operator extracted from the associated functions.csv) for the initial population.
		
		This method is called by 'fx_gen_function_gen' and 'fx_gen_root_node_build'.
		
		Arguments required: none
		'''
		
		self.pop_node_type = 'func'
		rnd = np.random.randint(0, len(self.functions[:,0])) # call the previously loaded .csv which contains all operators
		self.pop_node_label = self.functions[rnd][0]
		self.pop_node_arity = int(self.functions[rnd][1])
		
		return
		
	
	### Terminal Nodes ###
	
	def fx_gen_terminal_node_build(self):
	
		'''
		Build the Terminal nodes for the intial population.
		
		This method is called by 'fx_gen_tree_build'.
		
		Arguments required: none
		'''
			
		self.pop_node_depth = self.pop_tree_depth_base # set the final node_depth (same as 'gp.pop_node_depth' + 1)
		
		for j in range(1, len(self.tree[3]) ): # increment through all nodes (exclude 0) in array 'tree'
		
			if int(self.tree[4][j]) == self.pop_node_depth - 1: # find parent nodes which reside at the prior depth
			
				for k in range(1,(int(self.tree[8][j]) + 1)): # increment through each degree of arity for each parent node
					self.pop_node_parent = int(self.tree[3][j]) # set the parent 'NODE_ID'  ...
					self.fx_gen_terminal_gen() # ... generate a Terminal node
					
		return
		
	
	def fx_gen_terminal_gen(self):
	
		'''
		Generate a single Terminal node for the initial population.
		
		This method is called by 'fx_gen_terminal_node_build'.
		
		Arguments required: none
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
		
		This method is called by 'fx_gen_terminal_gen' and 'fx_gen_function_gen'.
		
		Arguments required: none
		'''
				
		self.pop_node_type = 'term'
		rnd = np.random.randint(0, len(self.terminals) - 1) # call the previously loaded .csv which contains all terminals
		self.pop_node_label = self.terminals[rnd]
		self.pop_node_arity = 0
		
		return
		
	
	### The Lovely Children ###
			
	def fx_gen_child_link(self, parent_arity_sum, prior_sibling_arity, prior_siblings):
	
		'''
		Link each parent node to its children in the intial population.
		
		This method is called by 'fx_gen_function_gen'.
		
		Arguments required: parent_arity_sum, prior_sibling_arity, prior_siblings
		'''
		
		c_buffer = 0
		
		for n in range(1, len(self.tree[3]) ): # increment through all nodes (exclude 0) in array 'tree'
		
			if int(self.tree[4][n]) == self.pop_node_depth - 1: # find all nodes that reside at the prior (parent) 'node_depth'
			
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
					
				else: print('\n\t\033[31m ERROR! In fx_gen_child_link: pop_node_arity =', self.pop_node_arity, '\033[0;0m'); self.fx_karoo_pause(0)
					
		return
		
	
	def fx_gen_node_commit(self):
	
		'''
		Commit the values of a new node (root, function, or terminal) to the array 'tree'.
		
		This method is called by 'fx_gen_root_node_build' and 'fx_gen_function_gen' and 'fx_gen_terminal_gen'.
		
		Arguments required: none
		'''
		
		self.tree = np.append(self.tree, [ [self.pop_TREE_ID],[self.pop_tree_type],[self.pop_tree_depth_base],[self.pop_NODE_ID],[self.pop_node_depth],[self.pop_node_type],[self.pop_node_label],[self.pop_node_parent],[self.pop_node_arity],[self.pop_node_c1],[self.pop_node_c2],[self.pop_node_c3],[self.pop_fitness] ], 1)
		
		self.pop_NODE_ID = self.pop_NODE_ID + 1
		
		return
		
	
	def fx_gen_tree_build(self, TREE_ID, tree_type, tree_depth_base):
	
		'''
		This method combines 4 sub-methods into a single method for ease of deployment. It is designed to executed 
		within a loop such that an entire population is built. However, it may also be run from the command line, 
		passing a single TREE_ID to the method.
		
		'tree_type' is either (f)ull or (g)row. Note, however, that when the user selects 'ramped 50/50' at launch, 
		it is still (f) or (g) which are passed to this method.
		
		This method is called by a 'fx_evolve_crossover' and 'fx_evolve_grow_mutate' and 'fx_karoo_construct'.
		
		Arguments required: TREE_ID, tree_type, tree_depth_base
		'''
		
		self.fx_gen_tree_initialise(TREE_ID, tree_type, tree_depth_base) # initialise a new Tree
		self.fx_gen_root_node_build() # build the Root node
		self.fx_gen_function_node_build() # build the Function nodes
		self.fx_gen_terminal_node_build() # build the Terminal nodes
		
		return # each Tree is written to 'gp.tree'
		
	
	#++++++++++++++++++++++++++++++++++++++++++
	#   Methods to Evaluate a Tree            |
	#++++++++++++++++++++++++++++++++++++++++++		
	
	def fx_eval_poly(self, tree):
	
		'''
		Evaluate a Tree and generate its multivariate expression (both raw and Sympified).
				
		We need to extract the variables from the expression. However, these variables are no longer correlated
		to the original variables listed across the top of each column of data.csv. Therefore, we must re-assign 
		the respective values for each subsequent row in the data .csv, for each Tree's unique expression.
		
		Arguments required: tree
		'''
		
		self.algo_raw = self.fx_eval_label(tree, 1) # pass the root 'node_id', then flatten the Tree to a string
		self.algo_sym = sympify(self.algo_raw) # convert string to a functional expression (the coolest line in Karoo! :)
		
		return
				
	
	def fx_eval_label(self, tree, node_id):
	
		'''
		Evaluate all or part of a Tree (starting at node_id) and return a raw mutivariate expression ('algo_raw').
		
		In the main code, this method is called once per Tree, but may be called at any time to prepare an expression 
		for any full or partial (branch) Tree contained in 'population'.
		
		Pass the starting node for recursion via the local variable 'node_id' where the local variable 'tree' is a 
		copy of the Tree you desire to evaluate.
		
		Arguments required: tree, node_id
		'''
		
		# if tree[6, node_id] == 'not': tree[6, node_id] = ', not' # temp until this can be fixed at data_load
		
		node_id = int(node_id)
		
		if tree[8, node_id] == '0': # arity of 0 for the pattern '[term]'
			return '(' + tree[6, node_id] + ')' # 'node_label' (function or terminal)
			
		else:
			if tree[8, node_id] == '1': # arity of 1 for the explicit pattern 'not [term]'
				return self.fx_eval_label(tree, tree[9, node_id]) + tree[6, node_id] # original code
				
			elif tree[8, node_id] == '2': # arity of 2 for the pattern '[func] [term] [func]'
				return self.fx_eval_label(tree, tree[9, node_id]) + tree[6, node_id] + self.fx_eval_label(tree, tree[10, node_id])
				
			elif tree[8, node_id] == '3': # arity of 3 for the explicit pattern 'if [term] then [term] else [term]'
				return tree[6, node_id] + self.fx_eval_label(tree, tree[9, node_id]) + ' then ' + self.fx_eval_label(tree, tree[10, node_id]) + ' else ' + self.fx_eval_label(tree, tree[11, node_id])
						
	
	def fx_eval_id(self, tree, node_id):
	
		'''
		Evaluate all or part of a Tree and return a list of all 'NODE_ID's.
	
		This method generates a list of all 'NODE_ID's from the given Node and below. It is used primarily to generate 
		'branch' for the multi-generational mutation of Trees.
	
		Pass the starting node for recursion via the local variable 'node_id' where the local variable 'tree' is a copy 
		of the Tree you desire to evaluate.
		
		Arguments required: tree, node_id	
		'''
		
		node_id = int(node_id)
		
		if tree[8, node_id] == '0': # arity of 0 for the pattern '[NODE_ID]'
			return tree[3, node_id] # 'NODE_ID'
			
		else:
			if tree[8, node_id] == '1': # arity of 1 for the pattern '[NODE_ID], [NODE_ID]'
				return tree[3, node_id] + ', ' + self.fx_eval_id(tree, tree[9, node_id])
				
			elif tree[8, node_id] == '2': # arity of 2 for the pattern '[NODE_ID], [NODE_ID], [NODE_ID]'
				return tree[3, node_id] + ', ' + self.fx_eval_id(tree, tree[9, node_id]) + ', ' + self.fx_eval_id(tree, tree[10, node_id])
				
			elif tree[8, node_id] == '3': # arity of 3 for the pattern '[NODE_ID], [NODE_ID], [NODE_ID], [NODE_ID]'
				return tree[3, node_id] + ', ' + self.fx_eval_id(tree, tree[9, node_id]) + ', ' + self.fx_eval_id(tree, tree[10, node_id]) + ', ' + self.fx_eval_id(tree, tree[11, node_id])
				
	
	def fx_eval_generation(self):
	
		'''
		This method invokes the evaluation of an entire generation of Trees, as engaged by karoo_gp_server.py and the 
		'cont' function of karoo_go_main.py. It automatically evaluates population_b before invoking the copy of _b to _a.
				
		Arguments required: none
		'''
		
		if self.display != 's':
			if self.display == 'i': print('')
			print('\n Evaluate all Trees in Generation', self.generation_id)
			if self.display == 'i': self.fx_karoo_pause(0)
			
		self.fx_evolve_tree_renum(self.population_b) # population renumber
		self.fx_fitness_gym(self.population_b) # run 'fx_eval', 'fx_fitness', 'fx_fitness_store', and fitness record
		self.fx_archive_tree_write(self.population_b, 'a') # archive current population as foundation for next generation
		
		if self.display != 's':
			print('\n Copy gp.population_b to gp.population_a\n')
			
		return
		
	
	#++++++++++++++++++++++++++++++++++++++++++
	#   Methods to Train and Test a Tree      |
	#++++++++++++++++++++++++++++++++++++++++++
	
	def fx_fitness_gym(self, population):
	
		'''		
		Part 1 evaluates each expression against the data, line for line. This is the most time consuming and
		computationally expensive part of genetic programming. When GPUs are available, the performance can increase
		by many orders of magnitude for datasets measured in millions of data.
		
		Part 2 evaluates every Tree in each generation to determine which have the best, overall fitness score. This 
		could be the highest or lowest depending upon if the fitness function is maximising (higher is better) or 
		minimising (lower is better). The total fitness score is then saved with each Tree in the external .csv file.
		
		Part 3 compares the fitness of each Tree to the prior best fit in order to track those that improve with each
		comparison. For matching functions, all the Trees will have the same fitness score, but they may present more 
		than one solution. For minimisation and maximisation functions, the final Tree should present the best overall 
		fitness for that generation. It is important to note that Part 3 does *not* in any way influence the Tournament 
		Selection which is a stand-alone process.
		
		Arguments required: population
		'''
		
		fitness_best = 0
		self.fittest_dict = {}
		time_sum = 0
		
		for tree_id in range(1, len(population)):
		
			### PART 1 - GENERATE MULTIVARIATE EXPRESSION FOR EACH TREE ###
			self.fx_eval_poly(population[tree_id]) # extract the expression
			if self.display not in ('s'): print('\t\033[36mTree', population[tree_id][0][1], 'yields (sym):\033[1m', self.algo_sym, '\033[0;0m')
			
			
			### PART 2 - EVALUATE FITNESS FOR EACH TREE AGAINST TRAINING DATA ###
			fitness = 0
			
			expr = str(self.algo_sym) # get sympified expression and process it with TF - tested 2017 02/02
			result = self.fx_fitness_eval(expr, self.data_train)
			fitness = result['fitness'] # extract fitness score
			
			if self.display == 'i':
				print('\t \033[36m with fitness sum:\033[1m', fitness, '\033[0;0m\n')
				
			self.fx_fitness_store(population[tree_id], fitness) # store Fitness with each Tree
			
			
			### PART 3 - COMPARE FITNESS OF ALL TREES IN CURRENT GENERATION ###
			if self.kernel == 'c': # display best fit Trees for the CLASSIFY kernel
				if fitness >= fitness_best: # find the Tree with Maximum fitness score
					fitness_best = fitness # set best fitness score
					self.fittest_dict.update({tree_id:self.algo_sym}) # add to dictionary if fitness >= prior
					
			elif self.kernel == 'r': # display best fit Trees for the REGRESSION kernel
				if fitness_best == 0: fitness_best = fitness # set the baseline first time through
				if fitness <= fitness_best: # find the Tree with Minimum fitness score
					fitness_best = fitness # set best fitness score
					self.fittest_dict.update({tree_id:self.algo_sym}) # add to dictionary if fitness <= prior
					
			elif self.kernel == 'm': # display best fit Trees for the MATCH kernel
				if fitness == self.data_train_rows: # find the Tree with a perfect match for all data rows
					fitness_best = fitness # set best fitness score
					self.fittest_dict.update({tree_id:self.algo_sym}) # add to dictionary if all rows match
					
			# elif self.kernel == '[other]': # display best fit Trees for the [other] kernel
				# if fitness [>=, <=] fitness_best: # find the Tree with [Maximum or Minimum] fitness score
					# fitness_best = fitness # set best fitness score
					# self.fittest_dict.update({tree_id:self.algo_sym}) # add to dictionary
			
		print('\n\033[36m ', len(list(self.fittest_dict.keys())), 'trees\033[1m', np.sort(list(self.fittest_dict.keys())), '\033[0;0m\033[36moffer the highest fitness scores.\033[0;0m')
		if self.display == 'g': self.fx_karoo_pause(0)
		
		return
		
	
	def fx_fitness_eval(self, expr, data, get_labels = False):
	
		'''		
		Computes tree expression using TensorFlow (TF) returning results and fitness scores.
		
		This method orchestrates most of the TF routines by parsing input string expression and converting it into TF 
		operation graph which then is processed in an isolated TF session to compute the results and corresponding fitness 
		values. 
		
			'self.tf_device' - controls which device will be used for computations (CPU or GPU).
			'self.tf_device_log' - controls device placement logging (debug only).

		Args:
			'expr' - a string containing math expression to be computed on the data. Variable names should match corresponding 
			terminal names in 'self.terminals'. Only algebraic operations are currently supported (+, -, *, /, **).
			
			'data' - an 'n by m' matrix of the data points containing n observations and m features each. Variable order should 
			match corresponding order of terminals in 'self.terminals'.

			'get_labels' - a boolean flag which controls whether classification labels should be extracted from the results.
			This is applied only to the CLASSIFY kernel and defaults to 'False'.

		Returns:
			A dict mapping keys to the following outputs:
				'result' - an array of the results of applying given expression to the data
				'labels' - an array of the labels extracted from the results; defined only for CLASSIFY kernel, None otherwise
				'solution' - an array of the solution values extracted from the data (variable 's' in the dataset)
				'pairwise_fitness' - an array of the element-wise results of applying corresponding fitness kernel function
				'fitness' - aggregated scalar fitness score
				
		Arguments required: expr, data
		'''
		
		# Initialize TensorFlow session
		tf.reset_default_graph() # Reset TF internal state and cache (after previous processing)
		config = tf.ConfigProto(log_device_placement=self.tf_device_log, allow_soft_placement=True)
		config.gpu_options.allow_growth = True
		
		with tf.Session(config=config) as sess:
			with sess.graph.device(self.tf_device):
			
				# 1 - Load data into TF
				tensors = {}
				for i in range(len(self.terminals)):
					var = self.terminals[i]
					tensors[var] = tf.constant(data[:, i], dtype=tf.float32)
					
				# 2- Transform string expression into TF operation graph
				result = self.fx_fitness_expr_parse(expr, tensors)
				
				labels = tf.no_op() # a placeholder, applies only to CLASSIFY kernel
				solution = tensors['s'] # solution value is assumed to be stored in 's' terminal
				
				# 3- Add fitness computation into TF graph
				if self.kernel == 'c': # CLASSIFY kernels
				
					'''
					Creates element-wise fitness computation TensorFlow (TF) sub-graph for CLASSIFY kernel.
					
					This method uses the 'sympified' (SymPy) expression ('algo_sym') created in 'fx_eval_poly' and the data set 
					loaded at run-time to evaluate the fitness of the selected kernel.
					
					This multiclass classifer compares each row of a given Tree to the known solution, comparing estimated values 
					(labels) generated by Karoo GP against the correct labels. This method is able to work with any number of 
					class labels, from 2 to n. The left-most bin includes -inf. The right-most bin includes +inf. Those inbetween 
					are by default confined to the spacing of 1.0 each, as defined by:
					
						(solution - 1) < result <= solution
					
					The skew adjusts the boundaries of the bins such that they fall on both the negative and positive sides of the 
					origin. At the time of this writing, an odd number of class labels will generate an extra bin on the positive 
					side of origin as it has not yet been determined the effect of enabling the middle bin to include both a 
					negative and positive space.
					
					Arguments required: result, solution		
					'''
					
					if get_labels: labels = tf.map_fn(self.fx_fitness_labels_map, result, dtype=(tf.int32, tf.string), swap_memory=True)
					
					skew = (self.class_labels // 2) - 1
					
					rule11 = tf.equal(solution, 0)
					rule12 = tf.less_equal(result, 0 - skew)
					rule13 = tf.logical_and(rule11, rule12)
					
					rule21 = tf.equal(solution, self.class_labels - 1)
					rule22 = tf.greater(result, solution - 1 - skew)
					rule23 = tf.logical_and(rule21, rule22)
					
					rule31 = tf.less(solution - 1 - skew, result)
					rule32 = tf.less_equal(result, solution - skew)
					rule33 = tf.logical_and(rule31, rule32)
					
					pairwise_fitness = tf.cast(tf.logical_or(tf.logical_or(rule13, rule23), rule33), tf.int32)
					
				elif self.kernel == 'r': # REGRESSION kernel
					pairwise_fitness = tf.abs(solution - result)
					
				elif self.kernel == 'm': # MATCH kernel
					# pairwise_fitness = tf.cast(tf.equal(solution, result), tf.int32) # breaks due to floating points
					RTOL, ATOL = 1e-05, 1e-08
					pairwise_fitness = tf.cast(tf.less_equal(tf.abs(solution - result), ATOL + RTOL * tf.abs(result)), tf.int32)
					
				# elif self.kernel == '[other]': # [OTHER] kernel
					# pairwise_fitness = tf.cast(tf.___(solution, result)
					
				else: raise Exception('Kernel type is wrong or missing. You entered {}'.format(self.kernel))
				
				fitness = tf.reduce_sum(pairwise_fitness)
				
				# Process TF graph and collect the results
				result, labels, solution, fitness, pairwise_fitness = sess.run([result, labels, solution, fitness, pairwise_fitness])
				
		return {'result': result, 'labels': labels, 'solution': solution, 'fitness': float(fitness), 'pairwise_fitness': pairwise_fitness}
		
	
	def fx_fitness_expr_parse(self, expr, tensors):
	
		'''		
		Extract expression tree from the string algo_sym and transform into TensorFlow (TF) graph.
		
		Arguments required: expr, tensors
		'''
		
		tree = ast.parse(expr, mode='eval').body
		
		return self.fx_fitness_node_parse(tree, tensors)


	def fx_fitness_chain_bool(self, values, operation, tensors):

		'''
		Chains a sequence of boolean operations (e.g. 'a and b and c') into a single TensorFlow (TF) sub graph.

		Arguments required: values, operation, tensors
		'''

		x = tf.cast(self.fx_fitness_node_parse(values[0], tensors), tf.bool)
		if len(values) > 1:
			return operation(x, self.fx_fitness_chain_bool(values[1:], operation, tensors))
		else:
			return x


	def fx_fitness_chain_compare(self, comparators, ops, tensors):

		'''
		Chains a sequence of comparison operations (e.g. 'a > b < c') into a single TensorFlow (TF) sub graph.

		Arguments required: comparators, ops, tensors
		'''

		x = self.fx_fitness_node_parse(comparators[0], tensors)
		y = self.fx_fitness_node_parse(comparators[1], tensors)
		if len(comparators) > 2:
			return tf.logical_and(operators[type(ops[0])](x, y), self.fx_fitness_chain_compare(comparators[1:], ops[1:], tensors))
		else:
			return operators[type(ops[0])](x, y)
		
	
	def fx_fitness_node_parse(self, node, tensors):
	
		'''		
		Recursively transforms parsed expression tree into TensorFlow (TF) graph.
		
		Arguments required: node, tensors
		'''
		
		if isinstance(node, ast.Name): # <tensor_name>
			return tensors[node.id]
		
		elif isinstance(node, ast.Num): # <number>
			shape = tensors[list(tensors.keys())[0]].get_shape()
			return tf.constant(node.n, shape=shape, dtype=tf.float32)
			
		elif isinstance(node, ast.BinOp): # <left> <operator> <right>, e.g., x + y
			return operators[type(node.op)](self.fx_fitness_node_parse(node.left, tensors), self.fx_fitness_node_parse(node.right, tensors))
			
		elif isinstance(node, ast.UnaryOp): # <operator> <operand> e.g., -1
			return operators[type(node.op)](self.fx_fitness_node_parse(node.operand, tensors))

		elif isinstance(node, ast.Call):  # <function>(<arguments>) e.g., sin(x)
			return operators[node.func.id](*[self.fx_fitness_node_parse(arg, tensors) for arg in node.args])

		elif isinstance(node, ast.BoolOp):  # <left> <bool_operator> <right> e.g. x or y
			return self.fx_fitness_chain_bool(node.values, operators[type(node.op)], tensors)

		elif isinstance(node, ast.Compare):  # <left> <compare> <right> e.g., a > z
			return self.fx_fitness_chain_compare([node.left] + node.comparators, node.ops, tensors)
			
		else: raise TypeError(node)
		
	
	def fx_fitness_labels_map(self, result):
	
		'''
		Creates label extraction TensorFlow (TF) sub-graph for CLASSIFY kernel defined as a sequence of boolean conditions. 
		Outputs an array of tuples containing label extracted from the result and corresponding boolean condition triggered.
		
		The original (pre-TensorFlow) code is as follows:
		
			skew = (self.class_labels / 2) - 1 # '-1' keeps a binary classification splitting over the origin
			if solution == 0 and result <= 0 - skew; fitness = 1: # check for first class (the left-most bin)
			elif solution == self.class_labels - 1 and result > solution - 1 - skew; fitness = 1: # check for last class (the right-most bin)
			elif solution - 1 - skew < result <= solution - skew; fitness = 1: # check for class bins between first and last
			else: fitness = 0 # no class match
		
		Arguments required: result
		'''
		
		skew = (self.class_labels // 2) - 1
		label_rules = {self.class_labels - 1: (tf.constant(self.class_labels - 1), tf.constant(' > {}'.format(self.class_labels - 2 - skew)))}
		
		for class_label in range(self.class_labels - 2, 0, -1):
			cond = (class_label - 1 - skew < result) & (result <= class_label - skew)
			label_rules[class_label] = tf.cond(cond, lambda: (tf.constant(class_label), tf.constant(' <= {}'.format(class_label - skew))), lambda: label_rules[class_label + 1])
			
		zero_rule = tf.cond(result <= 0 - skew, lambda: (tf.constant(0), tf.constant(' <= {}'.format(0 - skew))), lambda: label_rules[1])
			
		return zero_rule
		
	
	def fx_fitness_store(self, tree, fitness):
	
		'''
		Records the fitness and length of the raw algorithm (multivariate expression) to the Numpy array. Parsimony can 
		be used to apply pressure to the evolutionary process to select from a set of trees with the same fitness function 
		the one(s) with the simplest (shortest) multivariate expression.
		
		Arguments required: tree, fitness
		'''
	
		fitness = float(fitness)
		fitness = round(fitness, self.precision)
		
		tree[12][1] = fitness # store the fitness with each tree
		tree[12][2] = len(str(self.algo_raw)) # store the length of the raw algo for parsimony
		# if len(tree[3]) > 4: # if the Tree array is wide enough -- SEE SCRATCHPAD
		
		return
		
	
	def fx_fitness_tournament(self, tourn_size):
	
		'''
		Multiple contenders ('tourn_size') are randomly selected and then compared for their respective fitness, as 
		determined in 'fx_fitness_gym'. The tournament is engaged to select a single Tree for each invocation of the
		genetic operators: reproduction, mutation (point, branch), and crossover (sexual reproduction).
		
		The original Tournament Selection drew directly from the foundation generation (gp.generation_a). However, 
		with the introduction of a minimum number of nodes as defined by the user ('gp.tree_depth_min'), 
		'gp.gene_pool' limits the Trees to those which meet all criteria.
		
		Stronger boundary parameters (a reduced gap between the min and max number of nodes) may invoke more compact 
		solutions, but also runs the risk of elitism, even total population die-off where a healthy population once existed.
		
		Arguments required: tourn_size
		'''
		
		tourn_test = 0
		# short_test = 0 # an incomplete parsimony test (seeking shortest solution)
		
		if self.display == 'i': print('\n\tEnter the tournament ...')
		
		for n in range(tourn_size):
			# tree_id = np.random.randint(1, self.tree_pop_max + 1) # former method of selection from the unfiltered population
			rnd = np.random.randint(len(self.gene_pool)) # select one Tree at random from the gene pool
			tree_id = int(self.gene_pool[rnd])
			
			fitness = float(self.population_a[tree_id][12][1]) # extract the fitness from the array
			fitness = round(fitness, self.precision) # force 'result' and 'solution' to the same number of floating points
			
			if self.fitness_type == 'max': # if the fitness function is Maximising
			
				# first time through, 'tourn_test' will be initialised below
				
				if fitness > tourn_test: # if the current Tree's 'fitness' is greater than the priors'
					if self.display == 'i': print('\t\033[36m Tree', tree_id, 'has fitness', fitness, '>', tourn_test, 'and leads\033[0;0m')
					tourn_lead = tree_id # set 'TREE_ID' for the new leader
					tourn_test = fitness # set 'fitness' of the new leader
					# short_test = int(self.population_a[tree_id][12][2]) # set len(algo_raw) of new leader
					
				elif fitness == tourn_test: # if the current Tree's 'fitness' is equal to the priors'
					if self.display == 'i': print('\t\033[36m Tree', tree_id, 'has fitness', fitness, '=', tourn_test, 'and leads\033[0;0m')
					tourn_lead = tree_id # in case there is no variance in this tournament
					# tourn_test remains unchanged
					
					# NEED TO ADD: option for parsimony
					# if int(self.population_a[tree_id][12][2]) < short_test:
						# short_test = int(self.population_a[tree_id][12][2]) # set len(algo_raw) of new leader
						# print '\t\033[36m with improved parsimony score of:\033[1m', short_test, '\033[0;0m'
						
				elif fitness < tourn_test: # if the current Tree's 'fitness' is less than the priors'
					if self.display == 'i': print('\t\033[36m Tree', tree_id, 'has fitness', fitness, '<', tourn_test, 'and is ignored\033[0;0m')
					# tourn_lead remains unchanged
					# tourn_test remains unchanged
					
				else: print('\n\t\033[31m ERROR! In fx_fitness_tournament: fitness =', fitness, 'and tourn_test =', tourn_test, '\033[0;0m'); self.fx_karoo_pause(0)
				
			
			elif self.fitness_type == 'min': # if the fitness function is Minimising
			
				if tourn_test == 0: # first time through, 'tourn_test' is given a baseline value
					tourn_test = fitness
					
				if fitness < tourn_test: # if the current Tree's 'fitness' is less than the priors'
					if self.display == 'i': print('\t\033[36m Tree', tree_id, 'has fitness', fitness, '<', tourn_test, 'and leads\033[0;0m')
					tourn_lead = tree_id # set 'TREE_ID' for the new leader
					tourn_test = fitness # set 'fitness' of the new leader
					
				elif fitness == tourn_test: # if the current Tree's 'fitness' is equal to the priors'
					if self.display == 'i': print('\t\033[36m Tree', tree_id, 'has fitness', fitness, '=', tourn_test, 'and leads\033[0;0m')
					tourn_lead = tree_id # in case there is no variance in this tournament
					# tourn_test remains unchanged
					
				elif fitness > tourn_test: # if the current Tree's 'fitness' is greater than the priors'
					if self.display == 'i': print('\t\033[36m Tree', tree_id, 'has fitness', fitness, '>', tourn_test, 'and is ignored\033[0;0m')
					# tourn_lead remains unchanged
					# tourn_test remains unchanged
					
				else: print('\n\t\033[31m ERROR! In fx_fitness_tournament: fitness =', fitness, 'and tourn_test =', tourn_test, '\033[0;0m'); self.fx_karoo_pause(0)
					
		
		tourn_winner = np.copy(self.population_a[tourn_lead]) # copy full Tree so as to not inadvertantly modify the original tree
		
		if self.display == 'i': print('\n\t\033[36mThe winner of the tournament is Tree:\033[1m', tourn_winner[0][1], '\033[0;0m')
		
		return tourn_winner
		
	
	def fx_fitness_gene_pool(self):
	
		'''
		The gene pool was introduced as means by which advanced users could define additional constraints on the evolved
		functions, in an effort to guide the evolutionary process. The first constraint introduced is the 'mininum number
		of nodes' parameter (gp.tree_depth_min). This defines the minimum number of nodes (in the context of Karoo, this 
		refers to both functions (operators) and terminals (operands)).
		
		When the minimum node count is human guided, it can keep the solution from defaulting to a local minimum, as with
		't/t' in the Kepler problem, by forcing a more complex solution. If you find that when engaging the Regression 
		kernel you are met with a solution which is too simple (eg: linear instead of non-linear), try increasing the 
		minimum number of nodes (with the launch of Karoo, or mid-stream by way of the pause menu).
		
		What's more, you can add additional constraints to the Gene Pool, thereby customizing how the next generation is
		selected.
		
		At this time, the gene pool does *not* limit the number of times any given Tree may be selected for mutation or 
		reproduction nor does it take into account parsimony (seeking the simplest multivariate expression).
		
		This method is automatically invoked with every Tournament Selection ('fx_fitness_tournament').
		
		Arguments required: none
		'''
		
		self.gene_pool = []
		if self.display == 'i': print('\n Prepare a viable gene pool ...'); self.fx_karoo_pause(0)
		
		for tree_id in range(1, len(self.population_a)):
		
			self.fx_eval_poly(self.population_a[tree_id]) # extract the expression
			
			if len(self.population_a[tree_id][3])-1 >= self.tree_depth_min and self.algo_sym != 1: # check if Tree meets the requirements
				if self.display == 'i': print('\t\033[36m Tree', tree_id, 'has >=', self.tree_depth_min, 'nodes and is added to the gene pool\033[0;0m')
				self.gene_pool.append(self.population_a[tree_id][0][1])
				
		if len(self.gene_pool) > 0 and self.display == 'i': print('\n\t The total population of the gene pool is', len(self.gene_pool)); self.fx_karoo_pause(0)
		
		elif len(self.gene_pool) <= 0: # the evolutionary constraints were too tight, killing off the entire population
			# self.generation_id = self.generation_id - 1 # revert the increment of the 'generation_id'
			# self.generation_max = self.generation_id # catch the unused "cont" values in the 'fx_karoo_pause' method
			print("\n\t\033[31m\033[3m 'They're dead Jim. They're all dead!'\033[0;0m There are no Trees in the gene pool. You should archive your populations and (q)uit."); self.fx_karoo_pause(0)
		
		return
		
	
	def fx_fitness_test_classify(self, result):
	
		'''
		Print the Precision-Recall and Confusion Matrix for a CLASSIFICATION run against the test data.

		From scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
			Precision (P) = true_pos / true_pos + false_pos
			Recall (R) = true_pos / true_pos + false_neg
			harmonic mean of Precision and Recall (F1) = 2(P x R) / (P + R)
			
		From scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
			y_pred = result, the estimated target values (labels) generated by Karoo GP
			y_true = solution, the correct target values (labels) associated with the data
			
		Arguments required: result
		'''
		
		for i in range(len(result['result'])):
			print('\t\033[36m Data row {} predicts class:\033[1m {} ({} True)\033[0;0m\033[36m as {:.2f}{}\033[0;0m'.format(i, int(result['labels'][0][i]), int(result['solution'][i]), result['result'][i], result['labels'][1][i]))
			
		print('\n Fitness score: {}'.format(result['fitness']))
		print('\n Precision-Recall report:\n', skm.classification_report(result['solution'], result['labels'][0]))
		print('Confusion matrix:\n', skm.confusion_matrix(result['solution'], result['labels'][0]))
		
		return
		
		
	def fx_fitness_test_regress(self, result):
	
		'''
		Print the Fitness score and Mean Squared Error for a REGRESSION run against the test data.
		'''
		
		for i in range(len(result['result'])):
			print('\t\033[36m Data row {} predicts value:\033[1m {:.2f} ({:.2f} True)\033[0;0m'.format(i, result['result'][i], result['solution'][i]))
			
		MSE, fitness = skm.mean_squared_error(result['result'], result['solution']), result['fitness']
		print('\n\t Regression fitness score: {}'.format(fitness))
		print('\t Mean Squared Error: {}'.format(MSE))
		
		return
		
		
	def fx_fitness_test_match(self, result):
	
		'''
		Print the accuracy for a MATCH kernel run against the test data.
		'''
		
		for i in range(len(result['result'])):
			print('\t\033[36m Data row {} predicts match:\033[1m {:.2f} ({:.2f} True)\033[0;0m'.format(i, result['result'][i], result['solution'][i]))
			
		print('\n\tMatching fitness score: {}'.format(result['fitness']))
		
		return
		
	
	# def fx_fitness_test_[other](self, result):
	
		# '''
		# Print the [statistical measure] for a [OTHER] kernel run against the test data.
		# '''
		
		# for i in range(len(result['result'])):
			# print '\t\033[36m Data row {} predicts value:\033[1m {} ({} label)\033[0;0m'.format(i, int(result['result'][i]), int(result['solution'][i]))
			
		# print '\n\tFitness score: {}'.format(result['fitness'])
	
		# return
		
	
	#++++++++++++++++++++++++++++++++++++++++++
	#   Methods to Evolve a Population        |
	#++++++++++++++++++++++++++++++++++++++++++
	
	def fx_evolve_point_mutate(self, tree):
	
		'''
		Mutate a single point in any Tree (Grow or Full).
		
		Arguments required: tree
		'''
		
		node = np.random.randint(1, len(tree[3])) # randomly select a point in the Tree (including root)
		if self.display == 'i': print('\t\033[36m with', tree[5][node], 'node\033[1m', tree[3][node], '\033[0;0m\033[36mchosen for mutation\n\033[0;0m')		
		elif self.display == 'db': print('\n\n\033[33m *** Point Mutation *** \033[0;0m\n\n\033[36m This is the unaltered tourn_winner:\033[0;0m\n', tree)
		
		if tree[5][node] == 'root':
			rnd = np.random.randint(0, len(self.functions[:,0])) # call the previously loaded .csv which contains all operators
			tree[6][node] = self.functions[rnd][0] # replace function (operator)
			
		elif tree[5][node] == 'func':
			rnd = np.random.randint(0, len(self.functions[:,0])) # call the previously loaded .csv which contains all operators
			tree[6][node] = self.functions[rnd][0] # replace function (operator)
			
		elif tree[5][node] == 'term':
			rnd = np.random.randint(0, len(self.terminals) - 1) # call the previously loaded .csv which contains all terminals
			tree[6][node] = self.terminals[rnd] # replace terminal (variable)
		
		else: print('\n\t\033[31m ERROR! In fx_evolve_point_mutate, node_type =', tree[5][node], '\033[0;0m'); self.fx_karoo_pause(0)
		
		tree = self.fx_evolve_fitness_wipe(tree) # wipe fitness data
		
		if self.display == 'db': print('\n\033[36m This is tourn_winner after node\033[1m', node, '\033[0;0m\033[36mmutation and updates:\033[0;0m\n', tree); self.fx_karoo_pause(0)
		
		return tree, node # 'node' is returned only to be assigned to the 'tourn_trees' record keeping
		
	
	def fx_evolve_full_mutate(self, tree, branch):
	
		'''
		Mutate a branch of a Full method Tree.
		
		The full mutate method is straight-forward. A branch was generated and passed to this method. As the size and 
		shape of the Tree must remain identical, each node is mutated sequentially (copied from the new Tree to replace
		the old, node for node), where functions remain functions and terminals remain terminals.
		
		Arguments required: tree, branch
		'''
		
		if self.display == 'db': print('\n\n\033[33m *** Full Mutation: function to function *** \033[0;0m\n\n\033[36m This is the unaltered tourn_winner:\033[0;0m\n', tree)
		
		for n in range(len(branch)):
			
			# 'root' is not made available for Full mutation as this would build an entirely new Tree
			
			if tree[5][branch[n]] == 'func':
				if self.display == 'i': print('\t\033[36m  from\033[1m', tree[5][branch[n]], '\033[0;0m\033[36mto\033[1m func \033[0;0m')
							
				rnd = np.random.randint(0, len(self.functions[:,0])) # call the previously loaded .csv which contains all operators
				tree[6][branch[n]] = self.functions[rnd][0] # replace function (operator)
				
			elif tree[5][branch[n]] == 'term':
				if self.display == 'i': print('\t\033[36m  from\033[1m', tree[5][branch[n]], '\033[0;0m\033[36mto\033[1m term \033[0;0m')
							
				rnd = np.random.randint(0, len(self.terminals) - 1) # call the previously loaded .csv which contains all terminals
				tree[6][branch[n]] = self.terminals[rnd] # replace terminal (variable)
				
		tree = self.fx_evolve_fitness_wipe(tree) # wipe fitness data

		if self.display == 'db': print('\n\033[36m This is tourn_winner after nodes\033[1m', branch, '\033[0;0m\033[36mwere mutated and updated:\033[0;0m\n', tree); self.fx_karoo_pause(0)
				
		return tree
		
	
	def fx_evolve_grow_mutate(self, tree, branch):
	
		'''
		Mutate a branch of a Grow method Tree.
		
		A branch is selected within a given tree.
		
		If the point of mutation ('branch_top') resides at 'tree_depth_max', we do not need to grow a new tree. As the 
		methods for building trees always assume root (node 0) to be a function, we need only mutate this terminal node
		to another terminal node, and this branch mutate method is complete.

		If the top of that branch is a terminal which does not reside at 'tree_depth_max', then it may either remain a 
		terminal (in which case a new value is randomly assigned) or it may mutate into a function. If it becomes a 
		function, a new branch (mini-tree) is generated to be appended to that nodes current location. The same is true 
		for function-to-function mutation. Either way, the new branch will be only as deep as allowed by the distance 
		from it's branch_top to the bottom of the tree.
		
		If however a function mutates into a terminal, the entire branch beneath the function is deleted from the array
		and the entire array is updated, to fix parent/child links, associated arities, and node IDs.
		
		Arguments required: tree, branch		
		'''
		
		branch_top = int(branch[0]) # replaces 2 instances, below; tested 2016 07/09
		branch_depth = self.tree_depth_max - int(tree[4][branch_top]) # 'tree_depth_max' - depth at 'branch_top' to set max potential size of new branch - 2016 07/10
		
		if branch_depth < 0: # this has never occured ... yet
			print('\n\t\033[31m ERROR! In fx_evolve_grow_mutate: branch_depth < 0\033[0;0m')
			print('\t branch_depth =', branch_depth); self.fx_karoo_pause(0)
			
		elif branch_depth == 0: # the point of mutation ('branch_top') chosen resides at the maximum allowable depth, so mutate term to term
		
			if self.display == 'i': print('\t\033[36m max depth branch node\033[1m', tree[3][branch_top], '\033[0;0m\033[36mmutates from \033[1mterm\033[0;0m \033[36mto \033[1mterm\033[0;0m\n')
			if self.display == 'db': print('\n\n\033[33m *** Grow Mutation: terminal to terminal *** \033[0;0m\n\n\033[36m This is the unaltered tourn_winner:\033[0;0m\n', tree)
			
			rnd = np.random.randint(0, len(self.terminals) - 1) # call the previously loaded .csv which contains all terminals
			tree[6][branch_top] = self.terminals[rnd] # replace terminal (variable)

			if self.display == 'db': print('\n\033[36m This is tourn_winner after terminal\033[1m', branch_top, '\033[0;0m\033[36mmutation, branch deletion, and updates:\033[0;0m\n', tree); self.fx_karoo_pause(0)
			
		else: # the point of mutation ('branch_top') chosen is at least one degree of depth from the maximum allowed
		
			# type_mod = '[func or term]' # TEST & DEBUG: force to 'func' or 'term' and comment the next 3 lines
			rnd = np.random.randint(2)
			if rnd == 0: type_mod = 'func' # randomly selected as Function
			elif rnd == 1: type_mod = 'term' # randomly selected as Terminal
			
			if type_mod == 'term': # mutate 'branch_top' to a terminal and delete all nodes beneath (no subsequent nodes are added to this branch)
				
				if self.display == 'i': print('\t\033[36m branch node\033[1m', tree[3][branch_top], '\033[0;0m\033[36mmutates from\033[1m', tree[5][branch_top], '\033[0;0m\033[36mto\033[1m term \n\033[0;0m')
				if self.display == 'db': print('\n\n\033[33m *** Grow Mutation: branch_top to terminal *** \033[0;0m\n\n\033[36m This is the unaltered tourn_winner:\033[0;0m\n', tree)
				
				rnd = np.random.randint(0, len(self.terminals) - 1) # call the previously loaded .csv which contains all terminals
				tree[5][branch_top] = 'term' # replace type ('func' to 'term' or 'term' to 'term')
				tree[6][branch_top] = self.terminals[rnd] # replace label
				
				tree = np.delete(tree, branch[1:], axis = 1) # delete all nodes beneath point of mutation ('branch_top')
				tree = self.fx_evolve_node_arity_fix(tree) # fix all node arities
				tree = self.fx_evolve_child_link_fix(tree) # fix all child links
				tree = self.fx_evolve_node_renum(tree) # renumber all 'NODE_ID's
				
				if self.display == 'db': print('\n\033[36m This is tourn_winner after terminal\033[1m', branch_top, '\033[0;0m\033[36mmutation, branch deletion, and updates:\033[0;0m\n', tree); self.fx_karoo_pause(0)
				
			
			if type_mod == 'func': # mutate 'branch_top' to a function (a new 'gp.tree' will be copied, node by node, into 'tourn_winner')
				
				if self.display == 'i': print('\t\033[36m branch node\033[1m', tree[3][branch_top], '\033[0;0m\033[36mmutates from\033[1m', tree[5][branch_top], '\033[0;0m\033[36mto\033[1m func \n\033[0;0m')
				if self.display == 'db': print('\n\n\033[33m *** Grow Mutation: branch_top to function *** \033[0;0m\n\n\033[36m This is the unaltered tourn_winner:\033[0;0m\n', tree)
				
				self.fx_gen_tree_build('mutant', self.pop_tree_type, branch_depth) # build new Tree ('gp.tree') with a maximum depth which matches 'branch'
				
				if self.display == 'db': print('\n\033[36m This is the new Tree to be inserted at node\033[1m', branch_top, '\033[0;0m\033[36min tourn_winner:\033[0;0m\n', self.tree); self.fx_karoo_pause(0)
				
				# because we already know the maximum depth to which this branch can grow, there is no need to prune after insertion
				tree = self.fx_evolve_branch_top_copy(tree, branch) # copy root of new 'gp.tree' to point of mutation ('branch_top') in 'tree' ('tourn_winner')
				tree = self.fx_evolve_branch_body_copy(tree) # copy remaining nodes in new 'gp.tree' to 'tree' ('tourn_winner')
				
		tree = self.fx_evolve_fitness_wipe(tree) # wipe fitness data
		
		return tree
		
	
	def fx_evolve_crossover(self, parent, branch_x, offspring, branch_y):
	
		'''
		Refer to the method 'fx_karoo_crossover' for a full description of the genetic operator Crossover.
		
		This method is called twice to produce 2 offspring per pair of parent Trees. Note that in the method 
		'karoo_fx_crossover' the parent/branch relationships are swapped from the first run to the second, such that 
		this method receives swapped components to produce the alternative offspring. Therefore 'parent_b' is first 
		passed to 'offspring' which will receive 'branch_a'. With the second run, 'parent_a' is passed to 'offspring' which 
		will receive 'branch_b'.
		
		Arguments required: parent, branch_x, offspring, branch_y (parents_a / _b, branch_a / _b from 'fx_karoo_crossover')
		'''
		
		crossover = int(branch_x[0]) # pointer to the top of the 1st parent branch passed from 'fx_karoo_crossover'
		branch_top = int(branch_y[0]) # pointer to the top of the 2nd parent branch passed from 'fx_karoo_crossover'
		
		if self.display == 'db': print('\n\n\033[33m *** Crossover *** \033[0;0m')
		
		if len(branch_x) == 1: # if the branch from the parent contains only one node (terminal)
		
			if self.display == 'i': print('\t\033[36m  terminal crossover from \033[1mparent', parent[0][1], '\033[0;0m\033[36mto \033[1moffspring', offspring[0][1], '\033[0;0m\033[36mat node\033[1m', branch_top, '\033[0;0m')
			
			if self.display == 'db':
				print('\n\033[36m In a copy of one parent:\033[0;0m\n', offspring)
				print('\n\033[36m ... we remove nodes\033[1m', branch_y, '\033[0;0m\033[36mand replace node\033[1m', branch_top, '\033[0;0m\033[36mwith a terminal from branch_x\033[0;0m'); self.fx_karoo_pause(0)
				
			offspring[5][branch_top] = 'term' # replace type
			offspring[6][branch_top] = parent[6][crossover] # replace label with that of a particular node in 'branch_x'
			offspring[8][branch_top] = 0 # set terminal arity
			
			offspring = np.delete(offspring, branch_y[1:], axis = 1) # delete all nodes beneath point of mutation ('branch_top')
			offspring = self.fx_evolve_child_link_fix(offspring) # fix all child links
			offspring = self.fx_evolve_node_renum(offspring) # renumber all 'NODE_ID's
			
			if self.display == 'db': print('\n\033[36m This is the resulting offspring:\033[0;0m\n', offspring); self.fx_karoo_pause(0)
			
		
		else: # we are working with a branch from 'parent' >= depth 1 (min 3 nodes)
		
			if self.display == 'i': print('\t\033[36m  branch crossover from \033[1mparent', parent[0][1], '\033[0;0m\033[36mto \033[1moffspring', offspring[0][1], '\033[0;0m\033[36mat node\033[1m', branch_top, '\033[0;0m')
			
			# self.fx_gen_tree_build('test', 'f', 2) # TEST & DEBUG: disable the next 'self.tree ...' line
			self.tree = self.fx_evolve_branch_copy(parent, branch_x) # generate stand-alone 'gp.tree' with properties of 'branch_x'
			
			if self.display == 'db':
				print('\n\033[36m From one parent:\033[0;0m\n', parent)
				print('\n\033[36m ... we copy branch_x\033[1m', branch_x, '\033[0;0m\033[36mas a new, sub-tree:\033[0;0m\n', self.tree); self.fx_karoo_pause(0)
				
			if self.display == 'db':
				print('\n\033[36m ... and insert it into a copy of the second parent in place of the selected branch\033[1m', branch_y,':\033[0;0m\n', offspring); self.fx_karoo_pause(0)
				
			offspring = self.fx_evolve_branch_top_copy(offspring, branch_y) # copy root of 'branch_y' ('gp.tree') to 'offspring'
			offspring = self.fx_evolve_branch_body_copy(offspring) # copy remaining nodes in 'branch_y' ('gp.tree') to 'offspring'
			offspring = self.fx_evolve_tree_prune(offspring, self.tree_depth_max) # prune to the max Tree depth + adjustment - tested 2016 07/10
			
		offspring = self.fx_evolve_fitness_wipe(offspring) # wipe fitness data
		
		return offspring
		
	
	def fx_evolve_branch_select(self, tree):
	
		'''
		Select all nodes in the 'tourn_winner' Tree at and below the randomly selected starting point.
		
		While Grow mutation uses this method to select a region of the 'tourn_winner' to delete, Crossover uses this 
		method to select a region of the 'tourn_winner' which is then converted to a stand-alone tree. As such, it is 
		imperative that the nodes be in the correct order, else all kinds of bad things happen.
		
		Arguments required: tree
		'''
		
		branch = np.array([]) # the array is necessary in order to len(branch) when 'branch' has only one element
		branch_top = np.random.randint(2, len(tree[3])) # randomly select a non-root node
		branch_eval = self.fx_eval_id(tree, branch_top) # generate tuple of 'branch_top' and subseqent nodes
		branch_symp = sympify(branch_eval) # convert string into something useful
		branch = np.append(branch, branch_symp) # append list to array
		
		branch = np.sort(branch) # sort nodes in branch for Crossover.
		
		if self.display == 'i': print('\t \033[36mwith nodes\033[1m', branch, '\033[0;0m\033[36mchosen for mutation\033[0;0m')
		
		return branch
		
	
	def fx_evolve_branch_top_copy(self, tree, branch):
	
		'''
		Copy the point of mutation ('branch_top') from 'gp.tree' to 'tree'.
		
		This method works with 3 inputs: local 'tree' is being modified; local 'branch' is a section of 'tree' which 
		will be removed; and global 'gp.tree' (recycling from initial population generation) is the new Tree to be 
		copied into 'tree', replacing 'branch'.
		
		This method is used in both Grow Mutation and Crossover.
		
		Arguments required: tree, branch
		'''
		
		branch_top = int(branch[0])
		
		tree[5][branch_top] = 'func' # update type ('func' to 'term' or 'term' to 'term'); this modifies gp.tree[5[1] from 'root' to 'func'
		tree[6][branch_top] = self.tree[6][1] # copy node_label from new tree
		tree[8][branch_top] = self.tree[8][1] # copy node_arity from new tree
		
		tree = np.delete(tree, branch[1:], axis = 1) # delete all nodes beneath point of mutation ('branch_top')
		
		c_buffer = self.fx_evolve_c_buffer(tree, branch_top) # generate c_buffer for point of mutation ('branch_top')
		tree = self.fx_evolve_child_insert(tree, branch_top, c_buffer) # insert new nodes
		tree = self.fx_evolve_node_renum(tree) # renumber all 'NODE_ID's

		if self.display == 'db':
			print('\n\t ... inserted node 1 of', len(self.tree[3])-1)
			print('\n\033[36m This is the Tree after a new node is inserted:\033[0;0m\n', tree); self.fx_karoo_pause(0)
		
		return tree
		
	
	def fx_evolve_branch_body_copy(self, tree):
	
		'''
		Copy the body of 'gp.tree' to 'tree', one node at a time.
		
		This method works with 3 inputs: local 'tree' is being modified; local 'branch' is a section of 'tree' which 
		will be removed; and global 'gp.tree' (recycling from initial population generation) is the new Tree to be 
		copied into 'tree', replacing 'branch'.
		
		This method is used in both Grow Mutation and Crossover.
		
		Arguments required: tree
		'''
				
		node_count = 2 # set node count for 'gp.tree' to 2 as the new root has already replaced 'branch_top' in 'fx_evolve_branch_top_copy'
		
		while node_count < len(self.tree[3]): # increment through all nodes in the new Tree ('gp.tree'), starting with node 2
		
			for j in range(1, len(tree[3])): # increment through all nodes in tourn_winner ('tree')
			
				if self.display == 'db': print('\tScanning tourn_winner node_id:', j)
				
				if tree[5][j] == '':					
					tree[5][j] = self.tree[5][node_count] # copy 'node_type' from branch to tree
					tree[6][j] = self.tree[6][node_count] # copy 'node_label' from branch to tree
					tree[8][j] = self.tree[8][node_count] # copy 'node_arity' from branch to tree
					
					if tree[5][j] == 'term':
						tree = self.fx_evolve_child_link_fix(tree) # fix all child links
						tree = self.fx_evolve_node_renum(tree) # renumber all 'NODE_ID's
						
					if tree[5][j] == 'func':
						c_buffer = self.fx_evolve_c_buffer(tree, j) # generate 'c_buffer' for point of mutation ('branch_top')
						tree = self.fx_evolve_child_insert(tree, j, c_buffer) # insert new nodes
						tree = self.fx_evolve_child_link_fix(tree) # fix all child links
						tree = self.fx_evolve_node_renum(tree) # renumber all 'NODE_ID's
						
					if self.display == 'db':
						print('\n\t ... inserted node', node_count, 'of', len(self.tree[3])-1)
						print('\n\033[36m This is the Tree after a new node is inserted:\033[0;0m\n', tree); self.fx_karoo_pause(0)
						
					node_count = node_count + 1 # exit loop when 'node_count' reaches the number of columns in the array 'gp.tree'
							
		return tree
		
	
	def fx_evolve_branch_copy(self, tree, branch):
	
		'''
		This method prepares a stand-alone Tree as a copy of the given branch.
		
		This method is used with Crossover.
		
		Arguments required: tree, branch
		'''
		
		new_tree = np.array([ ['TREE_ID'],['tree_type'],['tree_depth_base'],['NODE_ID'],['node_depth'],['node_type'],['node_label'],['node_parent'],['node_arity'],['node_c1'],['node_c2'],['node_c3'],['fitness'] ])
		
		# tested 2015 06/08
		for n in range(len(branch)):
		
			node = branch[n]
			branch_top = int(branch[0])
			
			TREE_ID = 'copy'
			tree_type = tree[1][1]
			tree_depth_base = int(tree[4][branch[-1]]) - int(tree[4][branch_top]) # subtract depth of 'branch_top' from the last in 'branch'
			NODE_ID = tree[3][node]
			node_depth = int(tree[4][node]) - int(tree[4][branch_top]) # subtract the depth of 'branch_top' from the current node depth
			node_type = tree[5][node]
			node_label = tree[6][node]
			node_parent = '' # updated by 'fx_evolve_parent_link_fix', below
			node_arity = tree[8][node]
			node_c1 = '' # updated by 'fx_evolve_child_link_fix', below
			node_c2 = ''
			node_c3 = ''
			fitness = ''
			
			new_tree = np.append(new_tree, [ [TREE_ID],[tree_type],[tree_depth_base],[NODE_ID],[node_depth],[node_type],[node_label],[node_parent],[node_arity],[node_c1],[node_c2],[node_c3],[fitness] ], 1)
			
		new_tree = self.fx_evolve_node_renum(new_tree)
		new_tree = self.fx_evolve_child_link_fix(new_tree)
		new_tree = self.fx_evolve_parent_link_fix(new_tree)
		new_tree = self.fx_archive_tree_clean(new_tree)
		
		return new_tree
		
	
	def fx_evolve_c_buffer(self, tree, node):
	
		'''
		This method serves the very important function of determining the links from parent to child for any given 
		node. The single, simple formula [parent_arity_sum + prior_sibling_arity - prior_siblings] perfectly determines 
		the correct position of the child node, already in place or to be inserted, no matter the depth nor complexity 
		of the tree.
		
		This method is currently called from the evolution methods, but will soon (I hope) be called from the first 
		generation Tree generation methods (above) such that the same method may be used repeatedly.
		
		Arguments required: tree, node
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
		
	
	def fx_evolve_child_link(self, tree, node, c_buffer):
	
		'''
		Link each parent node to its children.
		
		Arguments required: tree, node, c_buffer
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
				
			else: print('\n\t\033[31m ERROR! In fx_evolve_child_link: node', node, 'has arity', tree[8][node]); self.fx_karoo_pause(0)
				
		return tree
		
	
	def fx_evolve_child_link_fix(self, tree):
	
		'''
		In a given Tree, fix 'node_c1', 'node_c2', 'node_c3' for all nodes.
		
		This is required anytime the size of the array 'gp.tree' has been modified, as with both Grow and Full mutation.
		
		Arguments required: tree
		'''
		
		# tested 2015 06/04
		for node in range(1, len(tree[3])):
		
			c_buffer = self.fx_evolve_c_buffer(tree, node) # generate c_buffer for each node
			tree = self.fx_evolve_child_link(tree, node, c_buffer) # update child links for each node
			
		return tree
		
	
	def fx_evolve_child_insert(self, tree, node, c_buffer):
	
		'''
		Insert child nodes.
		
		Arguments required: tree, node, c_buffer
		'''
		
		if int(tree[8][node]) == 0: # if arity = 0
			print('\n\t\033[31m ERROR! In fx_evolve_child_insert: node', node, 'has arity 0\033[0;0m'); self.fx_karoo_pause(0)
		
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
			
		else: print('\n\t\033[31m ERROR! In fx_evolve_child_insert: node', node, 'arity > 3\033[0;0m'); self.fx_karoo_pause(0)
				
		return tree
		
	
	def fx_evolve_parent_link_fix(self, tree):
	
		'''
		In a given Tree, fix 'parent_id' for all nodes.
		
		This is automatically handled in all mutations except with Crossover due to the need to copy branches 'a' and 
		'b' to their own trees before inserting them into copies of	the parents.
		
		Technically speaking, the 'node_parent' value is not used by any methods. The parent ID can be completely out 
		of whack and the expression will work perfectly. This is maintained for the sole purpose of granting the user 
		a friendly, makes-sense interface which can be read in both directions.
		
		Arguments required: tree
		'''
		
		### THIS METHOD MAY NOT BE REQUIRED AS SORTING 'branch' SEEMS TO HAVE FIXED 'parent_id' ###
		
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
		
	
	def fx_evolve_node_arity_fix(self, tree):
	
		'''
		In a given Tree, fix 'node_arity' for all nodes labeled 'term' but with arity 2.
		
		This is required after a function has been replaced by a terminal, as may occur with both Grow mutation and 
		Crossover.
		
		Arguments required: tree
		'''
		
		# tested 2015 05/31
		for n in range(1, len(tree[3])): # increment through all nodes (exclude 0) in array 'tree'
		
			if tree[5][n] == 'term': # check for discrepency
				tree[8][n] = '0' # set arity to 0
				tree[9][n] = '' # wipe 'node_c1'
				tree[10][n] = '' # wipe 'node_c2'
				tree[11][n] = '' # wipe 'node_c3'
				
		return tree
		
		
	def fx_evolve_node_renum(self, tree):
	
		'''
		Renumber all 'NODE_ID' in a given tree.
		
		This is required after a new generation is evolved as the NODE_ID numbers are carried forward from the previous 
		generation but are no longer in order.
		
		Arguments required: tree
		'''
		
		for n in range(1, len(tree[3])):
		
			tree[3][n] = n  # renumber all Trees in given population
			
		return tree
		
	
	def fx_evolve_fitness_wipe(self, tree):
	
		'''
		Remove all fitness data from a given tree.
		
		This is required after a new generation is evolved as the fitness of the same Tree prior to its mutation will 
		no longer apply.
		
		Arguments required: tree
		'''
		
		tree[12][1:] = '' # wipe fitness data
		
		return tree
		
	
	def fx_evolve_tree_prune(self, tree, depth):
	
		'''
		This method reduces the depth of a Tree. Used with Crossover, the input value 'branch' can be a partial Tree 
		(branch) or a full tree, and it will operate correctly. The input value 'depth' becomes the new maximum depth,
		where depth is defined as the local maximum + the user defined adjustment.
		
		Arguments required: tree, depth
		'''
		
		nodes = []
		
		# tested 2015 06/08
		for n in range(1, len(tree[3])):
		
			if int(tree[4][n]) == depth and tree[5][n] == 'func':
				rnd = np.random.randint(0, len(self.terminals) - 1) # call the previously loaded .csv which contains all terminals
				tree[5][n] = 'term' # mutate type 'func' to 'term'
				tree[6][n] = self.terminals[rnd] # replace label
				
			elif int(tree[4][n]) > depth: # record nodes deeper than the maximum allowed Tree depth
				nodes.append(n)
				
			else: pass # as int(tree[4][n]) < depth and will remain untouched
			
		tree = np.delete(tree, nodes, axis = 1) # delete nodes deeper than the maximum allowed Tree depth
		tree = self.fx_evolve_node_arity_fix(tree) # fix all node arities
		
		return tree
		
		
	def fx_evolve_tree_renum(self, population):
	
		'''
		Renumber all 'TREE_ID' in a given population.
		
		This is required after a new generation is evolved as the TREE_ID numbers are carried forward from the previous 
		generation but are no longer in order.
		
		Arguments required: population
		'''
		
		for tree_id in range(1, len(population)):
		
			population[tree_id][0][1] = tree_id  # renumber all Trees in given population
			
		return population
		
	
	def fx_evolve_pop_copy(self, pop_a, title):
	
		'''
		Copy one population to another.
		
		Simply copying a list of arrays generates a pointer to the original list. Therefore we must append each array 
		to a new, empty array and then build a list of those new arrays.
		
		Arguments required: pop_a, title
		'''
		
		pop_b = [title] # an empty list stores a copy of the prior generation
	
		for tree in range(1, len(pop_a)): # increment through each Tree in the current population

			tree_copy = np.copy(pop_a[tree]) # copy each array in the current population
			pop_b.append(tree_copy) # add each copied Tree to the new population list
			
		return pop_b
		
	
	#++++++++++++++++++++++++++++++++++++++++++
	#   Methods to Display a Tree             |
	#++++++++++++++++++++++++++++++++++++++++++		
			
	def fx_display_tree(self, tree):

		'''
		Display all or part of a Tree on-screen.
		
		This method displays all sequential node_ids from 'start' node through bottom, within the given tree.
		
		Arguments required: tree
		'''
		
		ind = ''
		print('\n\033[1m\033[36m Tree ID', int(tree[0][1]), '\033[0;0m')
		
		for depth in range(0, self.tree_depth_max + 1): # increment through all possible Tree depths - tested 2016 07/09
			print('\n', ind,'\033[36m Tree Depth:', depth, 'of', tree[2][1], '\033[0;0m')
			
			for node in range(1, len(tree[3])): # increment through all nodes (redundant, I know)
				if int(tree[4][node]) == depth:
					print('')
					print(ind,'\033[1m\033[36m NODE:', tree[3][node], '\033[0;0m')
					print(ind,'  type:', tree[5][node])
					print(ind,'  label:', tree[6][node], '\tparent node:', tree[7][node])
					print(ind,'  arity:', tree[8][node], '\tchild node(s):', tree[9][node], tree[10][node], tree[11][node])
					
			ind = ind + '\t'
			
		print('')
		self.fx_eval_poly(tree) # generate the raw and sympified equation for the entire Tree
		print('\t\033[36mTree', tree[0][1], 'yields (raw):', self.algo_raw, '\033[0;0m')
		print('\t\033[36mTree', tree[0][1], 'yields (sym):\033[1m', self.algo_sym, '\033[0;0m')
		
		return
		
	
	def fx_display_branch(self, tree, start):
	
		'''
		Display a Tree branch on-screen.
		
		This method displays all sequential node_ids from 'start' node through bottom, within the given branch.
		
		This method is not used by Karoo GP at this time.
		
		Arguments required: tree, start
		'''
		
		branch = np.array([]) # the array is necessary in order to len(branch) when 'branch' has only one element
		branch_eval = self.fx_eval_id(tree, start) # generate tuple of given 'branch'
		branch_symp = sympify(branch_eval) # convert string from tuple to list
		branch = np.append(branch, branch_symp) # append list to array
		ind = ''
		
		# for depth in range(int(tree[4][start]), int(tree[2][1]) + self.tree_depth_max + 1): # increment through all Tree depths - tested 2016 07/09
		for depth in range(int(tree[4][start]), self.tree_depth_max + 1): # increment through all Tree depths - tested 2016 07/09
			print('\n', ind,'\033[36m Tree Depth:', depth, 'of', tree[2][1], '\033[0;0m')
			
			for n in range(0, len(branch)): # increment through all nodes listed in the branch
				node = branch[n]
				
				if int(tree[4][node]) == depth:
					print('')
					print(ind,'\033[1m\033[36m NODE:', node, '\033[0;0m')
					print(ind,'  type:', tree[5][node])
					print(ind,'  label:', tree[6][node], '\tparent node:', tree[7][node])
					print(ind,'  arity:', tree[8][node], '\tchild node(s):', tree[9][node], tree[10][node], tree[11][node])
					
			ind = ind + '\t'
					
		print('')
		self.fx_eval_poly(tree) # generate the raw and sympified equation for the entire Tree
		print('\t\033[36mTree', tree[0][1], 'yields (raw):', self.algo_raw, '\033[0;0m')
		print('\t\033[36mTree', tree[0][1], 'yields (sym):\033[1m', self.algo_sym, '\033[0;0m')
		
		return
		
	
	#++++++++++++++++++++++++++++++++++++++++++
	#   Methods to Archive                    |
	#++++++++++++++++++++++++++++++++++++++++++
	
	def fx_archive_tree_clean(self, tree):
	
		'''
		This method aesthetically cleans the Tree array, removing redundant data.
		
		Arguments required: tree
		'''
		
		tree[0][2:] = '' # A little clean-up to make things look pretty :)
		tree[1][2:] = '' # Ignore the man behind the curtain!
		tree[2][2:] = '' # Yes, I am a bit OCD ... but you *know* you appreciate clean arrays.
		
		return tree
		
	
	def fx_archive_tree_append(self, tree):
	
		'''
		Append Tree array to the foundation Population.
		
		Arguments required: tree
		'''
		
		self.fx_archive_tree_clean(tree) # clean 'tree' prior to storing
		self.population_a.append(tree) # append 'tree' to population list
		
		return
		
	
	def fx_archive_tree_write(self, population, key):
	
		'''
		Save population_* to disk.
		
		Arguments required: population, key
		'''
		
		with open(self.filename[key], 'a') as csv_file:
			target = csv.writer(csv_file, delimiter=',')
			if self.generation_id != 1: target.writerows(['']) # empty row before each generation
			target.writerows([['Karoo GP by Kai Staats', 'Generation:', str(self.generation_id)]])
			
			for tree in range(1, len(population)):
				target.writerows(['']) # empty row before each Tree
				for row in range(0, 13): # increment through each row in the array Tree
					target.writerows([population[tree][row]])
					
		return
		
	
	def fx_archive_params_write(self, app): # tested 2017 02/13
	
		'''
		Save run-time configuration parameters to disk.
		
		Arguments required: none
		'''
		
		file = open(self.path + '/log_config.txt', 'w')
		file.write('Karoo GP ' + app)
		file.write('\n launched: ' + str(self.datetime))
		file.write('\n dataset: ' + str(self.dataset))
		file.write('\n')
		file.write('\n kernel: ' + str(self.kernel))
		file.write('\n precision: ' + str(self.precision))
		file.write('\n')
		# file.write('tree type: ' + tree_type)
		# file.write('tree depth base: ' + str(tree_depth_base))
		file.write('\n tree depth max: ' + str(self.tree_depth_max))
		file.write('\n min node count: ' + str(self.tree_depth_min))
		file.write('\n')
		file.write('\n genetic operator Reproduction: ' + str(self.evolve_repro))
		file.write('\n genetic operator Point Mutation: ' + str(self.evolve_point))
		file.write('\n genetic operator Branch Mutation: ' + str(self.evolve_branch))
		file.write('\n genetic operator Crossover: ' + str(self.evolve_cross))
		file.write('\n')
		file.write('\n tournament size: ' + str(self.tourn_size))
		file.write('\n population: ' + str(self.tree_pop_max))
		file.write('\n number of generations: ' + str(self.generation_id))		
		file.write('\n\n')
		file.close()
		
		
		file = open(self.path + '/log_test.txt', 'w')
		file.write('Karoo GP ' + app)
		file.write('\n launched: ' + str(self.datetime))
		file.write('\n dataset: ' + str(self.dataset))
		file.write('\n')
		
		if len(self.fittest_dict) > 0:
		
			fitness_best = 0
			fittest_tree = 0
			
			# original method, using pre-built fittest_dict
			# file.write('\n The leading Trees and their associated expressions are:')
			# for n in sorted(self.fittest_dict):
			# file.write('\n\t ' + str(n) + ' : ' + str(self.fittest_dict[n]))
			
			# revised method, re-evaluating all Trees from stored fitness score
			for tree_id in range(1, len(self.population_b)):
			
				fitness = float(self.population_b[tree_id][12][1])
				
				if self.kernel == 'c': # display best fit Trees for the CLASSIFY kernel
					if fitness >= fitness_best: # find the Tree with Maximum fitness score
						fitness_best = fitness; fittest_tree = tree_id # set best fitness Tree
						
				elif self.kernel == 'r': # display best fit Trees for the REGRESSION kernel
					if fitness_best == 0: fitness_best = fitness # set the baseline first time through
					if fitness <= fitness_best: # find the Tree with Minimum fitness score
						fitness_best = fitness; fittest_tree = tree_id # set best fitness Tree
						
				elif self.kernel == 'm': # display best fit Trees for the MATCH kernel
					if fitness == self.data_train_rows: # find the Tree with a perfect match for all data rows
						fitness_best = fitness; fittest_tree = tree_id # set best fitness Tree
						
				# elif self.kernel == '[other]': # display best fit Trees for the [other] kernel
					# if fitness [>=, <=] fitness_best: # find the Tree with [Maximum or Minimum] fitness score
						# fitness_best = fitness; fittest_tree = tree_id # set best fitness Tree
						
				# print 'fitness_best:', fitness_best, 'fittest_tree:', fittest_tree
				
				
			# test the most fit Tree and write to the .txt log
			self.fx_eval_poly(self.population_b[int(fittest_tree)]) # generate the raw and sympified equation for the given Tree using SymPy
			expr = str(self.algo_sym) # get simplified expression and process it by TF - tested 2017 02/02
			result = self.fx_fitness_eval(expr, self.data_test, get_labels=True)
			
			file.write('\n\n Tree ' + str(fittest_tree) + ' is the most fit, with expression:')
			file.write('\n\n ' + str(self.algo_sym))
			
			if self.kernel == 'c':
				file.write('\n\n Classification fitness score: {}'.format(result['fitness']))
				file.write('\n\n Precision-Recall report:\n {}'.format(skm.classification_report(result['solution'], result['labels'][0])))
				file.write('\n Confusion matrix:\n {}'.format(skm.confusion_matrix(result['solution'], result['labels'][0])))
				
			elif self.kernel == 'r':
				MSE, fitness = skm.mean_squared_error(result['result'], result['solution']), result['fitness']
				file.write('\n\n Regression fitness score: {}'.format(fitness))
				file.write('\n Mean Squared Error: {}'.format(MSE))
				
			elif self.kernel == 'm':
				file.write('\n\n Matching fitness score: {}'.format(result['fitness']))
				
			# elif self.kernel == '[other]':
				# file.write( ... )
		
		else: file.write('\n\n There were no evolved solutions generated in this run... your species has gone extinct!')
		
		file.write('\n\n')
		file.close()
		
		return
		

