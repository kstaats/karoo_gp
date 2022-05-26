# Karoo GP Base Class
# Define the methods and global variables used by Karoo GP

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
#import sklearn.cross_validation as skcv # Python 2.7
import sklearn.model_selection as skcv

from sympy import sympify
from datetime import datetime
from collections import OrderedDict

from . import pause as menu

# np.random.seed(1000) # for reproducibility


### TensorFlow Imports and Definitions ###
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

# import tensorflow as tf
import tensorflow.compat.v1 as tf; tf.disable_v2_behavior() # from https://www.tensorflow.org/guide/migrate on 20210125
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
    This Base_BP class contains all methods for Karoo GP. Method names are differentiated from global variable names
    (defined below) by the prefix 'fx_' followed by an object and action, as in fx_display_tree(), with a few
    expections, such as fx_fitness_gene_pool().

    The method categories (denoted by +++ banners +++) are as follows:
        fx_karoo_         Methods to Run Karoo GP
        fx_data_          Methods to Load and Archive Data
        fx_init_          Methods to Construct the 1st Generation
        fx_eval_          Methods to Evaluate a Tree
        fx_fitness_       Methods to Train and Test a Tree for Fitness
        fx_nextgen_       Methods to Construct the next Generation
        fx_evolve_        Methods to Evolve a Population
        fx_display_       Methods to Visualize a Tree

    Error checks are quickly located by searching for 'ERROR!'
    '''

    def __init__(self):

        '''
        ### Global variables used for data management ###
        self.data_train         store train data for processing in TF
        self.data_test          store test data for processing in TF
        self.tf_device          set TF computation backend device (CPU or GPU)
        self.tf_device_log      employed for TensorFlow debugging

        self.data_train_cols    number of cols in the TRAINING data - see fx_data_load()
        self.data_train_rows    number of rows in the TRAINING data - see fx_data_load()
        self.data_test_cols     number of cols in the TEST data - see fx_data_load()
        self.data_test_rows     number of rows in the TEST data - see fx_data_load()

        self.functions          user defined functions (operators) from the associated files/[functions].csv
        self.terminals          user defined variables (operands) from the top row of the associated [data].csv
        self.coeff              user defined coefficients (NOT YET IN USE)
        self.fitness_type       fitness type
        self.datetime           date-time stamp of when the unique directory is created
        self.path               full path to the unique directory created with each run
        self.dataset            local path and dataset filename

        ### Global variables used for evolutionary management ###
        self.population_a       the root generation from which Trees are chosen for mutation and reproduction
        self.population_b       the generation constructed from gp.population_a (recyled)
        self.gene_pool          once-per-generation assessment of trees that meet min and max boundary conditions
        self.gen_id             simple n + 1 increment
        self.fitness_type       set in fx_data_load() as either a minimising or maximising function
        self.tree               axis-1, 13 element Numpy array that defines each Tree, stored in 'gp.population'
        self.pop_*              13 variables that define each Tree - see fx_init_tree_initialise()
        '''

        self.algo_raw = [] # the raw expression generated by Sympy per Tree -- CONSIDER MAKING THIS VARIABLE LOCAL
        self.algo_sym = [] # the expression generated by Sympy per Tree -- CONSIDER MAKING THIS VARIABLE LOCAL
        self.fittest_dict = {} # all Trees which share the best fitness score
        self.gene_pool = [] # store all Tree IDs for use by Tournament
        self.class_labels = 0 # the number of true class labels (data_y)

        return


    #+++++++++++++++++++++++++++++++++++++++++++++
    #   Methods to Run Karoo GP                  |
    #+++++++++++++++++++++++++++++++++++++++++++++

    def fx_karoo_gp(self, kernel, tree_type, tree_depth_base, tree_depth_max, tree_depth_min, tree_pop_max, gen_max, tourn_size, filename, evolve_repro, evolve_point, evolve_branch, evolve_cross, display, precision, swim, mode):

        '''
        This method enables the engagement of the entire Karoo GP application. Instead of returning the user to the pause
        menu, this script terminates at the command-line, providing support for bash and chron job execution.

        Calld by: user script karoo_gp.py

        Arguments required: (see below)
        '''

        ### PART 1 - set global variables to those local values passed from the user script ###
        self.kernel = kernel # fitness function
        # tree_type is passed between methods to construct specific trees
        # tree_depth_base is passed between methods to construct specific trees
        self.tree_depth_max = tree_depth_max # maximum Tree depth for the entire run; limits bloat
        self.tree_depth_min = tree_depth_min # minimum number of nodes
        self.tree_pop_max = tree_pop_max # maximum number of Trees per generation
        self.gen_max = gen_max # maximum number of generations
        self.tourn_size = tourn_size # number of Trees selected for each tournament
        # filename is passed between methods to work with specific populations
        self.evolve_repro = evolve_repro # quantity of a population generated through Reproduction
        self.evolve_point = evolve_point # quantity of a population generated through Point Mutation
        self.evolve_branch = evolve_branch # quantity of a population generated through Branch Mutation
        self.evolve_cross = evolve_cross # quantity of a population generated through Crossover
        self.display = display # display mode is set to (s)ilent # level of on-screen feedback
        self.precision = precision # the number of floating points for the round function in 'fx_fitness_eval'
        self.swim = swim # pass along the gene_pool restriction methodology
        # mode is engaged at the end of the run, below

        ### PART 2 - construct first generation of Trees ###
        self.fx_data_load(filename)
        self.gen_id = 1 # set initial generation ID
        self.population_a = ['Karoo GP by Kai Staats, Generation ' + str(self.gen_id)] # initialise population_a to host the first generation
        self.population_b = ['placeholder'] # initialise population_b to satisfy fx_karoo_pause()
        self.fx_init_construct(tree_type, tree_depth_base) # construct the first population of Trees

        if self.kernel == 'p': # terminate here for Play mode
            self.fx_display_tree(self.tree) # print the current Tree
            self.fx_data_tree_write(self.population_a, 'a') # save this one Tree to disk
            sys.exit()

        elif self.gen_max == 1: # terminate here if constructing just one generation
            self.fx_data_tree_write(self.population_a, 'a') # save this single population to disk
            print ('\n We have constructed a single, stochastic population of', self.tree_pop_max,'Trees, and saved to disk')
            sys.exit()

        else: print ('\n We have constructed the first, stochastic population of', self.tree_pop_max,'Trees')

        ### PART 3 - evaluate first generation of Trees ###
        print ('\n Evaluate the first generation of Trees ...')
        self.fx_fitness_gym(self.population_a) # generate expression, evaluate fitness, compare fitness
        self.fx_data_tree_write(self.population_a, 'a') # save the first generation of Trees to disk

        ### PART 4 - evolve multiple generations of Trees ###
        menu = 1
        while menu != 0: # this allows the user to add generations mid-run and not get buried in nested iterations
            for self.gen_id in range(self.gen_id + 1, self.gen_max + 1): # evolve additional generations of Trees

                print ('\n Evolve a population of Trees for Generation', self.gen_id, '...')
                self.population_b = ['Karoo GP by Kai Staats - Evolving Generation'] # initialise population_b to host the next generation
                self.fx_fitness_gene_pool() # generate the viable gene pool (compares against gp.tree_depth_min)
                self.fx_nextgen_reproduce() # method 1 - Reproduction
                self.fx_nextgen_point_mutate() # method 2 - Point Mutation
                self.fx_nextgen_branch_mutate() # method 3 - Branch Mutation
                self.fx_nextgen_crossover() # method 4 - Crossover
                self.fx_eval_generation() # evaluate all Trees in a single generation
                self.population_a = self.fx_evolve_pop_copy(self.population_b, ['Karoo GP by Kai Staats - Generation ' + str(self.gen_id)])

            if mode == 's': menu = 0 # (s)erver mode - termination with completiont of prescribed run
            else: # (d)esktop mode - user is given an option to quit, review, and/or modify parameters; 'add' generations continues the run
                print ('\n\t\033[32m Enter \033[1m?\033[0;0m\033[32m to review your options or \033[1mq\033[0;0m\033[32muit\033[0;0m')
                menu = self.fx_karoo_pause()

        self.fx_karoo_terminate() # archive populations and return to karoo_gp.py for a clean exit

        return


    def fx_karoo_pause_refer(self):

        '''
        Enables (g)eneration, (i)nteractive, and (d)e(b)ug display modes to offer the (pause) menu at each prompt.

        See fx_karoo_pause() for an explanation of the value being passed.

        Called by: the functions called by PART 4 of fx_karoo_gp()

        Arguments required: none
        '''

        menu = 1
        while menu == 1: menu = self.fx_karoo_pause()

        return


    def fx_karoo_pause(self):

        '''
        Pause the program execution and engage the user, providing a number of options.

        Called by: fx_karoo_pause_refer

        Arguments required: [0,1,2] where (0) refers to an end-of-run; (1) refers to any use of the (pause) menu from
        within the run, and anticipates ENTER as an escape from the menu to continue the run; and (2) refers to an
        'ERROR!' for which the user may want to archive data before terminating. At this point in time, (2) is
        associated with each error but does not provide any special options).
        '''

        ### PART 1 - reset and pack values to send to menu.pause ###
        menu_dict = {'input_a':'',
            'input_b':0,
            'display':self.display,
            'tree_depth_max':self.tree_depth_max,
            'tree_depth_min':self.tree_depth_min,
            'tree_pop_max':self.tree_pop_max,
            'gen_id':self.gen_id,
            'gen_max':self.gen_max,
            'tourn_size':self.tourn_size,
            'evolve_repro':self.evolve_repro,
            'evolve_point':self.evolve_point,
            'evolve_branch':self.evolve_branch,
            'evolve_cross':self.evolve_cross,
            'fittest_dict':self.fittest_dict,
            'pop_a_len':len(self.population_a),
            'pop_b_len':len(self.population_b),
            'path':self.path}

        menu_dict = menu.pause(menu_dict) # call the external function menu.pause

        ### PART 2 - unpack values returned from menu.pause ###
        input_a = menu_dict['input_a']
        input_b = menu_dict['input_b']
        self.display = menu_dict['display']
        self.tree_depth_min = menu_dict['tree_depth_min']
        self.gen_max = menu_dict['gen_max']
        self.tourn_size = menu_dict['tourn_size']
        self.evolve_repro = menu_dict['evolve_repro']
        self.evolve_point = menu_dict['evolve_point']
        self.evolve_branch = menu_dict['evolve_branch']
        self.evolve_cross = menu_dict['evolve_cross']

        ### PART 3 - execute the user queries returned from menu.pause ###
        if input_a == 'esc': return 2 # breaks out of the fx_karoo_gp() or fx_karoo_pause_refer() loop

        elif input_a == 'eval': # evaluate a Tree against the TEST data
            self.fx_eval_poly(self.population_b[input_b]) # generate the raw and sympified expression for the given Tree using SymPy
            #print ('\n\t\033[36mTree', input_b, 'yields (raw):', self.algo_raw, '\033[0;0m') # print the raw expression
            print ('\n\t\033[36mTree', input_b, 'yields (sym):\033[1m', self.algo_sym, '\033[0;0m') # print the sympified expression

            result = self.fx_fitness_eval(str(self.algo_sym), self.data_test, get_pred_labels = True) # might change to algo_raw evaluation
            if self.kernel == 'c': self.fx_fitness_test_classify(result) # TF tested 2017 02/02
            elif self.kernel == 'r': self.fx_fitness_test_regress(result)
            elif self.kernel == 'm': self.fx_fitness_test_match(result)
            # elif self.kernel == '[other]': # use others as a template

        elif input_a == 'print_a': # print a Tree from population_a
            self.fx_display_tree(self.population_a[input_b])

        elif input_a == 'print_b': # print a Tree from population_b
            self.fx_display_tree(self.population_b[input_b])

        elif input_a == 'pop_a': # list all Trees in population_a
            print ('')
            for tree_id in range(1, len(self.population_a)):
                self.fx_eval_poly(self.population_a[tree_id]) # extract the expression
                print ('\t\033[36m Tree', self.population_a[tree_id][0][1], 'yields (sym):\033[1m', self.algo_sym, '\033[0;0m')

        elif input_a == 'pop_b': # list all Trees in population_b
            print ('')
            for tree_id in range(1, len(self.population_b)):
                self.fx_eval_poly(self.population_b[tree_id]) # extract the expression
                print ('\t\033[36m Tree', self.population_b[tree_id][0][1], 'yields (sym):\033[1m', self.algo_sym, '\033[0;0m')

        elif input_a == 'load': # load population_s to replace population_a
            self.fx_data_recover(self.filename['s']) # NEED TO replace 's' with a user defined filename

        elif input_a == 'write': # write the evolving population_b to disk
            self.fx_data_tree_write(self.population_b, 'b')
            print ('\n\t All current members of the evolving population_b saved to karoo_gp/runs/[date-time]/population_b.csv')

        elif input_a == 'add': # check for added generations, then exit fx_karoo_pause and continue the run
            self.gen_max = self.gen_max + input_b # if input_b > 0: self.gen_max = self.gen_max + input_b - REMOVED 2019 06/05

        elif input_a == 'quit': self.fx_karoo_terminate() # archive populations and exit

        return 1


    def fx_karoo_terminate(self):
        '''
        Terminates the evolutionary run (if yet in progress), saves parameters and data to disk, and cleanly returns
        the user to karoo_gp.py and the command line.

        Called by: fx_karoo_gp() and fx_karoo_pause_refer()

        Arguments required: none
        '''

        self.fx_data_params_write()
        target = open(self.filename['f'], 'w'); target.close() # initialize the .csv file for the final population
        self.fx_data_tree_write(self.population_b, 'f') # save the final generation of Trees to disk
        print ('\n\t\033[32m Your Trees and runtime parameters are archived in karoo_gp/runs/[date-time]/\033[0;0m')

        print ('\n\033[3m "It is not the strongest of the species that survive, nor the most intelligent,\033[0;0m')
        print ('\033[3m  but the one most responsive to change."\033[0;0m --Charles Darwin\n')
        print ('\033[3m Congrats!\033[0;0m Your Karoo GP run is complete.\n')
        sys.exit()

        return


    #+++++++++++++++++++++++++++++++++++++++++++++
    #   Methods to Load and Archive Data         |
    #+++++++++++++++++++++++++++++++++++++++++++++

    def fx_data_load(self, filename):

        '''
        The data and function .csv files are loaded according to the fitness function kernel selected by the user. An
        alternative dataset may be loaded at launch, by appending a command line argument. The data is then split into
        both TRAINING and TEST segments in order to validate the success of the GP training run. Datasets less than
        10 rows will not be split, rather copied in full to both TRAINING and TEST as it is assumed you are conducting
        a system validation run, as with the built-in MATCH kernel and associated dataset.

        Called by: fx_karoo_gp

        Arguments required: filename (of the dataset)
        '''

        ### PART 1 - load the associated data set, operators, operands, fitness type, and coefficients ###
        # full_path = os.path.realpath(__file__); karoo_dir = os.path.dirname(full_path) # for user Marco Cavaglia
        karoo_dir = os.path.dirname(os.path.realpath(__file__))

        data_dict = {'c':karoo_dir + '/files/data_CLASSIFY.csv', 'r':karoo_dir + '/files/data_REGRESS.csv', 'm':karoo_dir + '/files/data_MATCH.csv', 'p':karoo_dir + '/files/data_PLAY.csv'}

        if len(sys.argv) == 1: # load data from the default karoo_gp/files/ directory
            data_x = np.loadtxt(data_dict[self.kernel], skiprows = 1, delimiter = ',', dtype = float); data_x = data_x[:,0:-1] # load all but the right-most column
            data_y = np.loadtxt(data_dict[self.kernel], skiprows = 1, usecols = (-1,), delimiter = ',', dtype = float) # load only right-most column (class labels)
            header = open(data_dict[self.kernel],'r') # open file to be read (below)
            self.dataset = data_dict[self.kernel] # copy the name only

        elif len(sys.argv) == 2: # load an external data file
            data_x = np.loadtxt(sys.argv[1], skiprows = 1, delimiter = ',', dtype = float); data_x = data_x[:,0:-1] # load all but the right-most column
            data_y = np.loadtxt(sys.argv[1], skiprows = 1, usecols = (-1,), delimiter = ',', dtype = float) # load only right-most column (class labels)
            header = open(sys.argv[1],'r') # open file to be read (below)
            self.dataset = sys.argv[1] # copy the name only

        elif len(sys.argv) > 2: # receive filename and additional arguments from karoo_gp.py via argparse
            data_x = np.loadtxt(filename, skiprows = 1, delimiter = ',', dtype = float); data_x = data_x[:,0:-1] # load all but the right-most column
            data_y = np.loadtxt(filename, skiprows = 1, usecols = (-1,), delimiter = ',', dtype = float) # load only right-most column (class labels)
            header = open(filename,'r') # open file to be read (below)
            self.dataset = filename # copy the name only

        fitt_dict = {'c':'max', 'r':'min', 'm':'max', 'p':''}
        self.fitness_type = fitt_dict[self.kernel] # load fitness type

        func_dict = {'c':karoo_dir + '/files/operators_CLASSIFY.csv', 'r':karoo_dir + '/files/operators_REGRESS.csv', 'm':karoo_dir + '/files/operators_MATCH.csv', 'p':karoo_dir + '/files/operators_PLAY.csv'}
        self.functions = np.loadtxt(func_dict[self.kernel], delimiter=',', skiprows=1, dtype = str) # load the user defined functions (operators)
        self.terminals = header.readline().split(','); self.terminals[-1] = self.terminals[-1].replace('\n','') # load the user defined terminals (operands)
        self.class_labels = len(np.unique(data_y)) # load the user defined true labels for classification or solutions for regression
        #self.coeff = np.loadtxt(karoo_dir + '/files/coefficients.csv', delimiter=',', skiprows=1, dtype = str) # load the user defined coefficients - NOT USED YET

        ### PART 2 - from the dataset, extract TRAINING and TEST data ###
        if len(data_x) < 11: # for small datasets we will not split them into TRAINING and TEST components
            data_train = np.c_[data_x, data_y]
            data_test = np.c_[data_x, data_y]

        else: # if larger than 10, we run the data through the SciKit Learn's 'random split' function
            x_train, x_test, y_train, y_test = skcv.train_test_split(data_x, data_y, test_size = 0.2) # 80/20 TRAIN/TEST split
            data_x, data_y = [], [] # clear from memory

            data_train = np.c_[x_train, y_train] # recombine each row of data with its associated class label (right column)
            x_train, y_train = [], [] # clear from memory

            data_test = np.c_[x_test, y_test] # recombine each row of data with its associated class label (right column)
            x_test, y_test = [], [] # clear from memory

        self.data_train_cols = len(data_train[0,:]) # qty count
        self.data_train_rows = len(data_train[:,0]) # qty count
        self.data_test_cols = len(data_test[0,:]) # qty count
        self.data_test_rows = len(data_test[:,0]) # qty count

        ### PART 3 - load TRAINING and TEST data for TensorFlow processing - tested 2017 02/02
        self.data_train = data_train # Store train data for processing in TF
        self.data_test = data_test # Store test data for processing in TF
        self.tf_device = "/gpu:0" # Set TF computation backend device (CPU or GPU); gpu:n = 1st, 2nd, or ... GPU device
        self.tf_device_log = False # TF device usage logging (for debugging)

        ### PART 4 - create a unique directory and initialise all .csv files ###
        self.datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
        basename = os.path.basename(filename)  # extract the filename (if any)
        root, ext = os.path.splitext(basename)  # split root from extension
        # generate a unique directory name
        self.path = os.path.join(os.getcwd(), 'runs',
                                 root + '_' + self.datetime + '/')

        if not os.path.isdir(self.path):
            os.makedirs(self.path)  # make a unique directory

        self.filename = {} # a dictionary to hold .csv filenames

        self.filename.update( {'a':self.path + 'population_a.csv'} )
        target = open(self.filename['a'], 'w'); target.close() # initialise a .csv file for population 'a' (foundation)

        self.filename.update( {'b':self.path + 'population_b.csv'} )
        target = open(self.filename['b'], 'w'); target.close() # initialise a .csv file for population 'b' (evolving)

        self.filename.update( {'f':self.path + 'population_f.csv'} )
        target = open(self.filename['f'], 'w'); target.close() # initialise a .csv file for the final population (test)

        self.filename.update( {'s':self.path + 'population_s.csv'} )
        target = open(self.filename['s'], 'w'); target.close() # initialise a .csv file to manually load (seed)

        return


    def fx_data_recover(self, population):

        '''
        This method is used to load a saved population of Trees, as invoked through the (pause) menu where population_r
        replaces population_a in the karoo_gp/runs/[date-time]/ directory.

        Called by: fx_karoo_pause

        Arguments required: population (filename['s'])
        '''

        with open(population, 'rb') as csv_file:
            target = csv.reader(csv_file, delimiter=',')
            n = 0 # track row count

            for row in target:
                print ('row', row)

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

        print ('\n', self.population_a)

        return


    def fx_data_tree_clean(self, tree):

        '''
        This method aesthetically cleans the Tree array, removing redundant data.

        Called by: fx_data_tree_append, fx_evolve_branch_copy

        Arguments required: tree
        '''

        tree[0][2:] = '' # A little clean-up to make things look pretty :)
        tree[1][2:] = '' # Ignore the man behind the curtain!
        tree[2][2:] = '' # Yes, I am a bit OCD ... but you *know* you appreciate clean arrays.

        return tree


    def fx_data_tree_append(self, tree):

        '''
        Append Tree array to the foundation Population.

        Called by: fx_init_construct

        Arguments required: tree
        '''

        self.fx_data_tree_clean(tree) # clean 'tree' prior to storing
        self.population_a.append(tree) # append 'tree' to population list

        return


    def fx_data_tree_write(self, population, key):

        '''
        Save population_* to disk.

        Called by: fx_karoo_gp, fx_eval_generation

        Arguments required: population, key
        '''

        with open(self.filename[key], 'a') as csv_file:
            target = csv.writer(csv_file, delimiter=',')
            if self.gen_id != 1: target.writerows(['']) # empty row before each generation
            target.writerows([['Karoo GP by Kai Staats', 'Generation:', str(self.gen_id)]])

            for tree in range(1, len(population)):
                target.writerows(['']) # empty row before each Tree
                for row in range(0, 13): # increment through each row in the array Tree
                    target.writerows([population[tree][row]])

        return


    def fx_data_params_write(self): # tested 2017 02/13; argument 'app' removed to simplify termination 2019 06/08

        '''
        Save run-time configuration parameters to disk.

        Called by: fx_karoo_gp, fx_karoo_pause

        Arguments required: app
        '''

        file = open(self.path + 'log_config.txt', 'w')
        file.write('Karoo GP')
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
        file.write('\n number of generations: ' + str(self.gen_id))
        file.write('\n\n')
        file.close()


        file = open(self.path + 'log_test.txt', 'w')
        file.write('Karoo GP')
        file.write('\n launched: ' + str(self.datetime))
        file.write('\n dataset: ' + str(self.dataset))
        file.write('\n')

        if len(self.fittest_dict) > 0:

            fitness_best = 0
            fittest_tree = 0

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

                # elif self.kernel == '[other]': # use others as a template

                # print ('fitness_best:', fitness_best, 'fittest_tree:', fittest_tree)


            # test the most fit Tree and write to the .txt log
            self.fx_eval_poly(self.population_b[int(fittest_tree)]) # generate the raw and sympified expression for the given Tree using SymPy
            expr = str(self.algo_sym) # get simplified expression and process it by TF - tested 2017 02/02
            result = self.fx_fitness_eval(expr, self.data_test, get_pred_labels = True)

            file.write('\n\n Tree ' + str(fittest_tree) + ' is the most fit, with expression:')
            file.write('\n\n ' + str(self.algo_sym))

            if self.kernel == 'c':
                file.write('\n\n Classification fitness score: {}'.format(result['fitness']))
                file.write('\n\n Precision-Recall report:\n {}'.format(skm.classification_report(result['solution'], result['pred_labels'][0])))
                file.write('\n Confusion matrix:\n {}'.format(skm.confusion_matrix(result['solution'], result['pred_labels'][0])))

            elif self.kernel == 'r':
                MSE, fitness = skm.mean_squared_error(result['result'], result['solution']), result['fitness']
                file.write('\n\n Regression fitness score: {}'.format(fitness))
                file.write('\n Mean Squared Error: {}'.format(MSE))

            elif self.kernel == 'm':
                file.write('\n\n Matching fitness score: {}'.format(result['fitness']))

            # elif self.kernel == '[other]': # use others as a template

        else: file.write('\n\n There were no evolved solutions generated in this run... your species has gone extinct!')

        file.write('\n\n')
        file.close()

        return


    #+++++++++++++++++++++++++++++++++++++++++++++
    #   Methods to Construct the 1st Generation  |
    #+++++++++++++++++++++++++++++++++++++++++++++

    def fx_init_construct(self, tree_type, tree_depth_base):

        '''
        This method constructs the initial population of Tree type 'tree_type' and of the size tree_depth_base. The Tree
        can be Full, Grow, or "Ramped Half/Half" as defined by John Koza.

        Called by: fx_karoo_gp

        Arguments required: tree_type, tree_depth_base
        '''

        if self.display == 'i':
            print ('\n\t\033[32m Press \033[36m\033[1m?\033[0;0m\033[32m at any \033[36m\033[1m(pause)\033[0;0m\033[32m, or \033[36m\033[1mENTER\033[0;0m \033[32mto continue the run\033[0;0m'); self.fx_karoo_pause_refer()

        if tree_type == 'r': # Ramped 50/50

            TREE_ID = 1
            for n in range(1, int((self.tree_pop_max / 2) / tree_depth_base) + 1): # split the population into equal parts
                for depth in range(1, tree_depth_base + 1): # build 2 Trees at each depth
                    self.fx_init_tree_build(TREE_ID, 'f', depth) # build a Full Tree
                    self.fx_data_tree_append(self.tree) # append Tree to the list 'gp.population_a'
                    TREE_ID = TREE_ID + 1

                    self.fx_init_tree_build(TREE_ID, 'g', depth) # build a Grow Tree
                    self.fx_data_tree_append(self.tree) # append Tree to the list 'gp.population_a'
                    TREE_ID = TREE_ID + 1

            if TREE_ID < self.tree_pop_max: # eg: split 100 by 2*3 and it will produce only 96 Trees ...
                for n in range(self.tree_pop_max - TREE_ID + 1): # ... so we complete the run
                    self.fx_init_tree_build(TREE_ID, 'g', tree_depth_base)
                    self.fx_data_tree_append(self.tree)
                    TREE_ID = TREE_ID + 1

            else: pass

        else: # Full or Grow
            for TREE_ID in range(1, self.tree_pop_max + 1):
                self.fx_init_tree_build(TREE_ID, tree_type, tree_depth_base) # build the 1st generation of Trees
                self.fx_data_tree_append(self.tree)

        return


    def fx_init_tree_build(self, TREE_ID, tree_type, tree_depth_base):

        '''
        This method combines 4 sub-methods into a single method for ease of deployment. It is designed to executed
        within a loop such that an entire population is built. However, it may also be run from the command line,
        passing a single TREE_ID to the method.

        'tree_type' is either (f)ull or (g)row. Note, however, that when the user selects 'ramped 50/50' at launch,
        it is still (f) or (g) which are passed to this method.

        Called by: fx_init_construct, fx_evolve_crossover, fx_evolve_grow_mutate

        Arguments required: TREE_ID, tree_type, tree_depth_base
        '''

        self.fx_init_tree_initialise(TREE_ID, tree_type, tree_depth_base) # initialise a new Tree
        self.fx_init_root_build() # build the Root node
        self.fx_init_function_build() # build the Function nodes
        self.fx_init_terminal_build() # build the Terminal nodes

        return # each Tree is written to 'gp.tree'


    def fx_init_tree_initialise(self, TREE_ID, tree_type, tree_depth_base):

        '''
        Assign 13 global variables to the array 'tree'.

        Build the array 'tree' with 13 rows and initally, just 1 column of labels. This array will grow horizontally as
        each new node is appended. The values of this array are stored as string characters, numbers forced to integers at
        the point of execution.

        Use of the debug (db) interface mode enables the user to watch the genetic operations as they work on the Trees.

        Called by: fx_init_tree_build

        Arguments required: TREE_ID, tree_type, tree_depth_base
        '''

        self.pop_TREE_ID = TREE_ID  # pos 0: a unique identifier for each tree
        self.pop_tree_type = tree_type  # pos 1: a global constant based upon the initial user setting
        self.pop_tree_depth_base = tree_depth_base  # pos 2: a global variable which conveys 'tree_depth_base' as unique to each new Tree
        self.pop_NODE_ID = 1  # pos 3: unique identifier for each node; this is the INDEX KEY to this array
        self.pop_node_depth = 0  # pos 4: depth of each node when committed to the array
        self.pop_node_type = ''  # pos 5: root, function, or terminal
        self.pop_node_label = ''  # pos 6: operator [+, -, *, ...] or terminal [a, b, c, ...]
        self.pop_node_parent = ''  # pos 7: parent node
        self.pop_node_arity = ''  # pos 8: number of nodes attached to each non-terminal node
        self.pop_node_c1 = ''  # pos 9: child node 1
        self.pop_node_c2 = ''  # pos 10: child node 2
        self.pop_node_c3 = ''  # pos 11: child node 3 (assumed max of 3 with boolean operator 'if')
        self.pop_fitness = ''  # pos 12: fitness score following Tree evaluation

        self.tree = np.array([ ['TREE_ID'],['tree_type'],['tree_depth_base'],['NODE_ID'],['node_depth'],['node_type'],['node_label'],['node_parent'],['node_arity'],['node_c1'],['node_c2'],['node_c3'],['fitness'] ])

        return


    ### Root Node ###

    def fx_init_root_build(self):

        '''
        Build the Root node for the initial population.

        Called by: fx_init_tree_build

        Arguments required: none
        '''

        self.fx_init_function_select() # select the operator for root

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

        else: print ('\n\t\033[31m ERROR! In fx_init_root_build: pop_node_arity =', self.pop_node_arity, '\033[0;0m'); self.fx_karoo_pause() # consider special instructions for this (pause) - 2019 06/08

        self.pop_node_type = 'root'

        self.fx_init_node_commit()

        return


    ### Function Nodes ###

    def fx_init_function_build(self):

        '''
        Build the Function nodes for the intial population.

        Called by: fx_init_tree_build

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
                        prior_sibling_arity = self.fx_init_function_gen(parent_arity_sum, prior_sibling_arity, prior_siblings) # ... generate a Function ndoe
                        prior_siblings = prior_siblings + 1 # sum sibling nodes (current depth) who will spawn their own children (cousins? :)

        return


    def fx_init_function_gen(self, parent_arity_sum, prior_sibling_arity, prior_siblings):

        '''
        Generate a single Function node for the initial population.

        Called by fx_init_function_build

        Arguments required: parent_arity_sum, prior_sibling_arity, prior_siblings
        '''

        if self.pop_tree_type == 'f': # user defined as (f)ull
            self.fx_init_function_select() # retrieve a function
            self.fx_init_child_link(parent_arity_sum, prior_sibling_arity, prior_siblings) # establish links to children

        elif self.pop_tree_type == 'g': # user defined as (g)row
            rnd = np.random.randint(2)

            if rnd == 0: # randomly selected as Function
                self.fx_init_function_select() # retrieve a function
                self.fx_init_child_link(parent_arity_sum, prior_sibling_arity, prior_siblings) # establish links to children

            elif rnd == 1: # randomly selected as Terminal
                self.fx_init_terminal_select() # retrieve a terminal
                self.pop_node_c1 = ''
                self.pop_node_c2 = ''
                self.pop_node_c3 = ''

        self.fx_init_node_commit() # commit new node to array
        prior_sibling_arity = prior_sibling_arity + self.pop_node_arity # sum the arity of prior siblings

        return prior_sibling_arity


    def fx_init_function_select(self):

        '''
        Define a single Function (operator extracted from the associated functions.csv) for the initial population.

        Called by: fx_init_function_gen, fx_init_root_build

        Arguments required: none
        '''

        self.pop_node_type = 'func'
        rnd = np.random.randint(0, len(self.functions[:,0])) # call the previously loaded .csv which contains all operators
        self.pop_node_label = self.functions[rnd][0]
        self.pop_node_arity = int(self.functions[rnd][1])

        return


    ### Terminal Nodes ###

    def fx_init_terminal_build(self):

        '''
        Build the Terminal nodes for the intial population.

        Called by: fx_init_tree_build

        Arguments required: none
        '''

        self.pop_node_depth = self.pop_tree_depth_base # set the final node_depth (same as 'gp.pop_node_depth' + 1)

        for j in range(1, len(self.tree[3]) ): # increment through all nodes (exclude 0) in array 'tree'

            if int(self.tree[4][j]) == self.pop_node_depth - 1: # find parent nodes which reside at the prior depth

                for k in range(1,(int(self.tree[8][j]) + 1)): # increment through each degree of arity for each parent node
                    self.pop_node_parent = int(self.tree[3][j]) # set the parent 'NODE_ID'  ...
                    self.fx_init_terminal_gen() # ... generate a Terminal node

        return


    def fx_init_terminal_gen(self):

        '''
        Generate a single Terminal node for the initial population.

        Called by: fx_init_terminal_build

        Arguments required: none
        '''

        self.fx_init_terminal_select() # retrieve a terminal
        self.pop_node_c1 = ''
        self.pop_node_c2 = ''
        self.pop_node_c3 = ''

        self.fx_init_node_commit() # commit new node to array

        return


    def fx_init_terminal_select(self):

        '''
        Define a single Terminal (variable extracted from the top row of the associated TRAINING data)

        Called by: fx_init_terminal_gen, fx_init_function_gen

        Arguments required: none
        '''

        self.pop_node_type = 'term'
        rnd = np.random.randint(0, len(self.terminals) - 1) # call the previously loaded .csv which contains all terminals
        self.pop_node_label = self.terminals[rnd]
        self.pop_node_arity = 0

        return


    ### The Lovely Children ###

    def fx_init_child_link(self, parent_arity_sum, prior_sibling_arity, prior_siblings):

        '''
        Link each parent node to its children in the intial population.

        Called by: fx_init_function_gen

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

                else: print ('\n\t\033[31m ERROR! In fx_init_child_link: pop_node_arity =', self.pop_node_arity, '\033[0;0m'); self.fx_karoo_pause() # consider special instructions for this (pause) - 2019 06/08

        return


    def fx_init_node_commit(self):

        '''
        Commit the values of a new node (root, function, or terminal) to the array 'tree'.

        Called by: fx_init_root_build, fx_init_function_gen, fx_init_terminal_gen

        Arguments required: none
        '''

        self.tree = np.append(self.tree, [ [self.pop_TREE_ID],[self.pop_tree_type],[self.pop_tree_depth_base],[self.pop_NODE_ID],[self.pop_node_depth],[self.pop_node_type],[self.pop_node_label],[self.pop_node_parent],[self.pop_node_arity],[self.pop_node_c1],[self.pop_node_c2],[self.pop_node_c3],[self.pop_fitness] ], 1)

        self.pop_NODE_ID = self.pop_NODE_ID + 1

        return


    #+++++++++++++++++++++++++++++++++++++++++++++
    #   Methods to Evaluate a Tree               |
    #+++++++++++++++++++++++++++++++++++++++++++++

    def fx_eval_poly(self, tree):

        '''
        Evaluate a Tree and generate its multivariate expression (both raw and Sympified).

        We need to extract the variables from the expression. However, these variables are no longer correlated
        to the original variables listed across the top of each column of data.csv. Therefore, we must re-assign
        the respective values for each subsequent row in the data .csv, for each Tree's unique expression.

        Called by: fx_karoo_pause, fx_data_params_write, fx_eval_label, fx_fitness_gym, fx_fitness_gene_pool, fx_display_tree

        Arguments required: tree
        '''

        self.algo_raw = self.fx_eval_label(tree, 1) # pass the root 'node_id', then flatten the Tree to a string
        self.algo_sym = sympify(self.algo_raw) # convert string to a functional expression (the coolest line in Karoo! :)

        return


    def fx_eval_label(self, tree, node_id):

        '''
        Evaluate all or part of a Tree (starting at node_id) and return a raw mutivariate expression ('algo_raw').

        This method is called once per Tree, but may be called at any time to prepare an expression for any full or
        partial (branch) Tree contained in 'population'. Pass the starting node for recursion via the local variable
        'node_id' where the local variable 'tree' is a copy of the Tree you desire to evaluate.

        Called by: fx_eval_poly, fx_eval_label (recursively)

        Arguments required: tree, node_id
        '''

        # if tree[6, node_id] == 'not': tree[6, node_id] = ', not' # temp until this can be fixed at data_load

        node_id = int(node_id)

        if tree[8, node_id] == '0': # arity of 0 for the pattern '[term]'
            return '(' + tree[6, node_id] + ')' # 'node_label' (function or terminal)

        else:
            if tree[8, node_id] == '1': # arity of 1 for the explicit pattern 'not [term]'
                return self.fx_eval_label(tree, tree[9, node_id]) + tree[6, node_id]

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

        Called by: fx_eval_id (recursively), fx_evolve_branch_select

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
        This method invokes the evaluation of an entire generation of Trees. It automatically evaluates population_b
        before invoking the copy of _b to _a.

        Called by: fx_karoo_gp

        Arguments required: none
        '''

        if self.display != 's':
            if self.display == 'i': print ('')
            print ('\n Evaluate all Trees in Generation', self.gen_id)
            if self.display == 'i': self.fx_karoo_pause_refer() # 2019 06/07

        for tree_id in range(1, len(self.population_b)): # renumber all Trees in given population - merged fx_evolve_tree_renum 2018 04/12
            self.population_b[tree_id][0][1] = tree_id

        self.fx_fitness_gym(self.population_b) # run fx_eval(), fx_fitness(), fx_fitness_store(), and fitness record
        self.fx_data_tree_write(self.population_b, 'a') # archive current population as foundation for next generation

        if self.display != 's':
            print ('\n Copy gp.population_b to gp.population_a\n')

        return


    #+++++++++++++++++++++++++++++++++++++++++++++
    #   Methods to Train and Test a Tree         |
    #+++++++++++++++++++++++++++++++++++++++++++++

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

        Called by: fx_karoo_gp, fx_eval_generations

        Arguments required: population
        '''

        fitness_best = 0
        self.fittest_dict = {}
        time_sum = 0

        for tree_id in range(1, len(population)):

            ### PART 1 - GENERATE MULTIVARIATE EXPRESSION FOR EACH TREE ###
            self.fx_eval_poly(population[tree_id]) # extract the expression
            if self.display not in ('s'): print ('\t\033[36mTree', population[tree_id][0][1], 'yields (sym):\033[1m', self.algo_sym, '\033[0;0m')

            ### PART 2 - EVALUATE FITNESS FOR EACH TREE AGAINST TRAINING DATA ###
            fitness = 0

            expr = str(self.algo_sym) # get sympified expression and process it with TF - tested 2017 02/02
            result = self.fx_fitness_eval(expr, self.data_train)
            fitness = result['fitness'] # extract fitness score

            if self.display == 'i':
                print ('\t \033[36m with fitness sum:\033[1m', fitness, '\033[0;0m\n')

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

            # elif self.kernel == '[other]': # use others as a template

        print ('\n\033[36m ', len(list(self.fittest_dict.keys())), 'trees\033[1m', np.sort(list(self.fittest_dict.keys())), '\033[0;0m\033[36moffer the highest fitness scores.\033[0;0m')
        if self.display == 'g': self.fx_karoo_pause_refer() # 2019 06/07

        return


    def fx_fitness_eval(self, expr, data, get_pred_labels = False):

        '''
        Computes tree expression using TensorFlow (TF) returning results and fitness scores.

        This method orchestrates most of the TF routines by parsing input string 'expression' and converting it into a TF
        operation graph which is then processed in an isolated TF session to compute the results and corresponding fitness
        values.

            'self.tf_device' - controls which device will be used for computations (CPU or GPU).
            'self.tf_device_log' - controls device placement logging (debug only).

        Args:
            'expr' - a string containing math expression to be computed on the data. Variable names should match corresponding
            terminal names in 'self.terminals'.

            'data' - an 'n by m' matrix of the data points containing n observations and m features per observation.
            Variable order should match corresponding order of terminals in 'self.terminals'.

            'get_pred_labels' - a boolean flag which controls whether the predicted labels should be extracted from the
            evolved results. This applies only to the CLASSIFY kernel and defaults to 'False'.

        Returns:
            A dict mapping keys to the following outputs:
                'result' - an array of the results of applying given expression to the data
                'pred_labels' - an array of the predicted labels extracted from the results; defined only for CLASSIFY kernel, else None
                'solution' - an array of the solution values extracted from the data (variable 's' in the dataset)
                'pairwise_fitness' - an array of the element-wise results of applying corresponding fitness kernel function
                'fitness' - aggregated scalar fitness score

        Called by: fx_karoo_pause, fx_data_params_write, fx_fitness_gym

        Arguments required: expr, data
        '''

        # Initialize TensorFlow session
        tf.reset_default_graph() # Reset TF internal state and cache (after previous processing)
        config = tf.ConfigProto(log_device_placement=self.tf_device_log, allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            with sess.graph.device(self.tf_device):

                # 1 - Load data into TF vectors
                tensors = {}
                for i in range(len(self.terminals)):
                    var = self.terminals[i]
                    tensors[var] = tf.constant(data[:, i], dtype=tf.float32) # converts data into vectors

                # 2- Transform string expression into TF operation graph
                result = self.fx_fitness_expr_parse(expr, tensors)
                pred_labels = tf.no_op() # a placeholder, applies only to CLASSIFY kernel
                solution = tensors['s'] # solution value is assumed to be stored in 's' terminal

                # 3- Add fitness computation into TF graph
                if self.kernel == 'c': # CLASSIFY kernel

                    '''
                    Creates element-wise fitness computation TensorFlow (TF) sub-graph for CLASSIFY kernel.

                    This method uses the 'sympified' (SymPy) expression ('algo_sym') created in fx_eval_poly() and the data set
                    loaded at run-time to evaluate the fitness of the selected kernel.

                    This multiclass classifer compares each row of a given Tree to the known solution, comparing predicted labels
                    generated by Karoo GP against the true classs labels. This method is able to work with any number of class
                    labels, from 2 to n. The left-most bin includes -inf. The right-most bin includes +inf. Those inbetween are
                    by default confined to the spacing of 1.0 each, as defined by:

                        (solution - 1) < result <= solution

                    The skew adjusts the boundaries of the bins such that they fall on both the negative and positive sides of the
                    origin. At the time of this writing, an odd number of class labels will generate an extra bin on the positive
                    side of origin as it has not yet been determined the effect of enabling the middle bin to include both a
                    negative and positive result.
                    '''

                    # was breaking with upgrade from Tensorflow 1.1 to 1.3; fixed by Iurii by replacing [] with () as of 20171026
                    # if get_pred_labels: pred_labels = tf.map_fn(self.fx_fitness_labels_map, result, dtype = [tf.int32, tf.string], swap_memory = True)
                    if get_pred_labels: pred_labels = tf.map_fn(self.fx_fitness_labels_map, result, dtype = (tf.int32, tf.string), swap_memory = True)

                    skew = (self.class_labels / 2) - 1

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

                    '''
                    A very, very basic REGRESSION kernel which is not designed to perform well in the real world. It requires
                    that you raise the minimum node count to keep it from converging on the value of '1'. Consider writing or
                    integrating a more sophisticated kernel.
                    '''

                    pairwise_fitness = tf.abs(solution - result)


                elif self.kernel == 'm': # MATCH kernel

                    '''
                    This is used for demonstration purposes only.
                    '''

                    # pairwise_fitness = tf.cast(tf.equal(solution, result), tf.int32) # breaks due to floating points
                    RTOL, ATOL = 1e-05, 1e-08 # fixes above issue by checking if a float value lies within a range of values
                    pairwise_fitness = tf.cast(tf.less_equal(tf.abs(solution - result), ATOL + RTOL * tf.abs(result)), tf.int32)

                # elif self.kernel == '[other]': # use others as a template

                else: raise Exception('Kernel type is wrong or missing. You entered {}'.format(self.kernel))

                fitness = tf.reduce_sum(pairwise_fitness)

                # Process TF graph and collect the results
                result, pred_labels, solution, fitness, pairwise_fitness = sess.run([result, pred_labels, solution, fitness, pairwise_fitness])

        return {'result': result, 'pred_labels': pred_labels, 'solution': solution, 'fitness': float(fitness), 'pairwise_fitness': pairwise_fitness}


    def fx_fitness_expr_parse(self, expr, tensors):

        '''
        Extract expression tree from the string algo_sym and transform into TensorFlow (TF) graph.

        Called by: fx_fitness_eval

        Arguments required: expr, tensors
        '''

        tree = ast.parse(expr, mode='eval').body

        return self.fx_fitness_node_parse(tree, tensors)


    def fx_fitness_chain_bool(self, values, operation, tensors):

        '''
        Chains a sequence of boolean operations (e.g. 'a and b and c') into a single TensorFlow (TF) sub graph.

        Called by: fx_fitness_node_parse

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

        Called by: fx_fitness_node_parse

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

        Called by: fx_fitness_expr_parse, fx_fitness_chain_bool, fx_fitness_chain_compare

        Arguments required: node, tensors
        '''

        if isinstance(node, ast.Name): # <tensor_name>
            return tensors[node.id]

        elif isinstance(node, ast.Num): # <number>
            #shape = tensors[tensors.keys()[0]].get_shape() # Python 2.7
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
        For the CLASSIFY kernel, creates a TensorFlow (TF) sub-graph defined as a sequence of boolean conditions based upon
        the quantity of true class labels provided in the data .csv. Outputs an array of tuples containing the predicted
        labels based upon the result and corresponding boolean condition triggered.

        For comparison, the original (pre-TensorFlow) cod follows:

            skew = (self.class_labels / 2) - 1 # '-1' keeps a binary classification splitting over the origin
            if solution == 0 and result <= 0 - skew; fitness = 1: # check for first class (the left-most bin)
            elif solution == self.class_labels - 1 and result > solution - 1 - skew; fitness = 1: # check for last class (the right-most bin)
            elif solution - 1 - skew < result <= solution - skew; fitness = 1: # check for class bins between first and last
            else: fitness = 0 # no class match

        Called by: fx_fitness_eval

        Arguments required: result
        '''

        skew = (self.class_labels / 2) - 1
        label_rules = {self.class_labels - 1: (tf.constant(self.class_labels - 1), tf.constant(' > {}'.format(self.class_labels - 2 - skew)))}

        for class_label in range(self.class_labels - 2, 0, -1):
            cond = (class_label - 1 - skew < result) & (result <= class_label - skew)
            label_rules[class_label] = tf.cond(cond, lambda: (tf.constant(class_label), tf.constant(' <= {}'.format(class_label - skew))), lambda: label_rules[class_label + 1])

        pred_label = tf.cond(result <= 0 - skew, lambda: (tf.constant(0), tf.constant(' <= {}'.format(0 - skew))), lambda: label_rules[1])

        return pred_label


    def fx_fitness_store(self, tree, fitness):

        '''
        Records the fitness and length of the raw algorithm (multivariate expression) to the Numpy array. Parsimony can
        be used to apply pressure to the evolutionary process to select from a set of trees with the same fitness function
        the one(s) with the simplest (shortest) multivariate expression.

        Called by: fx_fitness_gym

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
        determined in fx_fitness_gym(). The tournament is engaged to select a single Tree for each invocation of the
        genetic operators: reproduction, mutation (point, branch), and crossover (sexual reproduction).

        The original Tournament Selection drew directly from the foundation generation (gp.generation_a). However,
        with the introduction of a minimum number of nodes as defined by the user ('gp.tree_depth_min'),
        'gp.gene_pool' limits the Trees to those which meet all criteria.

        Stronger boundary parameters (a reduced gap between the min and max number of nodes) may invoke more compact
        solutions, but also runs the risk of elitism, even total population die-off where a healthy population once existed.

        Called by: fx_nextgen_reproduce, fx_nextgen_point_mutate, fx_nextgen_branch_mutate, fx_nextgen_crossover

        Arguments required: tourn_size
        '''

        tourn_test = 0
        # short_test = 0 # an incomplete parsimony test (seeking shortest solution)

        if self.display == 'i': print ('\n\tEnter the tournament ...')

        for n in range(tourn_size):
            # tree_id = np.random.randint(1, self.tree_pop_max + 1) # former method of selection from the unfiltered population
            rnd = np.random.randint(len(self.gene_pool)) # select one Tree at random from the gene pool
            tree_id = int(self.gene_pool[rnd])

            fitness = float(self.population_a[tree_id][12][1]) # extract the fitness from the array
            fitness = round(fitness, self.precision) # force 'result' and 'solution' to the same number of floating points

            if self.fitness_type == 'max': # if the fitness function is Maximising

                # first time through, 'tourn_test' will be initialised below

                if fitness > tourn_test: # if the current Tree's 'fitness' is greater than the priors'
                    if self.display == 'i': print ('\t\033[36m Tree', tree_id, 'has fitness', fitness, '>', tourn_test, 'and leads\033[0;0m')
                    tourn_lead = tree_id # set 'TREE_ID' for the new leader
                    tourn_test = fitness # set 'fitness' of the new leader
                    # short_test = int(self.population_a[tree_id][12][2]) # set len(algo_raw) of new leader

                elif fitness == tourn_test: # if the current Tree's 'fitness' is equal to the priors'
                    if self.display == 'i': print ('\t\033[36m Tree', tree_id, 'has fitness', fitness, '=', tourn_test, 'and leads\033[0;0m')
                    tourn_lead = tree_id # in case there is no variance in this tournament
                    # tourn_test remains unchanged

                    # NEED TO add option for parsimony
                    # if int(self.population_a[tree_id][12][2]) < short_test:
                        # short_test = int(self.population_a[tree_id][12][2]) # set len(algo_raw) of new leader
                        # print ('\t\033[36m with improved parsimony score of:\033[1m', short_test, '\033[0;0m')

                elif fitness < tourn_test: # if the current Tree's 'fitness' is less than the priors'
                    if self.display == 'i': print ('\t\033[36m Tree', tree_id, 'has fitness', fitness, '<', tourn_test, 'and is ignored\033[0;0m')
                    # tourn_lead remains unchanged
                    # tourn_test remains unchanged

                else: print ('\n\t\033[31m ERROR! In fx_fitness_tournament: fitness =', fitness, 'and tourn_test =', tourn_test, '\033[0;0m'); self.fx_karoo_pause() # consider special instructions for this (pause) - 2019 06/08


            elif self.fitness_type == 'min': # if the fitness function is Minimising

                if tourn_test == 0: # first time through, 'tourn_test' is given a baseline value
                    tourn_test = fitness

                if fitness < tourn_test: # if the current Tree's 'fitness' is less than the priors'
                    if self.display == 'i': print ('\t\033[36m Tree', tree_id, 'has fitness', fitness, '<', tourn_test, 'and leads\033[0;0m')
                    tourn_lead = tree_id # set 'TREE_ID' for the new leader
                    tourn_test = fitness # set 'fitness' of the new leader

                elif fitness == tourn_test: # if the current Tree's 'fitness' is equal to the priors'
                    if self.display == 'i': print ('\t\033[36m Tree', tree_id, 'has fitness', fitness, '=', tourn_test, 'and leads\033[0;0m')
                    tourn_lead = tree_id # in case there is no variance in this tournament
                    # tourn_test remains unchanged

                elif fitness > tourn_test: # if the current Tree's 'fitness' is greater than the priors'
                    if self.display == 'i': print ('\t\033[36m Tree', tree_id, 'has fitness', fitness, '>', tourn_test, 'and is ignored\033[0;0m')
                    # tourn_lead remains unchanged
                    # tourn_test remains unchanged

                else: print ('\n\t\033[31m ERROR! In fx_fitness_tournament: fitness =', fitness, 'and tourn_test =', tourn_test, '\033[0;0m'); self.fx_karoo_pause() # consider special instructions for this (pause) - 2019 06/08


        tourn_winner = np.copy(self.population_a[tourn_lead]) # copy full Tree so as to not inadvertantly modify the original tree

        if self.display == 'i': print ('\n\t\033[36mThe winner of the tournament is Tree:\033[1m', tourn_winner[0][1], '\033[0;0m')

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

        With additional or alternative constraints, you may customize how the next generation is selected.

        At this time, the gene pool does *not* limit the number of times any given Tree may be selected for reproduction or
        mutation nor does it take into account parsimony (seeking the simplest multivariate expression).

        This method is automatically invoked with every Tournament Selection - fx_fitness_tournament().

        Called by: fx_karoo_gp

        Arguments required: none
        '''

        self.gene_pool = []
        if self.display == 'i': print ('\n Prepare a viable gene pool ...'); self.fx_karoo_pause_refer() # 2019 06/07

        for tree_id in range(1, len(self.population_a)):

            self.fx_eval_poly(self.population_a[tree_id]) # extract the expression

            if self.swim == 'p': # each tree must have the min number of nodes defined by the user
                if len(self.population_a[tree_id][3])-1 >= self.tree_depth_min and self.algo_sym != 1: # check if Tree meets the requirements
                    if self.display == 'i': print ('\t\033[36m Tree', tree_id, 'has >=', self.tree_depth_min, 'nodes and is added to the gene pool\033[0;0m')
                    self.gene_pool.append(self.population_a[tree_id][0][1])

            elif self.swim == 'f': # each tree must contain at least one instance of each feature
                if len(np.intersect1d([self.population_a[tree_id][6]],[self.terminals])) == len(self.terminals)-1: # check if Tree contains at least one instance of each feature - 2018 04/14 APS, Ohio
                    if self.display == 'i': print ('\t\033[36m Tree', tree_id, 'includes at least one of each feature and is added to the gene pool\033[0;0m')
                    self.gene_pool.append(self.population_a[tree_id][0][1])

            # elif self.swim == '[other]' # use others as a template

        if len(self.gene_pool) > 0 and self.display == 'i': print ('\n\t The total population of the gene pool is', len(self.gene_pool)); self.fx_karoo_pause_refer() # 2019 06/07

        elif len(self.gene_pool) <= 0: # the evolutionary constraints were too tight, killing off the entire population
            # self.gen_id = self.gen_id - 1 # revert the increment of the 'gen_id'
            # self.gen_max = self.gen_id # catch the unused "cont" values in the fx_karoo_pause() method
            print ("\n\t\033[31m\033[3m 'They're dead Jim. They're all dead!'\033[0;0m There are no Trees in the gene pool. You should archive your population and (q)uit."); self.fx_karoo_pause_refer() # 2019 06/07

        return


    def fx_fitness_test_classify(self, result):

        '''
        Print the Precision-Recall and Confusion Matrix for a CLASSIFICATION run against the test data.

        From scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
            Precision (P) = true_pos / true_pos + false_pos
            Recall (R) = true_pos / true_pos + false_neg
            harmonic mean of Precision and Recall (F1) = 2(P x R) / (P + R)

        From scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
            y_pred = result, the predicted labels generated by Karoo GP
            y_true = solution, the true labels associated with the data

        Called by: fx_karoo_pause

        Arguments required: result
        '''

        for i in range(len(result['result'])):
            print ('\t\033[36m Data row {} predicts class:\033[1m {} ({} True)\033[0;0m\033[36m as {:.2f}{}\033[0;0m'.format(i, int(result['pred_labels'][0][i]), int(result['solution'][i]), result['result'][i], result['pred_labels'][1][i]))

        print ('\n Fitness score: {}'.format(result['fitness']))
        print ('\n Precision-Recall report:\n', skm.classification_report(result['solution'], result['pred_labels'][0]))
        print (' Confusion matrix:\n', skm.confusion_matrix(result['solution'], result['pred_labels'][0]))

        return


    def fx_fitness_test_regress(self, result):

        '''
        Print the Fitness score and Mean Squared Error for a REGRESSION run against the test data.

        Called by: fx_karoo_pause

        Arguments required: result

        '''

        for i in range(len(result['result'])):
            print ('\t\033[36m Data row {} predicts value:\033[1m {:.2f} ({:.2f} True)\033[0;0m'.format(i, result['result'][i], result['solution'][i]))

        MSE, fitness = skm.mean_squared_error(result['result'], result['solution']), result['fitness']
        print ('\n\t Regression fitness score: {}'.format(fitness))
        print ('\t Mean Squared Error: {}'.format(MSE))

        return


    def fx_fitness_test_match(self, result):

        '''
        Print the accuracy for a MATCH kernel run against the test data.

        Called by: fx_karoo_pause

        Arguments required: result
        '''

        for i in range(len(result['result'])):
            print ('\t\033[36m Data row {} predicts match:\033[1m {:.2f} ({:.2f} True)\033[0;0m'.format(i, result['result'][i], result['solution'][i]))

        print ('\n\tMatching fitness score: {}'.format(result['fitness']))

        return


    # def fx_fitness_test_[other](self, result): # use others as a template


    #+++++++++++++++++++++++++++++++++++++++++++++
    #   Methods to Construct the next Generation |
    #+++++++++++++++++++++++++++++++++++++++++++++

    def fx_nextgen_reproduce(self):

        '''
        Through tournament selection, a single Tree from the prior generation is copied without mutation to the next
        generation. This is analogous to a member of the prior generation directly entering the gene pool of the
        subsequent (younger) generation.

        Called by: fx_karoo_gp

        Arguments required: none
        '''

        if self.display != 's':
            if self.display == 'i': print ('')
            print ('  Perform', self.evolve_repro, 'Reproductions ...')
            if self.display == 'i': self.fx_karoo_pause_refer() # 2019 06/07

        for n in range(self.evolve_repro): # quantity of Trees to be copied without mutation
            tourn_winner = self.fx_fitness_tournament(self.tourn_size) # perform tournament selection for each reproduction
            tourn_winner = self.fx_evolve_fitness_wipe(tourn_winner) # wipe fitness data
            self.population_b.append(tourn_winner) # append array to next generation population of Trees

        return


    def fx_nextgen_point_mutate(self):

        '''
        Through tournament selection, a copy of a Tree from the prior generation mutates before being added to the
        next generation. In this method, a single point is selected for mutation while maintaining function nodes as
        functions (operators) and terminal nodes as terminals (variables). The size and shape of the Tree will remain
        identical.

        Called by: fx_karoo_gp

        Arguments required: none
        '''

        if self.display != 's':
            if self.display == 'i': print ('')
            print ('  Perform', self.evolve_point, 'Point Mutations ...')
            if self.display == 'i': self.fx_karoo_pause_refer() # 2019 06/07

        for n in range(self.evolve_point): # quantity of Trees to be generated through mutation
            tourn_winner = self.fx_fitness_tournament(self.tourn_size) # perform tournament selection for each mutation
            tourn_winner, node = self.fx_evolve_point_mutate(tourn_winner) # perform point mutation; return single point for record keeping
            self.population_b.append(tourn_winner) # append array to next generation population of Trees

        return


    def fx_nextgen_branch_mutate(self):

        '''
        Through tournament selection, a copy of a Tree from the prior generation mutates before being added to the
        next generation. Unlike Point Mutation, in this method an entire branch is selected. If the evolutionary run is
        designated as Full, the size and shape of the Tree will remain identical, each node mutated sequentially, where
        functions remain functions and terminals remain terminals. If the evolutionary run is designated as Grow or
        Ramped Half/Half, the size and shape of the Tree may grow smaller or larger, but it may not exceed
        tree_depth_max as defined by the user.

        Called by: fx_karoo_gp

        Arguments required: none
        '''

        if self.display != 's':
            if self.display == 'i': print ('')
            print ('  Perform', self.evolve_branch, 'Branch Mutations ...')
            if self.display == 'i': self.fx_karoo_pause_refer() # 2019 06/07

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


    def fx_nextgen_crossover(self):

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

        Called by: fx_karoo_gp

        Arguments required: none
        '''

        if self.display != 's':
            if self.display == 'i': print ('')
            print ('  Perform', self.evolve_cross, 'Crossovers ...')
            if self.display == 'i': self.fx_karoo_pause_refer() # 2019 06/07

        #for n in range(self.evolve_cross / 2): # Python 2.7
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


    #+++++++++++++++++++++++++++++++++++++++++++++
    #   Methods to Evolve a Population           |
    #+++++++++++++++++++++++++++++++++++++++++++++

    def fx_evolve_point_mutate(self, tree):

        '''
        Mutate a single point in any Tree (Grow or Full).

        Called by: fx_nextgen_point_mutate

        Arguments required: tree
        '''

        node = np.random.randint(1, len(tree[3])) # randomly select a point in the Tree (including root)
        if self.display == 'i': print ('\t\033[36m with', tree[5][node], 'node\033[1m', tree[3][node], '\033[0;0m\033[36mchosen for mutation\n\033[0;0m')
        elif self.display == 'db': print ('\n\n\033[33m *** Point Mutation *** \033[0;0m\n\n\033[36m This is the unaltered tourn_winner:\033[0;0m\n', tree)

        if tree[5][node] == 'root':
            rnd = np.random.randint(0, len(self.functions[:,0])) # call the previously loaded .csv which contains all operators
            tree[6][node] = self.functions[rnd][0] # replace function (operator)

        elif tree[5][node] == 'func':
            rnd = np.random.randint(0, len(self.functions[:,0])) # call the previously loaded .csv which contains all operators
            tree[6][node] = self.functions[rnd][0] # replace function (operator)

        elif tree[5][node] == 'term':
            rnd = np.random.randint(0, len(self.terminals) - 1) # call the previously loaded .csv which contains all terminals
            tree[6][node] = self.terminals[rnd] # replace terminal (variable)

        else: print ('\n\t\033[31m ERROR! In fx_evolve_point_mutate, node_type =', tree[5][node], '\033[0;0m'); self.fx_karoo_pause() # consider special instructions for this (pause) - 2019 06/08

        tree = self.fx_evolve_fitness_wipe(tree) # wipe fitness data

        if self.display == 'db': print ('\n\033[36m This is tourn_winner after node\033[1m', node, '\033[0;0m\033[36mmutation and updates:\033[0;0m\n', tree); self.fx_karoo_pause_refer() # 2019 06/07

        return tree, node # 'node' is returned only to be assigned to the 'tourn_trees' record keeping


    def fx_evolve_full_mutate(self, tree, branch):

        '''
        Mutate a branch of a Full method Tree.

        The full mutate method is straight-forward. A branch was generated and passed to this method. As the size and
        shape of the Tree must remain identical, each node is mutated sequentially (copied from the new Tree to replace
        the old, node for node), where functions remain functions and terminals remain terminals.

        Called by: fx_nextgen_branch_mutate

        Arguments required: tree, branch
        '''

        if self.display == 'db': print ('\n\n\033[33m *** Full Mutation: function to function *** \033[0;0m\n\n\033[36m This is the unaltered tourn_winner:\033[0;0m\n', tree)

        for n in range(len(branch)):

            # 'root' is not made available for Full mutation as this would build an entirely new Tree

            if tree[5][branch[n]] == 'func':
                if self.display == 'i': print ('\t\033[36m  from\033[1m', tree[5][branch[n]], '\033[0;0m\033[36mto\033[1m func \033[0;0m')

                rnd = np.random.randint(0, len(self.functions[:,0])) # call the previously loaded .csv which contains all operators
                tree[6][branch[n]] = self.functions[rnd][0] # replace function (operator)

            elif tree[5][branch[n]] == 'term':
                if self.display == 'i': print ('\t\033[36m  from\033[1m', tree[5][branch[n]], '\033[0;0m\033[36mto\033[1m term \033[0;0m')

                rnd = np.random.randint(0, len(self.terminals) - 1) # call the previously loaded .csv which contains all terminals
                tree[6][branch[n]] = self.terminals[rnd] # replace terminal (variable)

        tree = self.fx_evolve_fitness_wipe(tree) # wipe fitness data

        if self.display == 'db': print ('\n\033[36m This is tourn_winner after nodes\033[1m', branch, '\033[0;0m\033[36mwere mutated and updated:\033[0;0m\n', tree); self.fx_karoo_pause_refer() # 2019 06/07

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

        Called by: fx_nextgen_branch_mutate

        Arguments required: tree, branch
        '''

        branch_top = int(branch[0]) # replaces 2 instances, below; tested 2016 07/09
        branch_depth = self.tree_depth_max - int(tree[4][branch_top]) # 'tree_depth_max' - depth at 'branch_top' to set max potential size of new branch - 2016 07/10

        if branch_depth < 0: # this has never occured ... yet
            print ('\n\t\033[31m ERROR! In fx_evolve_grow_mutate: branch_depth', branch_depth, '< 0'); self.fx_karoo_pause() # consider special instructions for this (pause) - 2019 06/08

        elif branch_depth == 0: # the point of mutation ('branch_top') chosen resides at the maximum allowable depth, so mutate term to term

            if self.display == 'i': print ('\t\033[36m max depth branch node\033[1m', tree[3][branch_top], '\033[0;0m\033[36mmutates from \033[1mterm\033[0;0m \033[36mto \033[1mterm\033[0;0m\n')
            if self.display == 'db': print ('\n\n\033[33m *** Grow Mutation: terminal to terminal *** \033[0;0m\n\n\033[36m This is the unaltered tourn_winner:\033[0;0m\n', tree)

            rnd = np.random.randint(0, len(self.terminals) - 1) # call the previously loaded .csv which contains all terminals
            tree[6][branch_top] = self.terminals[rnd] # replace terminal (variable)

            if self.display == 'db': print ('\n\033[36m This is tourn_winner after terminal\033[1m', branch_top, '\033[0;0m\033[36mmutation, branch deletion, and updates:\033[0;0m\n', tree); self.fx_karoo_pause_refer() # 2019 06/07

        else: # the point of mutation ('branch_top') chosen is at least one depth from the maximum allowed

            # type_mod = '[func or term]' # TEST & DEBUG: force to 'func' or 'term' and comment the next 3 lines
            rnd = np.random.randint(2)
            if rnd == 0: type_mod = 'func' # randomly selected as Function
            elif rnd == 1: type_mod = 'term' # randomly selected as Terminal

            if type_mod == 'term': # mutate 'branch_top' to a terminal and delete all nodes beneath (no subsequent nodes are added to this branch)

                if self.display == 'i': print ('\t\033[36m branch node\033[1m', tree[3][branch_top], '\033[0;0m\033[36mmutates from\033[1m', tree[5][branch_top], '\033[0;0m\033[36mto\033[1m term \n\033[0;0m')
                if self.display == 'db': print ('\n\n\033[33m *** Grow Mutation: branch_top to terminal *** \033[0;0m\n\n\033[36m This is the unaltered tourn_winner:\033[0;0m\n', tree)

                rnd = np.random.randint(0, len(self.terminals) - 1) # call the previously loaded .csv which contains all terminals
                tree[5][branch_top] = 'term' # replace type ('func' to 'term' or 'term' to 'term')
                tree[6][branch_top] = self.terminals[rnd] # replace label

                tree = np.delete(tree, branch[1:], axis = 1) # delete all nodes beneath point of mutation ('branch_top')
                tree = self.fx_evolve_node_arity_fix(tree) # fix all node arities
                tree = self.fx_evolve_child_link_fix(tree) # fix all child links
                tree = self.fx_evolve_node_renum(tree) # renumber all 'NODE_ID's

                if self.display == 'db': print ('\n\033[36m This is tourn_winner after terminal\033[1m', branch_top, '\033[0;0m\033[36mmutation, branch deletion, and updates:\033[0;0m\n', tree); self.fx_karoo_pause_refer() # 2019 06/07


            if type_mod == 'func': # mutate 'branch_top' to a function (a new 'gp.tree' will be copied, node by node, into 'tourn_winner')

                if self.display == 'i': print ('\t\033[36m branch node\033[1m', tree[3][branch_top], '\033[0;0m\033[36mmutates from\033[1m', tree[5][branch_top], '\033[0;0m\033[36mto\033[1m func \n\033[0;0m')
                if self.display == 'db': print ('\n\n\033[33m *** Grow Mutation: branch_top to function *** \033[0;0m\n\n\033[36m This is the unaltered tourn_winner:\033[0;0m\n', tree)

                self.fx_init_tree_build('mutant', self.pop_tree_type, branch_depth) # build new Tree ('gp.tree') with a maximum depth which matches 'branch'

                if self.display == 'db': print ('\n\033[36m This is the new Tree to be inserted at node\033[1m', branch_top, '\033[0;0m\033[36min tourn_winner:\033[0;0m\n', self.tree); self.fx_karoo_pause_refer() # 2019 06/07

                tree = self.fx_evolve_branch_insert(tree, branch) # insert new 'branch' at point of mutation 'branch_top' in tourn_winner 'tree'
                # because we already know the maximum depth to which this branch can grow, there is no need to prune after insertion

        tree = self.fx_evolve_fitness_wipe(tree) # wipe fitness data

        return tree


    def fx_evolve_crossover(self, parent, branch_x, offspring, branch_y):

        '''
        Refer to the method fx_nextgen_crossover() for a full description of the genetic operator Crossover.

        This method is called twice to produce 2 offspring per pair of parent Trees. Note that in the method
        'karoo_fx_crossover' the parent/branch relationships are swapped from the first run to the second, such that
        this method receives swapped components to produce the alternative offspring. Therefore 'parent_b' is first
        passed to 'offspring' which will receive 'branch_a'. With the second run, 'parent_a' is passed to 'offspring' which
        will receive 'branch_b'.

        Called by: fx_nextgen_crossover

        Arguments required: parent, branch_x, offspring, branch_y (parents_a / _b, branch_a / _b from fx_nextgen_crossover()
        '''

        crossover = int(branch_x[0]) # pointer to the top of the 1st parent branch passed from fx_nextgen_crossover()
        branch_top = int(branch_y[0]) # pointer to the top of the 2nd parent branch passed from fx_nextgen_crossover()

        if self.display == 'db': print ('\n\n\033[33m *** Crossover *** \033[0;0m')

        if len(branch_x) == 1: # if the branch from the parent contains only one node (terminal)

            if self.display == 'i': print ('\t\033[36m  terminal crossover from \033[1mparent', parent[0][1], '\033[0;0m\033[36mto \033[1moffspring', offspring[0][1], '\033[0;0m\033[36mat node\033[1m', branch_top, '\033[0;0m')

            if self.display == 'db':
                print ('\n\033[36m In a copy of one parent:\033[0;0m\n', offspring)
                print ('\n\033[36m ... we remove nodes\033[1m', branch_y, '\033[0;0m\033[36mand replace node\033[1m', branch_top, '\033[0;0m\033[36mwith a terminal from branch_x\033[0;0m'); self.fx_karoo_pause_refer() # 2019 06/07

            offspring[5][branch_top] = 'term' # replace type
            offspring[6][branch_top] = parent[6][crossover] # replace label with that of a particular node in 'branch_x'
            offspring[8][branch_top] = 0 # set terminal arity

            offspring = np.delete(offspring, branch_y[1:], axis = 1) # delete all nodes beneath point of mutation ('branch_top')
            offspring = self.fx_evolve_child_link_fix(offspring) # fix all child links
            offspring = self.fx_evolve_node_renum(offspring) # renumber all 'NODE_ID's

            if self.display == 'db': print ('\n\033[36m This is the resulting offspring:\033[0;0m\n', offspring); self.fx_karoo_pause_refer() # 2019 06/07


        else: # we are working with a branch from 'parent' >= depth 1 (min 3 nodes)

            if self.display == 'i': print ('\t\033[36m  branch crossover from \033[1mparent', parent[0][1], '\033[0;0m\033[36mto \033[1moffspring', offspring[0][1], '\033[0;0m\033[36mat node\033[1m', branch_top, '\033[0;0m')

            # self.fx_init_tree_build('test', 'f', 2) # TEST & DEBUG: disable the next 'self.tree ...' line
            self.tree = self.fx_evolve_branch_copy(parent, branch_x) # generate stand-alone 'gp.tree' with properties of 'branch_x'

            if self.display == 'db':
                print ('\n\033[36m From one parent:\033[0;0m\n', parent)
                print ('\n\033[36m ... we copy branch_x\033[1m', branch_x, '\033[0;0m\033[36mas a new, sub-tree:\033[0;0m\n', self.tree); self.fx_karoo_pause_refer() # 2019 06/07

            if self.display == 'db':
                print ('\n\033[36m ... and insert it into a copy of the second parent in place of the selected branch\033[1m', branch_y,':\033[0;0m\n', offspring); self.fx_karoo_pause_refer() # 2019 06/07

            offspring = self.fx_evolve_branch_insert(offspring, branch_y) # insert new 'branch_y' at point of mutation 'branch_top' in tourn_winner 'offspring'
            offspring = self.fx_evolve_tree_prune(offspring, self.tree_depth_max) # prune to the max Tree depth + adjustment - tested 2016 07/10

        offspring = self.fx_evolve_fitness_wipe(offspring) # wipe fitness data

        return offspring


    def fx_evolve_branch_select(self, tree):

        '''
        Select all nodes in the 'tourn_winner' Tree at and below the randomly selected starting point.

        While Grow mutation uses this method to select a region of the 'tourn_winner' to delete, Crossover uses this
        method to select a region of the 'tourn_winner' which is then converted to a stand-alone tree. As such, it is
        imperative that the nodes be in the correct order, else all kinds of bad things happen.

        Called by: fx_nextgen_branch, fx_nextgen_crossover

        Arguments required: tree
        '''

        branch = np.array([]) # the array is necessary in order to len(branch) when 'branch' has only one element
        branch_top = np.random.randint(2, len(tree[3])) # randomly select a non-root node
        branch_eval = self.fx_eval_id(tree, branch_top) # generate tuple of 'branch_top' and subseqent nodes
        branch_symp = sympify(branch_eval) # convert string into something useful
        branch = np.append(branch, branch_symp) # append list to array

        branch = np.sort(branch) # sort nodes in branch for Crossover.

        if self.display == 'i': print ('\t \033[36mwith nodes\033[1m', branch, '\033[0;0m\033[36mchosen for mutation\033[0;0m')

        # return branch per Antonio's fix 20210125
        return branch.astype(int)


    def fx_evolve_branch_insert(self, tree, branch):

        '''
        This method enables the insertion of Tree in place of a branch. It works with 3 inputs: local 'tree' is being
        modified; local 'branch' is a section of 'tree' which will be removed; and the global 'gp.tree' (recycling this
        variable from initial population generation) is the new Tree to be insertd into 'tree', replacing 'branch'.

        The end result is a Tree with a mutated branch. Pretty cool, huh?

        Called by: fx_evolve_grow_mutate, fx_evolve_grow_crossover

        Arguments required: tree, branch
        '''

        # *_branch_top_copy merged with *_body_copy 2018 04/12

        ### PART 1 - insert branch_top from 'gp.tree' into 'tree' ###

        branch_top = int(branch[0])

        tree[5][branch_top] = 'func' # update type ('func' to 'term' or 'term' to 'term'); this modifies gp.tree[5][1] from 'root' to 'func'
        tree[6][branch_top] = self.tree[6][1] # copy node_label from new tree
        tree[8][branch_top] = self.tree[8][1] # copy node_arity from new tree

        tree = np.delete(tree, branch[1:], axis = 1) # delete all nodes beneath point of mutation ('branch_top')

        c_buffer = self.fx_evolve_c_buffer(tree, branch_top) # generate c_buffer for point of mutation ('branch_top')
        tree = self.fx_evolve_child_insert(tree, branch_top, c_buffer) # insert a single new node ('branch_top')
        tree = self.fx_evolve_node_renum(tree) # renumber all 'NODE_ID's

        if self.display == 'db':
            print ('\n\t ... inserted node 1 of', len(self.tree[3])-1)
            print ('\n\033[36m This is the Tree after a new node is inserted:\033[0;0m\n', tree); self.fx_karoo_pause_refer() # 2019 06/07


        ### PART 2 - insert branch_body from 'gp.tree' into 'tree' ###

        node_count = 2 # set node count for 'gp.tree' to 2 as the new root has already replaced 'branch_top' (above)

        while node_count < len(self.tree[3]): # increment through all nodes in the new Tree ('gp.tree'), starting with node 2

            for j in range(1, len(tree[3])): # increment through all nodes in tourn_winner ('tree')

                if self.display == 'db': print ('\tScanning tourn_winner node_id:', j)

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
                        print ('\n\t ... inserted node', node_count, 'of', len(self.tree[3])-1)
                        print ('\n\033[36m This is the Tree after a new node is inserted:\033[0;0m\n', tree); self.fx_karoo_pause_refer() # 2019 06/07

                    node_count = node_count + 1 # exit loop when 'node_count' reaches the number of columns in the array 'gp.tree'

        return tree


    def fx_evolve_branch_copy(self, tree, branch):

        '''
        This method prepares a stand-alone Tree as a copy of the given branch.

        Called by: fx_evolve_crossover

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
            node_parent = '' # updated by fx_evolve_parent_link_fix(), below
            node_arity = tree[8][node]
            node_c1 = '' # updated by fx_evolve_child_link_fix(), below
            node_c2 = ''
            node_c3 = ''
            fitness = ''

            new_tree = np.append(new_tree, [ [TREE_ID],[tree_type],[tree_depth_base],[NODE_ID],[node_depth],[node_type],[node_label],[node_parent],[node_arity],[node_c1],[node_c2],[node_c3],[fitness] ], 1)

        new_tree = self.fx_evolve_node_renum(new_tree)
        new_tree = self.fx_evolve_child_link_fix(new_tree)
        new_tree = self.fx_evolve_parent_link_fix(new_tree)
        new_tree = self.fx_data_tree_clean(new_tree)

        return new_tree


    def fx_evolve_c_buffer(self, tree, node):

        '''
        This method serves the very important function of determining the links from parent to child for any given
        node. The single, simple formula [parent_arity_sum + prior_sibling_arity - prior_siblings] perfectly determines
        the correct position of the child node, already in place or to be inserted, no matter the depth nor complexity
        of the tree.

        This method is currently called from the evolution methods, but will soon (I hope) be called from the first
        generation Tree generation methods (above) such that the same method may be used repeatedly.

        Called by: fx_evolve_child_link_fix, fx_evolve_banch_top_copy, fx_evolve_branch_body_copy

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

        Called by: fx_evolve_child_link_fix

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

            else: print ('\n\t\033[31m ERROR! In fx_evolve_child_link: node', node, 'has arity', tree[8][node]); self.fx_karoo_pause() # consider special instructions for this (pause) - 2019 06/08

        return tree


    def fx_evolve_child_link_fix(self, tree):

        '''
        In a given Tree, fix 'node_c1', 'node_c2', 'node_c3' for all nodes.

        This is required anytime the size of the array 'gp.tree' has been modified, as with both Grow and Full mutation.

        Called by: fx_evolve_grow_mutate, fx_evolve_crossover, fx_evolve_branch_body_copy, fx_evolve_branch_copy

        Arguments required: tree
        '''

        # tested 2015 06/04
        for node in range(1, len(tree[3])):

            c_buffer = self.fx_evolve_c_buffer(tree, node) # generate c_buffer for each node
            tree = self.fx_evolve_child_link(tree, node, c_buffer) # update child links for each node

        return tree


    def fx_evolve_child_insert(self, tree, node, c_buffer):

        '''
        Insert child node into the copy of a parent Tree.

        Called by: fx_evolve_branch_insert

        Arguments required: tree, node, c_buffer
        '''

        if int(tree[8][node]) == 0: # if arity = 0
            print ('\n\t\033[31m ERROR! In fx_evolve_child_insert: node', node, 'has arity 0\033[0;0m'); self.fx_karoo_pause() # consider special instructions for this (pause) - 2019 06/08

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

        else: print ('\n\t\033[31m ERROR! In fx_evolve_child_insert: node', node, 'arity > 3\033[0;0m'); self.fx_karoo_pause() # consider special instructions for this (pause) - 2019 06/08

        return tree


    def fx_evolve_parent_link_fix(self, tree):

        '''
        In a given Tree, fix 'parent_id' for all nodes.

        This is automatically handled in all mutations except with Crossover due to the need to copy branches 'a' and
        'b' to their own trees before inserting them into copies of    the parents.

        Technically speaking, the 'node_parent' value is not used by any methods. The parent ID can be completely out
        of whack and the expression will work perfectly. This is maintained for the sole purpose of granting the user
        a friendly, makes-sense interface which can be read in both directions.

        Called by: fx_evolve_branch_copy

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

        Called by: fx_evolve_grow_mutate, fx_evolve_tree_prune

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

        Called by: fx_evolve_grow_mutate, fx_evolve_crossover, fx_evolve_branch_insert, fx_evolve_branch_copy

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

        Called by: fx_nextgen_reproduce, fx_nextgen_point_mutate, fx_nextgen_full_mutate, fx_nextgen_grow_mutate, fx_nextgen_crossover

        Arguments required: tree
        '''

        tree[12][1:] = '' # wipe fitness data

        return tree


    def fx_evolve_tree_prune(self, tree, depth):

        '''
        This method reduces the depth of a Tree. Used with Crossover, the input value 'branch' can be a partial Tree
        (branch) or a full tree, and it will operate correctly. The input value 'depth' becomes the new maximum depth,
        where depth is defined as the local maximum + the user defined adjustment.

        Called by: fx_evolve_crossover

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


    def fx_evolve_pop_copy(self, pop_a, title):

        '''
        Copy one population to another.

        Simply copying a list of arrays generates a pointer to the original list. Therefore we must append each array
        to a new, empty array and then build a list of those new arrays.

        Called by: fx_karoo_gp

        Arguments required: pop_a, title
        '''

        pop_b = [title] # an empty list stores a copy of the prior generation

        for tree in range(1, len(pop_a)): # increment through each Tree in the current population

            tree_copy = np.copy(pop_a[tree]) # copy each array in the current population
            pop_b.append(tree_copy) # add each copied Tree to the new population list

        return pop_b


    #+++++++++++++++++++++++++++++++++++++++++++++
    #   Methods to Visualize a Tree              |
    #+++++++++++++++++++++++++++++++++++++++++++++

    def fx_display_tree(self, tree):

        '''
        Display all or part of a Tree on-screen.

        This method displays all sequential node_ids from 'start' node through bottom, within the given tree.

        Called by: fx_karoo_gp, fx_karoo_pause

        Arguments required: tree
        '''

        ind = ''
        print ('\n\033[1m\033[36m Tree ID', int(tree[0][1]), '\033[0;0m')

        for depth in range(0, self.tree_depth_max + 1): # increment through all possible Tree depths - tested 2016 07/09
            print ('\n', ind,'\033[36m Tree Depth:', depth, 'of', tree[2][1], '\033[0;0m')

            for node in range(1, len(tree[3])): # increment through all nodes (redundant, I know)
                if int(tree[4][node]) == depth:
                    print ('')
                    print (ind,'\033[1m\033[36m NODE:', tree[3][node], '\033[0;0m')
                    print (ind,'  type:', tree[5][node])
                    print (ind,'  label:', tree[6][node], '\tparent node:', tree[7][node])
                    print (ind,'  arity:', tree[8][node], '\tchild node(s):', tree[9][node], tree[10][node], tree[11][node])

            ind = ind + '\t'

        print ('')
        self.fx_eval_poly(tree) # generate the raw and sympified expression for the entire Tree
        print ('\t\033[36mTree', tree[0][1], 'yields (raw):', self.algo_raw, '\033[0;0m')
        print ('\t\033[36mTree', tree[0][1], 'yields (sym):\033[1m', self.algo_sym, '\033[0;0m')

        return


    def fx_display_branch(self, tree, start):

        '''
        Display a Tree branch on-screen.

        This method displays all sequential node_ids from 'start' node through bottom, within the given branch.

        Called by: This method is not used by Karoo GP at this time.

        Arguments required: tree, start
        '''

        branch = np.array([]) # the array is necessary in order to len(branch) when 'branch' has only one element
        branch_eval = self.fx_eval_id(tree, start) # generate tuple of given 'branch'
        branch_symp = sympify(branch_eval) # convert string from tuple to list
        branch = np.append(branch, branch_symp) # append list to array
        ind = ''

        # for depth in range(int(tree[4][start]), int(tree[2][1]) + self.tree_depth_max + 1): # increment through all Tree depths - tested 2016 07/09
        for depth in range(int(tree[4][start]), self.tree_depth_max + 1): # increment through all Tree depths - tested 2016 07/09
            print ('\n', ind,'\033[36m Tree Depth:', depth, 'of', tree[2][1], '\033[0;0m')

            for n in range(0, len(branch)): # increment through all nodes listed in the branch
                node = branch[n]

                if int(tree[4][node]) == depth:
                    print ('')
                    print (ind,'\033[1m\033[36m NODE:', node, '\033[0;0m')
                    print (ind,'  type:', tree[5][node])
                    print (ind,'  label:', tree[6][node], '\tparent node:', tree[7][node])
                    print (ind,'  arity:', tree[8][node], '\tchild node(s):', tree[9][node], tree[10][node], tree[11][node])

            ind = ind + '\t'

        print ('')
        self.fx_eval_poly(tree) # generate the raw and sympified expression for the entire Tree
        print ('\t\033[36mTree', tree[0][1], 'yields (raw):', self.algo_raw, '\033[0;0m')
        print ('\t\033[36mTree', tree[0][1], 'yields (sym):\033[1m', self.algo_sym, '\033[0;0m')

        return


