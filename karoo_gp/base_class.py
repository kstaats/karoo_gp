# Karoo GP Base Class
# Define the methods and global variables used by Karoo GP

'''
A NOTE TO THE NEWBIE, EXPERT, AND BRAVE
Even if you are highly experienced in Genetic Programming, it is
recommended that you review the 'Karoo User Guide' before running
this application. While your computer will not burst into flames
nor will the sun collapse into a black hole if you do not, you will
likely find more enjoyment of this particular flavour of GP with
a little understanding of its intent and design.
'''

import os
import sys
import csv
import time
import json

import numpy as np
import sklearn.metrics as skm
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  # from https://www.tensorflow.org/guide/migrate
import sklearn.model_selection as skcv

from datetime import datetime

from . import Population, Functions, Terminals, \
              fx_fitness_labels_map_maker, fx_fitness_eval


### TensorFlow Imports and Definitions ###
if os.environ.get("TF_CPP_MIN_LOG_LEVEL") is None:
    # Set the log level, unless it's already set in the env.
    # This allows users to override this value with an env var.
    # 0 = all messages are logged (default behavior)
    # 1 = INFO messages are not printed
    # 2 = INFO and WARNING messages are not printed
    # 3 = INFO, WARNING, and ERROR messages are not printed
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # only print ERRORs


# set the terminal to print 320 characters before
# line-wrapping in order to view Trees
np.set_printoptions(linewidth=320)


class Base_GP(object):

    '''
    This Base_BP class contains all methods for Karoo GP. Method names
    are differentiated from global variable names (defined below) by
    the prefix 'fx_' followed by an object and action, as in fx_display_tree(),
    with a few expections, such as fx_fitness_gene_pool().

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

    def __init__(self, kernel='m', tree_type='r', tree_depth_base=3,
                 tree_depth_max=3, tree_depth_min=1, tree_pop_max=100,
                 gen_max=10, tourn_size=7, filename='', output_dir='',
                 evolve_repro=0.1, evolve_point=0.1, evolve_branch=0.2,
                 evolve_cross=0.6, display='s', precision=6, swim='p',
                 mode='s', seed=None, pause_callback=None):

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

        self.functions          user defined functions (operators) from
                                the associated files/[functions].csv
        self.terminals          user defined variables (operands) from
                                the top row of the associated [data].csv
        self.coeff              user defined coefficients (NOT YET IN USE)
        self.fitness_type       fitness type
        self.datetime           date-time stamp of when the unique directory is created
        self.path               full path to the unique directory created with each run
        self.dataset            local path and dataset filename

        ### Global variables used for evolutionary management ###
        self.population_a       the root generation from which Trees are chosen
                                for mutation and reproduction
        self.population_b       the generation constructed from gp.population_a (recyled)
        self.gene_pool          once-per-generation assessment of trees that
                                meet min and max boundary conditions
        self.gen_id             simple n + 1 increment
        self.fitness_type       set in fx_data_load() as either a minimising
                                or maximising function
        self.tree               axis-1, 13 element Numpy array that defines
                                each Tree, stored in 'gp.population'
        self.pop_*              13 variables that define each Tree -
                                see fx_init_tree_initialise()
        '''

        # the raw expression generated by Sympy per Tree --
        # CONSIDER MAKING THIS VARIABLE LOCAL
        self.algo_raw = []
        # the expression generated by Sympy per Tree --
        # CONSIDER MAKING THIS VARIABLE LOCAL
        self.algo_sym = []
        self.fittest_dict = {}  # all Trees which share the best fitness score
        self.gene_pool = []  # store all Tree IDs for use by Tournament
        self.class_labels = 0  # the number of true class labels (data_y)

        # API REFACTOR: Relocated from fx_karoo_gp

        ### PART 1 - set global variables to those local values passed from the user script ###
        self.kernel = kernel  # fitness function
        self.tree_type = tree_type  # passed between methods to construct specific trees
        self.tree_depth_base = tree_depth_base # passed between methods to construct specific trees
        self.tree_depth_max = tree_depth_max  # maximum Tree depth for the entire run; limits bloat
        self.tree_depth_min = tree_depth_min  # minimum number of nodes
        self.tree_pop_max = tree_pop_max  # maximum number of Trees per generation
        self.gen_max = gen_max  # maximum number of generations
        self.tourn_size = tourn_size  # number of Trees selected for each tournament
        self.filename = filename  # passed between methods to work with specific populations
        self.output_dir = output_dir  # determines where output records are saved
        self.evolve_repro = evolve_repro  # quantity of a population generated through Reproduction
        self.evolve_point = evolve_point  # quantity of a population generated through Point Mutation
        self.evolve_branch = evolve_branch  # quantity of a population generated through Branch Mutation
        self.evolve_cross = evolve_cross  # quantity of a population generated through Crossover
        self.display = display  # display mode is set to (s)ilent # level of on-screen feedback
        self.precision = precision  # the number of floating points for the round function in 'fx_fitness_eval'
        self.swim = swim  # pass along the gene_pool restriction methodology
        self.mode = mode  # mode is engaged in fit()
        self.pause_callback = pause_callback

        # initialize RNG(s) with the given seed
        self.seed = seed
        self.rng = np.random.default_rng(seed)  # this is used by Karoo
        # the following two are not used, set them just in case
        np.random.seed(seed)  # this was used by sklearn while classifying
        tf.set_random_seed(seed)

        ### PART 2 - construct first generation of Trees ###
        self.fx_data_load(filename)
        self.functions = Functions([f[0] for f in self.functions])  # Symbol only
        self.terminals = Terminals(self.terminals[:-1])

        self.fx_fitness_labels_map = fx_fitness_labels_map_maker(self.class_labels)

        self.log(f'\n\t\033[32m Press \033[36m\033[1m?\033[0;0m\033[32m at any '
                 f'\033[36m\033[1m(pause)\033[0;0m\033[32m, or '
                 f'\033[36m\033[1mENTER\033[0;0m \033[32mto continue the run\033[0;0m',
                 display=['i'])
        self.pause(display=['i'])

        # initialise population_a to host the first generation
        self.population = Population.generate(
            log=self.log, pause=self.pause, error=self.error,
            tree_type=self.tree_type, tree_depth_base=self.tree_depth_base,
            tree_depth_max=tree_depth_max, tree_pop_max=tree_pop_max,
            functions=self.functions, terminals=self.terminals, rng=self.rng,
            fitness_type=self.fitness_type
        )

        self.log(f'\n We have constructed the first, stochastic population of'
                 f'{self.tree_pop_max} Trees'
                 f'\n Evaluate the first generation of Trees ...')

        if self.kernel == 'p':
            self.fx_data_tree_write(self.population.trees, 'a')
            sys.exit()

        self.population.evaluate(
            log=self.log, pause=self.pause, error=self.error,
            data_train=self.data_train, kernel=self.kernel,
            data_train_rows=self.data_train_rows,
            tf_device_log=self.tf_device_log, class_labels=self.class_labels,
            tf_device=self.tf_device, terminals=self.terminals,
            precision=self.precision, savefile=self.savefile,
            fx_data_tree_write=self.fx_data_tree_write
        )


    def log(self, msg, display={'i', 'g', 'm', 'db'}):
        if self.display in display or display == 'all':
            print(msg)

    def pause(self, display={'i', 'g', 'm', 'db'}):
        if not self.pause_callback:
            self.log('No pause callback function provided')
            return
        if self.display in display or display is None:
            self.pause_callback(self)

    def error(self, msg, display={'i', 'g', 'm', 'db'}):
        self.log(msg, display)
        self.pause(display)


    #+++++++++++++++++++++++++++++++++++++++++++++
    #   Methods to Run Karoo GP                  |
    #+++++++++++++++++++++++++++++++++++++++++++++

    def fit(self, X=None, y=None):

        '''
        This method enables the engagement of the entire Karoo GP application.
        Instead of returning the user to the pause menu, this script terminates
        at the command-line, providing support for bash and chron job execution.

        Calld by: user script karoo_gp.py

        Arguments required: (see below)
        '''

        ### PART 4 - evolve multiple generations of Trees ###
        menu = 1
        while menu != 0:
            # this allows the user to add generations mid-run and
            # not get buried in nested iterations
            for gen in range(self.population.gen_id, self.gen_max):
                self.log(f'\n Evolve a population of Trees for Generation {gen + 1} ...')
                self.population = self.population.evolve(
                    self.swim, self.tree_depth_min, self.functions,
                    self.terminals, self.evolve_repro, self.evolve_point,
                    self.evolve_branch, self.evolve_cross, self.tree_pop_max,
                    self.tourn_size, self.precision, self.fitness_type,
                    self.tree_depth_max, self.data_train, self.kernel,
                    self.data_train_rows, self.tf_device_log, self.class_labels,
                    self.tf_device, self.savefile, self.rng, self.log,
                    self.pause, self.error, self.fx_data_tree_write,
                )
                self.log('\n Copy gp.population_b to gp.population_a\n')

            if self.mode == 's':
                # (s)erver mode - termination with completiont of prescribed run
                menu = 0
            else:
                # (d)esktop mode - user is given an option to quit, review,
                # and/or modify parameters; 'add' generations continues the run
                self.log('\n\t\033[32m Enter \033[1m?\033[0;0m\033[32m to review '
                         'your options or \033[1mq\033[0;0m\033[32muit\033[0;0m')
                menu = self.pause()

    def fx_karoo_terminate(self):
        '''
        Terminates the evolutionary run (if yet in progress),
        saves parameters and data to disk, and cleanly returns
        the user to karoo_gp.py and the command line.

        Called by: fx_karoo_gp() and fx_karoo_pause_refer()

        Arguments required: none
        '''

        self.fx_data_params_write()
        self.fx_data_params_write_json()
        # initialize the .csv file for the final population
        target = open(self.savefile['f'], 'w')
        target.close()
        # save the final generation of Trees to disk
        self.fx_data_tree_write(self.population.population_b, 'f')
        self.log('\n\t\033[32m Your Trees and runtime parameters are archived '
                 f'in {self.savefile["f"]}/\033[0;0m')

        self.log('\n\033[3m "It is not the strongest of the species that '
                 'survive, nor the most intelligent,\033[0;0m\n'
                 '\033[3m  but the one most responsive to change."'
                 '\033[0;0m --Charles Darwin\n\n'
                 '\033[3m Congrats!\033[0;0m Your Karoo GP run is complete.\n')
        sys.exit()


    #+++++++++++++++++++++++++++++++++++++++++++++
    #   Methods to Load and Archive Data         |
    #+++++++++++++++++++++++++++++++++++++++++++++

    def fx_data_load(self, filename):

        '''
        The data and function .csv files are loaded according to the fitness
        function kernel selected by the user. An alternative dataset may be
        loaded at launch, by appending a command line argument. The data is
        then split into both TRAINING and TEST segments in order to validate
        the success of the GP training run. Datasets less than 10 rows will
        not be split, rather copied in full to both TRAINING and TEST as it
        is assumed you are conducting a system validation run, as with the
        built-in MATCH kernel and associated dataset.

        Called by: fx_karoo_gp

        Arguments required: filename (of the dataset)
        '''

        ### PART 1 - load the associated data set, operators, operands, fitness type, and coefficients ###
        # for user Marco Cavaglia:
        #   full_path = os.path.realpath(__file__)
        #   karoo_dir = os.path.dirname(full_path)
        karoo_dir = os.path.dirname(os.path.realpath(__file__))

        data_dict = {
            'c': karoo_dir + '/files/data_CLASSIFY.csv',
            'r': karoo_dir + '/files/data_REGRESS.csv',
            'm': karoo_dir + '/files/data_MATCH.csv',
            'p': karoo_dir + '/files/data_PLAY.csv',
        }

        filename = filename or data_dict[self.kernel]

        data_x = np.loadtxt(filename, skiprows=1, delimiter=',', dtype=float)
        data_x = data_x[:,0:-1]  # load all but the right-most column
        # load only right-most column (class labels)
        data_y = np.loadtxt(filename, skiprows=1, usecols=(-1,),
                            delimiter=',', dtype=float)
        header = open(filename, 'r')  # open file to be read (below)
        self.dataset = filename  # copy the name only

        fitt_dict = {'c': 'max', 'r': 'min', 'm': 'max', 'p': ''}
        self.fitness_type = fitt_dict[self.kernel]  # load fitness type

        func_dict = {
            'c': karoo_dir + '/files/operators_CLASSIFY.csv',
            'r': karoo_dir + '/files/operators_REGRESS.csv',
            'm': karoo_dir + '/files/operators_MATCH.csv',
            'p': karoo_dir + '/files/operators_PLAY.csv',
        }
        # load the user defined functions (operators)
        self.functions = np.loadtxt(func_dict[self.kernel], delimiter=',',
                                    skiprows=1, dtype=str)
        # load the user defined terminals (operands)
        self.terminals = header.readline().split(',')
        self.terminals[-1] = self.terminals[-1].replace('\n', '')
        # load the user defined true labels for classification or
        # solutions for regression
        self.class_labels = len(np.unique(data_y))
        # load the user defined coefficients - NOT USED YET
        # self.coeff = np.loadtxt(karoo_dir + '/files/coefficients.csv',
        #                         delimiter=',', skiprows=1, dtype=str)

        ### PART 2 - from the dataset, extract TRAINING and TEST data ###
        if len(data_x) < 11:
            # for small datasets we will not split them into TRAINING and
            # TEST components
            data_train = np.c_[data_x, data_y]
            data_test = np.c_[data_x, data_y]

        else:
            # if larger than 10, we run the data through the
            # SciKit Learn's 'random split' function
            x_train, x_test, y_train, y_test = skcv.train_test_split(
                data_x, data_y, test_size=0.2, random_state=self.seed
            )  # 80/20 TRAIN/TEST split
            data_x, data_y = [], []  # clear from memory

            # recombine each row of data with its associated class label (right column)
            data_train = np.c_[x_train, y_train]
            x_train, y_train = [], []  # clear from memory

            # recombine each row of data with its associated class label (right column)
            data_test = np.c_[x_test, y_test]
            x_test, y_test = [], []  # clear from memory

        self.data_train_cols = len(data_train[0,:])  # qty count
        self.data_train_rows = len(data_train[:,0])  # qty count
        self.data_test_cols = len(data_test[0,:])  # qty count
        self.data_test_rows = len(data_test[:,0])  # qty count

        ### PART 3 - load TRAINING and TEST data for TensorFlow processing - tested 2017 02/02 ###
        self.data_train = data_train  # Store train data for processing in TF
        self.data_test = data_test  # Store test data for processing in TF
        # Set TF computation backend device (CPU or GPU);
        # gpu:n = 1st, 2nd, or ... GPU device
        self.tf_device = "/gpu:0"
        self.tf_device_log = False  # TF device usage logging (for debugging)

        ### PART 4 - create a unique directory and initialise all .csv files ###
        self.datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
        # generate a unique directory name
        runs_dir = os.path.join(os.getcwd(), 'runs')
        if self.output_dir:
            self.path = os.path.join(runs_dir, self.output_dir + '/')
        else:
            basename = os.path.basename(filename)  # extract the filename (if any)
            root, ext = os.path.splitext(basename)  # split root from extension
            self.path = os.path.join(runs_dir,
                                     root + '_' + self.datetime + '/')

        if not os.path.isdir(self.path):
            os.makedirs(self.path)  # make a unique directory

        self.savefile = {}  # a dictionary to hold .csv filenames

        self.savefile.update({'a': self.path + 'population_a.csv'})
        # initialise a .csv file for population 'a' (foundation)
        target = open(self.savefile['a'], 'w')
        target.close()

        self.savefile.update({'b': self.path + 'population_b.csv'})
        # initialise a .csv file for population 'b' (evolving)
        target = open(self.savefile['b'], 'w')
        target.close()

        self.savefile.update({'f': self.path + 'population_f.csv'})
        # initialise a .csv file for the final population (test)
        target = open(self.savefile['f'], 'w')
        target.close()

        self.savefile.update({'s': self.path + 'population_s.csv'})
        # initialise a .csv file to manually load (seed)
        target = open(self.savefile['s'], 'w')
        target.close()


    def fx_data_recover(self, population):

        '''
        This method is used to load a saved population of Trees,
        as invoked through the (pause) menu where population_r
        replaces population_a in the karoo_gp/runs/[date-time]/ directory.

        Called by: fx_karoo_pause

        Arguments required: population (filename['s'])

        TODO: MAY REDESIGN need up to update
        '''

        with open(population, 'rb') as csv_file:
            target = csv.reader(csv_file, delimiter=',')
            n = 0  # track row count

            for row in target:
                self.log(f'row {row}')

                n = n + 1
                if n == 1:
                    pass  # skip first empty row

                elif n == 2:
                    self.population_a = [row]  # write header to population_a

                else:
                    if row == []:
                        self.tree = np.array([[]])  # initialise Tree array
                    else:
                        if self.tree.shape[1] == 0:
                            # append first row to Tree
                            self.tree = np.append(self.tree, [row], axis=1)
                        else:
                            # append subsequent rows to Tree
                            self.tree = np.append(self.tree, [row], axis=0)

                    if self.tree.shape[0] == 13:
                        # append complete Tree to population list
                        self.population_a.append(self.tree)

        self.log(f'\n{self.population_a}')

    # used by: None
    def fx_data_tree_append(self, tree):

        '''
        Append Tree array to the foundation Population.

        Called by: fx_init_construct

        Arguments required: tree
        '''

        self.fx_data_tree_clean(tree)  # clean 'tree' prior to storing
        self.population_a.append(tree)  # append 'tree' to population list


    def fx_data_tree_write(self, trees, key):

        '''
        Save population_* to disk.

        Called by: fx_karoo_gp, fx_eval_generation

        Arguments required: population, key
        '''

        with open(self.savefile[key], 'a') as csv_file:
            target = csv.writer(csv_file, delimiter=',')
            if self.population.gen_id != 1:
                target.writerows([''])  # empty row before each generation
            target.writerows([['Karoo GP by Kai Staats', 'Generation:',
                               str(self.population.gen_id)]])

            for tree in trees:
                target.writerows([tree.save()])
                # target.writerows([''])  # empty row before each Tree
                # for row in range(0, 13):
                #     # increment through each row in the array Tree
                #     target.writerows([tree.root[row]])


    def fx_eval_fittest(self):
        '''
        Re-evaluate all Trees to find the fittest.
        '''
        fitness_best = 0
        fittest_tree = None

        # revised method, re-evaluating all Trees from stored fitness score
        for i, tree in enumerate(self.population.trees):

            fitness = tree.fitness

            if self.kernel == 'c':  # display best fit Trees for the CLASSIFY kernel
                # find the Tree with Maximum fitness score
                if fitness >= fitness_best:
                    # set best fitness Tree
                    fitness_best = fitness
                    fittest_tree = tree

            elif self.kernel == 'r':  # display best fit Trees for the REGRESSION kernel
                if fitness_best == 0:
                    fitness_best = fitness  # set the baseline first time through
                # find the Tree with Minimum fitness score
                if fitness <= fitness_best:
                    # set best fitness Tree
                    fitness_best = fitness
                    fittest_tree = tree

            elif self.kernel == 'm':  # display best fit Trees for the MATCH kernel
                # find the Tree with a perfect match for all data rows
                if fitness == self.data_train_rows:
                    # set best fitness Tree
                    fitness_best = fitness
                    fittest_tree = tree

            # elif self.kernel == '[other]':  # use others as a template

            # print('fitness_best:', fitness_best, 'fittest_tree:', fittest_tree)

        # get simplified expression and process it by TF - tested 2017 02/02
        fittest_tree.result = fx_fitness_eval(
            fittest_tree.expression, self.data_test, self.tf_device_log, self.kernel,
            self.class_labels, self.tf_device, self.terminals,
            self.fx_fitness_labels_map, get_pred_labels=True
        )
        return fittest_tree


    # tested 2017 02/13; argument 'app' removed to simplify termination 2019 06/08
    def fx_data_params_write(self):

        '''
        Save run-time configuration parameters to disk.

        Called by: fx_karoo_terminate

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
        file.write('\n number of generations: ' + str(self.population.gen_id))
        file.write('\n\n')
        file.close()


        file = open(self.path + 'log_test.txt', 'w')
        file.write('Karoo GP')
        file.write('\n launched: ' + str(self.datetime))
        file.write('\n dataset: ' + str(self.dataset))
        file.write('\n')

        # Which population the fittest_dict indexes refer to
        if len(self.population.fittest_dict) == 0:
            file.write('\n\n There were no evolved solutions generated in '
                       'this run... your species has gone extinct!')

        else:
            fittest = self.population.fittest()
            file.write(f'\n\n Tree {fittest.id} is the most fit, with '
                       f'expression:\n\n {fittest.expression}')

            result = fx_fitness_eval(
                fittest.expression, self.data_test, self.tf_device_log, self.kernel,
                self.class_labels, self.tf_device, self.terminals,
                self.fx_fitness_labels_map, get_pred_labels=True
            )

            if self.kernel == 'c':
                file.write(f'\n\n Classification fitness score: {result["fitness"]}')
                report = skm.classification_report(result["solution"],
                                                   result["pred_labels"][0])
                file.write(f'\n\n Precision-Recall report:\n {report}')
                matrix = skm.confusion_matrix(result['solution'],
                                              result['pred_labels'][0])
                file.write(f'\n Confusion matrix:\n {matrix}')

            elif self.kernel == 'r':
                MSE = skm.mean_squared_error(result['result'], result['solution'])
                fitness = result['fitness']
                file.write(f'\n\n Regression fitness score: {fitness}')
                file.write(f'\n Mean Squared Error: {MSE}')

            elif self.kernel == 'm':
                file.write(f'\n\n Matching fitness score: {result["fitness"]}')

            # elif self.kernel == '[other]':  # use others as a template

        file.write('\n\n')
        file.close()

        return


    def fx_data_params_write_json(self):
        '''
        Save run-time configuration parameters to disk as json.

        Called by: fx_karoo_terminate
        '''
        generic = dict(
            package='Karoo GP',
            launched=self.datetime,
            dataset=self.dataset,
        )
        config = dict(
            kernel=self.kernel,
            precision=self.precision,

            tree_type=self.tree_type,
            tree_depth_base=self.tree_depth_base,
            tree_depth_max=self.tree_depth_max,
            min_node_count=self.tree_depth_min,

            genetic_operators=dict(
                reproduction=self.evolve_repro,
                point_mutation=self.evolve_point,
                branch_mutation=self.evolve_branch,
                crossover=self.evolve_cross,
            ),

            tournament_size=self.tourn_size,
            population=self.tree_pop_max,
            number_of_generations=self.population.gen_id,
        )

        final_dict = dict(**generic, config=config)

        if len(self.population.fittest_dict) == 0:
            final_dict['outcome'] = 'FAILURE'
        else:
            tree = self.fx_eval_fittest()
            fittest_tree = dict(
                id=tree.id,
                expression=tree.expression,
            )
            result = tree.result
            score = dict(fitness=result['fitness'])
            if self.kernel == 'c':
                score['classification_report'] = skm.classification_report(
                    result['solution'], result['pred_labels'][0],
                    output_dict=True,
                )
                score['confusion_matrix'] = skm.confusion_matrix(
                    result['solution'], result['pred_labels'][0]
                ).tolist()

            elif self.kernel == 'r':
                MSE = skm.mean_squared_error(result['result'], result['solution'])
                score['mean_squared_error'] = float(MSE)

            final_dict = dict(
                **final_dict,
                outcome='SUCCESS',
                fittest_tree=fittest_tree,
                score=score,
            )

        with open(self.path + 'results.json', 'w') as f:
            json.dump(final_dict, f, indent=4)
