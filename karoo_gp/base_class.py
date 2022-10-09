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

import sys
import csv
import json
import operator
from pathlib import Path

import numpy as np
import sklearn.metrics as skm
import sklearn.model_selection as skcv

from sklearn.utils import check_random_state, check_X_y
from sklearn.base import BaseEstimator, TransformerMixin

from datetime import datetime

from . import Population, Tree, NodeData, get_function_node, get_nodes


class BaseGP(BaseEstimator):

    """
    This class contains the core attributes, objects and methods of Karoo GP.
    It's composed of a heirarchal structure of classes. Below are some of the
    more import attributes and methods for each class, and their organization:

    BaseGP
    ├─ .scoring = {field: func...}      - funcs are passed (y_true, y_pred)
    ├─ .nodes = Nodes                   - all active terminals, constants, fx
    ├─ .decoder(y)                      - Initialize with y_train
    │   └─ .transform(pred)             - transforms prediction to match y
    │
    ├─ .fitness_compare(a, b)           - determines the fitter of two trees
    |
    ├─ .get_nodes(types, depth)         - return nodes matching types & depth
    │
    ├─ .population = Population         - an isolated group of trees
    │   ├─ .fittest = Tree              - return the fittest tree
    │   ├─ .trees                       - list of Trees, length tree_pop_max
    │   │   ├─ ...
    │   │   └─ Tree                     - an evolvable expression tree
    │   │      ├─ .expression           - a string of sympified expr, e.g. a*b
    │   │      ├─ .score                - a dict of results matching `scoring`
    │   │      ├─ .root = Node          - a recursive node which forms a Tree
    │   │      │   ├─ .label            - a terminal ('a') or function ('*')
    │   │      │   ├─ .arity            - instructions for generating children
    │   │      │   ├─ .parent           - the node immediately above
    │   │      │   └─ .children         - the nodes immediately below
    │   │      │       └─ ...
    │   │      └─ mutate, crossover...  - methods used by population.evolve
    │   │
    │   ├─ .evaluate(X, y)              - predict and score trees
    │   └─ .evolve()                    - return a new generation of trees
    │
    ├─ .fit(X, y)                       - evolve expressions to predict y
    ├- .predict(X)                      - return predicted y values for X
    └- .score(pred, y)                  - return score of prediction against y
    """

    # Overridden by subclasses
    kernel = 'b'  # Base

    # Fit parameters (set later)
    scoring_ = None
    history_ = None
    datetime = None
    path = None
    X_hash_ = None
    test_split_ = None
    nodes = None
    population = None
    cache_ = None
    unfit = None  # Keep a record of trees which were marked unfit

    def __init__(
        self, tree_type='r', tree_depth_base=3, tree_depth_max=None,
        tree_depth_min=1, tree_pop_max=100, gen_max=10, tourn_size=7,
        filename='', output_dir='', evolve_repro=0.1, evolve_point=0.1,
        evolve_branch=0.2, evolve_cross=0.6, display='s', precision=None,
        swim='p', mode='s', random_state=None, pause_callback=None,
        engine_type='numpy', tf_device="/gpu:0", tf_device_log=False,
        functions=None, force_types=[['operator', 'cond']], terminals=None,
        constants=None, test_size=0.2, scoring=None, higher_is_better=False,
        prediction_transformer=None, cache=None):
        """Initialize a Karoo_GP object with given parameters"""

        # Model parameters
        self.tree_type = tree_type           # (f)ull, (g)row or (r)amped 50/50
        self.tree_depth_base = tree_depth_base # depth of initial population
        self.tree_depth_max = tree_depth_max # max allowed depth
        # TODO: This should be renamed 'tree_min_nodes' because it restricts the
        # gene pool based on the number of nodes, not the depth.
        self.tree_depth_min = tree_depth_min # min allowed number of nodes
        self.tree_pop_max = tree_pop_max     # number of trees per generation
        self.gen_max = gen_max               # number of generations to evolve
        self.tourn_size = tourn_size         # number of Trees per tournament
        self.filename = filename             # prefix for output files
        self.output_dir = output_dir         # path to output directory
        self.evolve_repro = evolve_repro     # ratio of next_gen reproduced
        self.evolve_point = evolve_point     # ratio of next_gen point mutated
        self.evolve_branch = evolve_branch   # ratio of next_gen branch mutated
        self.evolve_cross = evolve_cross     # ratio of next_gen made by crossover
        self.display = display               # determines when pause/log called
        self.precision = precision           # max decimal places. pred & score
        self.swim = swim                     # culling method
        self.mode = mode                     # determines if pauses after fit
        self.random_state = random_state     # follows sklearn convention
        self.pause_callback = pause_callback # called throughout based on disp
        self.engine_type = engine_type       # execute on cpu (np) or gpu (tf)
        self.tf_device = tf_device           # configure gpu
        self.tf_device_log = tf_device_log   # log dir for tensorflow
        self.functions = functions           # list of operators to use
        self.force_types = force_types       # default: root = operator/cond
        self.terminals = terminals           # list of terminal names to use
        self.constants = constants           # list of constants to include
        self.test_size = test_size           # how to portion train/test data
        self.scoring = scoring               # a dict of name/func pairs
        # Whether fitness_compare returns the higher- or lower-fitness tree
        self.higher_is_better = higher_is_better
        # A function called on the tree outputs, e.g. round/clip for classify
        self.prediction_transformer = prediction_transformer
        self.cache = cache                   # a dict of expr/score pairs

    def log(self, msg, display={'i', 'g', 'm', 'db'}):
        """Print a message to the console when in specified display mode"""
        if self.display in display or display == 'all':
            print(msg)

    def pause(self, display={'i', 'g', 'm', 'db'}):
        """Call the pause_callback when in specified display mode"""
        if not self.pause_callback:
            self.log('No pause callback function provided')
            return 0
        elif self.display in display:
            self.pause_callback(self)
        else:
            return 0

    def error(self, msg, display={'i', 'g', 'm', 'db'}):
        """Print an error message to the console when in display mode"""
        self.log(msg, display)
        self.pause(display)

    def log_history(self):
        """Add the score of the current fittest tree to history"""
        def recursively_merge_history(a, b):
            """Concatenate values in multi-level dict to history"""
            for k, v in a.items():
                if isinstance(v, dict):
                    if k not in b:
                        b[k] = {}
                    recursively_merge_history(v, b[k])
                elif isinstance(v, (int, float, str, list)):
                    if k not in b:
                        b[k] = []
                    b[k].append(v)
                else:
                    raise ValueError('Scoring must be a single- or multi-level'
                                     ' dict of int, float, str or list values')
        score = self.score(self.X_test, self.y_test)
        recursively_merge_history(score, self.history_)

    def save_population(self, population):
        """Write population to csv following default instructions.

        population:
        'a': Append current trees to 'population_a.csv'
        'b': Write next_gen_trees to 'population_b.csv' (overwrite)
        'f': Write final trees to 'population_f' (overwrite);
             called by terminate()
        's': Write current trees to 'population_s' (overwrite);
             called in interactive mode to edit manually and re-load

        TODO: Generalize: save_population(fname, next_gen=False, append=False).
        Move the current schema/logic to karoo-gp.py, and replace calls with
        lifecycle 'hooks' passed to BaseGP, e.g.:

        callbacks={end_of_generation: lambda model: model.save_population(...)}

        """

        # Select trees to save
        if self.population is None:  # Used to initialize empty csv's
            pop = []
        elif population in {'a', 'f', 's'}:
            gen_id = self.population.gen_id
            pop = [f'Karoo GP by Kai Staats - Generation {gen_id}']
            pop += self.population.save()
        elif population == 'b':
            pop = self.population.save(next_gen=True)
        else:
            raise ValueError(f'Unrecognized population: {population}')

        # Select the appropriate file
        fname = f'population_{population}.csv'
        mode = 'a' if population == 'a' else 'w'
        with open(self.path / fname, mode) as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerows([[p] for p in pop])
            # Add extra line after generations
            if (population == 'a' and self.population and
                self.population.gen_id > 0):
                writer.writerow('')
        return self.path / fname

    def load_population(self, path=None):
        """Replace current population with `population_s.csv` from output_dir

        TODO: This usually thows an error in interactive mode. We currently let
        the user save/load at any point of the population lifecycle. This means
        if e.g. you save a population before it's evaluated, and try to load
        if during evolution, the saved trees won't include fitness and can't
        be compared. This should be fixed by only allowing save/load at certain
        points in the lifecycle.

        TODO: Make more generic. Take the name as a parameter, pass
        'population_s' in karoo_gp.py.
        """
        if path is None:
            path = self.path / 'population_s.csv'
        gen_id = None
        trees = []
        with open(path) as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                item = row[0]
                if item.startswith('Karoo'):  # First line
                    gen_id = item.split('Generation ')[1]
                elif not item:
                    break
                else:
                    trees.append(item)
        self.population = Population.load(self, trees, int(gen_id))

    #+++++++++++++++++++++++++++++++++++++++++++++
    #   Methods to Run Karoo GP                  |
    #+++++++++++++++++++++++++++++++++++++++++++++

    def fitness_compare(self, a: Tree, b: Tree) -> Tree:
        """Return the fitter of two Trees, or the latter if equal

        TODO: Would make sense for trees to be directly comparable (a > b)
        but they don't hold a reference to BaseGP, and the user should be
        able to override this (default) compare_fitness function.
        """
        if a.is_unfit:
            return b  # Search for the first fit tree
        elif b.is_unfit:
            return a  # Skip unfit trees thereafter
        op = operator.gt if self.higher_is_better else operator.lt
        return a if op(a.score['fitness'], b.score['fitness']) else b

    def build_fittest_dict(self, trees):
        """Trees with equal or better fitness as prior best, from Tree 1-N

        TODO: This should probably return all the trees with the highest score,
        rather than accumulating as it goes through the lists, but it will be
        test-breaking.
        """
        fittest_dict = {}
        last_fittest = None
        def _fitter(t):
            return self.fitness_compare(last_fittest, t).id == t.id
        for tree in trees:
            if last_fittest is None or _fitter(tree):
                fittest_dict[tree.id] = tree.expression
                last_fittest = tree
        return fittest_dict

    #+++++++++++++++++++++++++++++++++++++++++++++
    #   'Check' Functions                        |
    #+++++++++++++++++++++++++++++++++++++++++++++

    # Following the sklearn convention, BaseGP.fit(X, y) validates all model
    # attributes passed to __init__ and/or updated manually, well as X and y.
    # Some sklearn library functions are used (e.g. check_X_y), others are
    # custom and packaged into the methods below.

    def check_model(self):
        """Initialize and/or validate model parameters"""

        if self.scoring_ is None:
            # RNG
            self.rng = check_random_state(self.random_state)  # TODO: rename `random_state_` (skl)

            # Scoring: optionally overwritten by child class
            self.scoring_ = (self.scoring if self.scoring is not None else
                             dict(fitness=skm.mean_absolute_error))
            self.history_ = {}
            # Engine
            self.cache_ = self.cache or {}
            if self.engine_type not in ('numpy', 'tensorflow'):
                raise ValueError(f'Unrecognized engine_type: {self.engine_type}')

            # File Manager
            self.datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
            runs_dir = Path.cwd() / 'runs'
            runs_dir.mkdir(exist_ok=True)
            if self.output_dir:
                self.path = runs_dir / self.output_dir
            else:
                kname = dict(b='BASE', r='REGRESS',
                             c='CLASSIFY', m='MATCH')[self.kernel]
                self.path = runs_dir / f'data_{kname}_{self.datetime}/'
            if not self.path.exists():    # initialize elog dir
                self.path.mkdir()
                for pop in ['a', 'b', 'f', 's']:  # initialize log files
                    self.save_population(pop)

    def check_population(self, X, y):
        """Initialize and/or validate population parameters"""
        # Nodes
        if self.nodes is None:

            # Terminals
            if self.terminals is None:
                terms = [f'f{i}' for i in range(X.shape[1])]
                self.terminals = terms
            else:
                if not isinstance(self.terminals, list):
                    raise ValueError('Terminals must be a list, got',
                                    type(self.terminals))
                elif not all(isinstance(t, str) for t in self.terminals):
                    raise ValueError('Terminal list items must be strings.')
                elif len(self.terminals) != X.shape[1]:
                    raise ValueError('Terminals list must be the same length'
                                     'as X samples.')
                terms = self.terminals
            terminals = [NodeData(t, 'terminal') for t in terms]

            # Constants
            if self.constants is None:
                constants = []
            else:
                if (not isinstance(self.constants, list) or
                    not all(isinstance(c, (int, float)) for c in self.constants)):
                    raise ValueError('constants must be a list of ints or floats')
                constants = [NodeData(c, 'constant') for c in self.constants]

            # Functions
            function_labels = self.functions or ['+', '-', '*', '/', '**']
            functions = [get_function_node(f) for f in function_labels]

            self.nodes = terminals + constants + functions

        # Population
        if self.population is None:
            self.population = Population.generate(
                model=self,
                tree_type=self.tree_type,
                tree_depth_base=self.tree_depth_base,
                tree_pop_max=self.tree_pop_max,
                force_types=self.force_types,
            )
            self.log(f'\n We have constructed the first, stochastic population of '
                     f'{self.tree_pop_max} Trees.')
            self.population.evaluate(X, y, self.X_train_hash)
            self.log_history()
            self.save_population('a')

        # Update max allowed depth
        if self.tree_depth_max is None:
            self.tree_depth_max_ = self.tree_depth_base
        elif self.tree_depth_max >= self.tree_depth_base:
            self.tree_depth_max_ = self.tree_depth_max
        else:
            raise ValueError(f'Max depth ({self.tree_depth_max}) must be '
                             f'greater than or equal to base depth ('
                             f'{self.tree_depth_base})')

    def check_test_split(self, X, y):
        """Split train/test data; reuse subsequently for same X, y

        TODO: It's probably wasteful to store X_train, etc. directly, and it's
        counter to the sklearn guidelines anyway. Instead we should store a
        boolean array of which values (if any) are test samples. This creates
        an issue with the interactive mode though, because the menu needs to
        be able to 'grab' X and y in order to 'eval' a tree, and currently it's
        only passed the model.
        """
        # Store a fingerprint of the dataset (X_hash) and cache the results
        # of `train_test_split`. If `fit()` is called muptiple times with the
        # same X, y, the same split will be used. Otherwise, split new data.
        X_hash = hash(X.data.tobytes())
        if self.X_hash_ == X_hash:
            # Load previously-split train/test data
            X_train, y_train = self.X_train, self.y_train
            X_test, y_test = self.X_test, self.y_test
        else:
            # For small datasets, it doesn't make sense to split.
            if len(X) < 11 or not self.test_size:
                X_train = X_test = X.copy()
                y_train = y_test = y.copy()
            # Split train and test sets
            else:
                X_train, X_test, y_train, y_test = skcv.train_test_split(
                    X, y, test_size=self.test_size, random_state=self.random_state)
            # Save fingerprint and train/test sets
            self.X_hash_ = X_hash
            self.X_train, self.y_train = X_train, y_train
            self.X_test, self.y_test = X_test, y_test

        # Initialize hash for training set
        self.X_train_hash = hash(self.X_train.data.tobytes())
        if self.X_train_hash not in self.cache_:
            self.cache_[self.X_train_hash] = {}

        return self.X_train, self.X_test, self.y_train, self.y_test

    def fit(self, X=None, y=None):
        """Evolve a population of trees based on training data"""

        self.log('Press ? to see options, or ENTER to continue the run',
                 display=['i'])
        self.pause(display=['i'])

        # Initialize model and all variables
        self.check_model()
        X, y = check_X_y(X, y)
        X_train, X_test, y_train, y_test = self.check_test_split(X, y)
        self.check_population(X_train, y_train)

        menu = 1
        while menu != 0:  # Supports adding generations mid-run
            for gen in range(self.population.gen_id, self.gen_max):

                # Evolve the next generation
                self.population = self.population.evolve(
                    self.tree_pop_max,
                    self.swim,
                    self.tree_depth_min,
                    self.tree_depth_max_,
                    self.tourn_size,
                    self.evolve_repro,
                    self.evolve_point,
                    self.evolve_branch,
                    self.evolve_cross,
                )

                # Evaluate new generation
                self.population.evaluate(X_train, y_train, self.X_train_hash)

                # Add best score to history
                self.log_history()
                self.save_population('a')

            if self.mode == 's':  # (s)erver mode: terminate after run
                menu = 0
            else:                 # (d)esktop mode: pause after run
                self.log('Enter ? to review your options or q to quit')
                menu = self.pause()

    def predict(self, X):
        """Return predicted y values for X using the fittest tree

        Primarily used externally, e.g. model.predict(X_test)
        """
        if not self.population.evaluated:
            tree = self.population.trees[0]
        else:
            tree = self.population.fittest()
        return self.tree_predict(tree, X)


    def tree_predict(self, tree, X):
        """Return predicted values for y given X for a single tree

        * If prediction_transformer (e.g. MultiClassifier), transform predictions
        * If precision is specified, round predictions to precision

        Primarily used internally by population during evaluation
        """
        y = tree.predict(X, self.terminals, self.engine_type)
        if self.prediction_transformer is not None:
            y = self.prediction_transformer.transform(y)
        if self.precision is not None:
            y = np.round(y, self.precision)
        return y

    def score(self, X, y):
        """Return score of the fittest tree on X and y"""
        return self.calculate_score(self.predict(X), y)

    def calculate_score(self, y_pred, y_true):
        """Return a dict with the results of each scoring function"""
        return {label: fx(y_true, y_pred)
                for label, fx in self.scoring_.items()}

    def get_nodes(self, *args, **kwargs):
        """Returns a subset of self.nodes based on node_types and depth"""
        return get_nodes(*args, **kwargs, lib=self.nodes)

    def fx_karoo_terminate(self):
        '''
        Terminates the evolutionary run (if yet in progress), saves parameters
        and data to disk, and cleanly returns the user to karoo_gp.py and the
        command line.

        TODO: Replace with a save() method, possibly call automatically when
        used via ContextManager, or manually.
        '''
        self.fx_data_params_write(self.kernel)
        self.fx_data_params_write_json(self.kernel)

        # save the final population
        loc = self.save_population('f')

        self.log(f'Your Trees and runtime parameters are archived in {loc}')
        self.log('\n\033[3m "It is not the strongest of the species that '
                 'survive, nor the most intelligent,\033[0;0m\n'
                 '\033[3m  but the one most responsive to change."'
                 '\033[0;0m --Charles Darwin\n\n'
                 '\033[3m Congrats!\033[0;0m Your Karoo GP run is complete.\n')
        sys.exit()


    #+++++++++++++++++++++++++++++++++++++++++++++
    #   Methods to Load and Archive Data         |
    #+++++++++++++++++++++++++++++++++++++++++++++

    # tested 2017 02/13; argument 'app' removed to simplify termination 2019 06/08
    def fx_data_params_write(self, kernel):

        '''
        Save run-time configuration parameters to disk.

        Called by: fx_karoo_terminate

        Arguments required: app
        '''
        with open(self.path / 'log_config.txt', 'w') as file:
            file.write('Karoo GP')
            file.write('\n launched: ' + str(self.datetime))
            file.write('\n')
            file.write('\n kernel: ' + str(kernel))
            file.write('\n precision: ' + str(self.precision))
            file.write('\n')
            # file.write('tree type: ' + tree_type)
            # file.write('tree depth base: ' + str(tree_depth_base))
            file.write('\n tree depth max: ' + str(self.tree_depth_max))
            file.write('\n min node count: ' + str(self.tree_depth_min))
            file.write('\n')
            file.write('\n genetic operator Reproduction: ' + str(self.evolve_repro))
            file.write('\n genetic operator Point Mutation: ' + str(self.evolve_point))
            file.write('\n genetic operator Node Mutation: ' + str(self.evolve_branch))
            file.write('\n genetic operator Crossover: ' + str(self.evolve_cross))
            file.write('\n')
            file.write('\n tournament size: ' + str(self.tourn_size))
            file.write('\n population: ' + str(self.tree_pop_max))
            file.write('\n number of generations: ' + str(self.population.gen_id))
            file.write('\n\n')

        with open(self.path / 'log_test.txt', 'w') as file:
            file.write('Karoo GP')
            file.write('\n launched: ' + str(self.datetime))
            file.write('\n')

            # Which population the fittest_dict indexes refer to
            if len(self.population.fittest_dict) == 0:
                file.write('\n\n There were no evolved solutions generated in '
                        'this run... your species has gone extinct!')

            else:
                fittest = self.population.fittest()
                file.write(f'\n\n Tree {fittest.id} is the most fit, with '
                        f'expression:\n\n {fittest.expression}')
                result = self.score(self.X_test, self.y_test)
                if kernel == 'c':
                    file.write(f'\n\n Classification fitness score: {result["fitness"]}')
                    file.write(f'\n\n Precision-Recall report:\n {result["classification_report"]}')
                    file.write(f'\n Confusion matrix:\n {result["confusion_matrix"]}')

                elif kernel == 'r':
                    file.write(f'\n\n Regression fitness score: {result["fitness"]}')
                    file.write(f'\n Mean Squared Error: {result["mean_squared_error"]}')

                elif kernel == 'm':
                    file.write(f'\n\n Matching fitness score: {result["fitness"]}')

            file.write('\n\n')

        return

    def fx_data_params_write_json(self, kernel):
        '''
        Save run-time configuration parameters to disk as json.

        Called by: fx_karoo_terminate
        '''
        generic = dict(
            package='Karoo GP',
            launched=self.datetime,
            # dataset=str(self.filename),  # Should be external to model
        )
        config = dict(
            kernel=kernel,
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
            tree = self.population.fittest()
            fittest_tree = dict(
                id=tree.id,
                expression=tree.expression,
            )
            final_dict = dict(
                **final_dict,
                outcome='SUCCESS',
                fittest_tree=fittest_tree,
                score=self.score(self.X_test, self.y_test),
            )
        with open(self.path / 'results.json', 'w') as f:
            json.dump(final_dict, f, indent=4)


# _____________________________________________________________________________
# KERNEL IMPLEMENTATIONS

class RegressorGP(BaseGP):
    """
    A trainable Regressor built on Karoo's BaseGP.

    'Absolute error' is the fitness function: the sum of the absolute
    difference between the prediction and actual for every sample. 'Mean
    squared error' is also calculated and included in score/history.

    Both functions round outputs to 6 significant figures.
    """
    kernel = 'r'  # Regressor

    def __init__(self, *args, **kwargs):
        """Add kernel-specific scoring function(s) and kwargs"""

        def absolute_error(y_true, y_pred):
            """Karoo's default regression fitness, rounded"""
            output = float(np.sum(np.abs(y_true - y_pred)))
            return self.sigfig_round(output)

        def sigfig_mse(y_true, y_pred):
            """SKLearn's MSE, but non-numpy and rounded"""
            output = float(skm.mean_squared_error(y_true, y_pred))
            return self.sigfig_round(output)

        kwargs['scoring'] = dict(fitness=absolute_error,
                                 mean_squared_error=sigfig_mse)
        kwargs['precision'] = 6
        super().__init__(*args, **kwargs)

    def sigfig_round(self, x, sigfigs=6):
        """Round to a given number of significant figures

        The model works with floats. Floats are stored as 'the most accurate
        fraction which generates its value', and the 'precision' is however
        close a fraction of some size (32 or 64 bit) can be. When converted to
        a Python float, np floats include more digits than are necessarily
        accurate in an arithmetic sense - though they are accurate to the
        definition of a float.

        We aim to follow the arithmetic sense, such that digits which aren't
        arithmetically significant are replaced with zeros (if left of the
        decimal) or removed.
        """
        return round(x, sigfigs - int(np.floor(np.log10(abs(x)))) - 1)


class MatchingGP(BaseGP):
    """
    A Matching algorithm built on Karoo's BaseGP.

    'Absolute matches' is the fitness function: the number of samples which
    were predicted correctly.

    TODO: remove?"""
    kernel = 'm'  # Match

    def __init__(self, *args, **kwargs):
        """Add kernel-specific scoring function(s) and kwargs"""

        def absolute_matches(y_true, y_pred):
            """The number of nearly* exact numeric matches"""
            RTOL, ATOL = 1e-05, 1e-08
            matches = np.less_equal(np.abs(y_true - y_pred),
                                    ATOL + RTOL * np.abs(y_true) # * 'nearly'
                                    ).astype(np.int32)
            return float(sum(matches))

        kwargs['scoring'] = dict(fitness=absolute_matches)
        kwargs['higher_is_better'] = True
        super().__init__(*args, **kwargs)


    def fit(self, X, y, *args, **kwargs):
        """Determine what would consitute a perfect score for the dataset"""
        self.perfect_score = len(y)
        super().fit(X, y, *args, **kwargs)

    def build_fittest_dict(self, trees):
        """After first generation, only include perfect scores"""
        fittest_dict = {}
        if self.population.gen_id == 1:
            last_fittest = None
            def _fitter(t):
                return self.fitness_compare(last_fittest, t).id == t.id
            for tree in trees:
                if last_fittest is None or _fitter(tree):
                    fittest_dict[tree.id] = tree.expression
                    last_fittest = tree
        else:
            for tree in trees:
                if tree.fitness == self.perfect_score:
                    fittest_dict[tree.id] = tree.expression
        return fittest_dict


class MultiClassifierGP(BaseGP):
    """
    A trainable Multiclass Classifier built on Karoo's BaseGP.

    'Number Correct' is the fitness function: the number of samples
    classified correctly. Also calculated for score/history are sklearn's
    'Classification Report' and 'Confusion Matrix'.

    This classes passes a ClassDecoder to BaseGP as the prediction_transformer.
    The decoder is fit on y-data for the number of unique values (classes).
    Then, numeric predictions from BaseGP are clipped/rounded to integers which
    match those classes.
    """

    kernel = 'c'  # Classify

    def __init__(self, *args, **kwargs):
        """Add kernel-specific scoring functions, setup decoder"""

        def n_correct(y_true, y_pred):
            """Default classification fitness: number of correct predictions"""
            return float(np.sum(y_true == y_pred))

        def cls_report_zero_div(y_true, y_pred):
            """Call sklearn.metrics function with default kwargs

            TODO: cli test expected output (from '..write_json()')
            classification report dict keys are floats. Should use int.
            """
            return skm.classification_report(
                y_true.astype(np.float32),
                y_pred.astype(np.float32),
                zero_division=0, output_dict=True)

        def conf_matrix_as_list(y_true, y_pred):
            """Call sklearn.metrics function but return as list"""
            return skm.confusion_matrix(y_true, y_pred).tolist()

        kwargs['scoring'] = dict(fitness=n_correct,
                                 classification_report=cls_report_zero_div,
                                 confusion_matrix=conf_matrix_as_list)
        kwargs['higher_is_better'] = True
        kwargs['prediction_transformer'] = ClassDecoder(n_classes=kwargs.pop('n_classes', None))
        super().__init__(*args, **kwargs)

    def fit(self, X, y, *args, **kwargs):
        """Initialize decoder to be used by scorer"""
        self.prediction_transformer.fit(y)
        super().fit(X, y, *args, **kwargs)


class ClassDecoder(TransformerMixin):
    """
    Transforms tree predictions (-2.3, -1, 3.3) into class labels (0, 0, 3).

    For a normalized input values (zero-centered, stdev=1), the model should
    have the greatest precision around zero. Class labels are expected to be
    integers from 0-N. For this reason, we skew the output range so that the
    model has the greatest precision around labels range.

    1. Fit the decoder to training labels (y). Determine the number of
       classes (i.e. unique labels) and calculate skew. Skew is the
       transformation needed to convert 0-N integers (0, 1, 2, 3) to
       zero-centered floats (-1.5, -0.5, 0.5, 1.5); in this case -1.5
       TODO: should be (-1, -0.33, 0.33, 1)
    2. Transform predictions:
        a. Engine output: (-2.3, -1, 3.3)
        b. Skew removed: (-0.8, 0.5, 4.8)
        c. Round*: (-1, 0, 5)
        d. Clip between 0 and n: (0, 0, 3)

    * 0.5 always rounds down: 1.5 -> 1, 2.5 -> 2, -0.5 -> -1
    """
    def __init__(self, n_classes=None):
        if n_classes is not None:
            self.n_classes = n_classes
            self._set_skew()
        else:
            self.skew = None

    def _set_skew(self):
        self.skew = (self.n_classes / 2) - 0.5

    def fit(self, y):
        self.n_classes = len(np.unique(y))
        self._set_skew()

    def transform(self, x):
        """Remove skew, round and clip 0-N"""
        # Recursively transform batches
        if len(x.shape) > 1:
            return np.array([self.transform(x_i) for x_i in x])
        # Transform single sample
        else:
            unskewed = x + self.skew
            rounded = np.where(unskewed % 1 <= 0.5,
                np.floor(unskewed), np.ceil(unskewed))
            clipped = np.minimum(np.maximum(rounded, 0), self.n_classes-1)
            return clipped.astype(np.int32)

    def inverse_transform(self, y):
        """Add skew"""
        return y - self.skew
