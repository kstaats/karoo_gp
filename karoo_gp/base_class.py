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
import json
import pathlib
import operator

import numpy as np
import sklearn.metrics as skm
import sklearn.model_selection as skcv

from sklearn.utils import check_random_state, check_X_y
from sklearn.base import BaseEstimator, TransformerMixin

from datetime import datetime

from . import Functions, Terminals, NumpyEngine, TensorflowEngine, \
              Population, Tree

# TODO: This is used to save the final population and the generated self.path
# is used by fx_data_params_write/_json. I think this should be moved back to
# BaseGP. Some of the save/load methods there are still unused/nonfunctioning,
# and they can be combined with the useful parts of this.
class DataLoader:
    def __init__(self, model):
        """Determine path and initialize log dir"""
        runs_dir = os.path.join(os.getcwd(), 'runs')
        if model.output_dir:
            self.path = os.path.join(runs_dir, model.output_dir + '/')
        else:
            basename = os.path.basename(model.filename)  # extract the filename (if any)
            root, _ = os.path.splitext(basename)  # split root from extension
            # TODO: datetime should be set when run is initialized, i.e. in
            # model.fit. However, sometimes the model terminates early, and
            # needs to have made an output dir already. Update this with a
            # comprehensive new approach to saving/logging data.
            self.path = os.path.join(runs_dir, f'{root}_{model.datetime}/')
        if not os.path.isdir(self.path):    # initialize elog dir
            os.makedirs(self.path)
        for fname in ['a', 'b', 'f', 's']:  # initialize log files
            self.save('', fname)

    def save(self, data, fname):
        """Save data to csv"""
        if fname in ['a', 'b', 'f', 's']:
            fname = f'population_{fname}.csv'
        with open(self.path + fname, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(data)

class BaseGP(BaseEstimator):

    """
    This Base_BP class al the core attributes, objects and methods of Karoo GP.
    It's composed of a heirarchal structure of classes. Below are some of the
    more import attributes and methods for each class, and their organization:

    BaseGP
    ├─ .scoring = {field: func...}      - funcs are passed (y_true, y_pred)
    ├─ .decoder(y)                      - Initialize with y_train
    │   └─ .transform(pred)             - transforms prediction to match y
    │
    ├─ .fitness_compare(a, b)           - determines the fitter of two trees
    │
    ├─ .terminals = Terminals           - list of terminals + constants
    │   └─ .get(instx)                  - returns list of terminals from instx
    │
    ├─ .functions = Functions           - operators and associated type/arity
    │   └─ .get(instx)                  - returns list of functions from instx
    │
    ├─ .engine = Engine                 - Numpy for cpu, Tensorflow for gpu
    │   └─ .predict(trees, X, X_hash)   - returns X predictions for each tree
    │
    ├─ .population = Population         - an isolated group of trees
    │   ├─ .fittest = Tree              - return the fittest tree
    │   ├─ .trees                       - list of Trees, length tree_pop_max
    │   │   ├─ ...
    │   │   └─ Tree                     - an evolvable expression tree
    │   │      ├─ .expression           - a string of sympified expr, e.g. a*b
    │   │      ├─ .score                - a dict of results matching `scoring`
    │   │      ├─ .root = Branch        - a recursive node which forms a Tree
    │   │      │   ├─ .symbol           - a terminal ('a') or function ('*')
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


    # Fit parameters (set later)
    engine = None
    scoring_ = None
    history_ = None
    X_hash_ = None
    test_split_ = None
    terminals_ = None
    functions_ = None
    population = None
    cache_ = None

    def __init__(
        self, tree_type='r', tree_depth_base=3, tree_depth_max=None,
        tree_depth_min=1, tree_pop_max=100, gen_max=10, tourn_size=7,
        filename='', output_dir='', evolve_repro=0.1, evolve_point=0.1,
        evolve_branch=0.2, evolve_cross=0.6, display='s', precision=None,
        swim='p', mode='s', random_state=None, pause_callback=None,
        engine_type='numpy', tf_device="/gpu:0", tf_device_log=False,
        functions=None, terminals=None, constants=None, test_size=0.2,
        scoring=None, higher_is_better=False, prediction_transformer=None,
        cache=None):
        """Initialize a Karoo_GP object with given parameters"""

        # Model parameters
        self.tree_type = tree_type           # (f)ull, (g)row or (r)amped 50/50
        self.tree_depth_base = tree_depth_base # depth of initial population
        self.tree_depth_max = tree_depth_max # max allowed depth
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
        self.random_state = random_state
        self.pause_callback = pause_callback # called throughout based on disp
        self.engine_type = engine_type
        self.tf_device = tf_device
        self.tf_device_log = tf_device_log
        self.functions = functions
        self.terminals = terminals
        self.constants = constants
        self.test_size = test_size           # how to portion train/test data
        self.scoring = scoring
        self.higher_is_better = higher_is_better
        self.prediction_transformer = prediction_transformer
        self.cache = cache

    def log(self, msg, display={'i', 'g', 'm', 'db'}):
        if self.display in display or display == 'all':
            print(msg)

    def pause(self, display={'i', 'g', 'm', 'db'}):
        if not self.pause_callback:
            self.log('No pause callback function provided')
            return 0
        elif self.display in display:
            self.pause_callback(self)
        else:
            return 0

    def error(self, msg, display={'i', 'g', 'm', 'db'}):
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


    #+++++++++++++++++++++++++++++++++++++++++++++
    #   Methods to Run Karoo GP                  |
    #+++++++++++++++++++++++++++++++++++++++++++++

    def fitness_compare(self, a: Tree, b: Tree) -> Tree:
        """Return the fitter of two Trees, or the latter if equal

        TODO: Would make sense for trees to be directly comparable (a > b)
        but they don't hold a reference to BaseGP, and the user should be
        able to override this (default) compare_fitness function.
        """
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
            if self.engine_type == 'numpy':
                self.engine = NumpyEngine(self)
            elif self.engine_type == 'tensorflow':
                self.engine = TensorflowEngine(self, self.tf_device, self.tf_device_log)
            else:
                raise ValueError(f'Unrecognized engine_type: {self.engine_type}')
            # Loader
            self.datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
            self.loader = DataLoader(self)       # set path, initialize logs

    def check_population(self, X, y):
        """Initialize and/or validate population parameters"""
        # Terminals
        if self.terminals_ is None:
            if self.terminals is None:
                terms = [f'f{i}' for i in range(X.shape[1])]
            elif (isinstance(self.terminals, list) and
                  all(isinstance(t, str) for t in self.terminals)):
                terms = self.terminals
            else:
                raise ValueError('terminals must be a list of strings')
            if self.constants is not None and (
                not isinstance(self.constants, list) or
                not all(type(c) in [int, float] for c in self.constants)):
                raise ValueError('constants must be a list of ints or floats')
            self.terminals_ = Terminals(terms, self.constants)
        if len(self.terminals_.variables) != X.shape[1]:
            raise ValueError('terminals and features must be same length')

        # Functions
        if self.functions_ is None:
            # Validate functions
            if isinstance(self.functions, Functions):
                self.functions_ = self.functions
            elif isinstance(self.functions, list):
                # TODO validate symbol against supported
                self.functions_ = Functions(self.functions)
            # TODO support function type as args, e.g. 'logic, arithmetic'
            elif self.functions is None:
                self.functions_ = Functions.arithmetic()
            else:
                raise ValueError('functions must be a list of strings or type '
                                 'Functions')

        # Population
        if self.population is None:
            self.population = Population.generate(
                model=self,
                functions=self.functions_,
                terminals=self.terminals_,
                tree_type=self.tree_type,
                tree_depth_base=self.tree_depth_base,
                tree_pop_max=self.tree_pop_max,
            )
            self.log(f'\n We have constructed the first, stochastic population of '
                     f'{self.tree_pop_max} Trees.')
            self.population.evaluate(X, y, self.X_train_hash)
            self.log_history()

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
        """Split train/test data; reuse subsequently for same X, y"""
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

    def fit(self, dataset=None, dataset_without_solution=None):
        """Evolve a population of trees based on training data"""

        self.log('Press ? to see options, or ENTER to continue the run',
                 display=['i'])
        self.pause(display=['i'])

        # Initialize model and all variables
        self.check_model()
        dataset, dataset_without_solution = check_X_y(dataset, dataset_without_solution)
        X_train, X_test, y_train, y_test = self.check_test_split(dataset, dataset_without_solution)
        self.check_population(X_train, y_train)

        menu = 1
        while menu != 0:  # Supports adding generations mid-run
            for _ in range(self.population.gen_id, self.gen_max):

                # Evolve the next generation
                self.population = self.population.evolve(
                    self.tree_pop_max,
                    self.functions_,
                    self.terminals_,
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

            if self.mode == 's':  # (s)erver mode: terminate after run
                menu = 0
            else:                 # (d)esktop mode: pause after run
                self.log('Enter ? to review your options or q to quit')
                menu = self.pause()

    def predict(self, X):
        """Return predicted y values for X using the fittest tree

        * Pass the fittest tree to batch_predict as a list (of 1)
        * Return the first result
        """
        if not self.population.evaluated:
            tree = self.population.trees[0]
        else:
            tree = self.population.fittest()
        return self.batch_predict(X, [tree])[0]

    def batch_predict(self, X, trees, X_hash=None):
        """Return predicted values for y given X for a list of trees

        * If prediction_transformer (e.g. MultiClassifier), transform predictions
        * If precision is specified, round predictions to precision
        """
        y = self.engine.predict(trees, X, X_hash)
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

    def fx_karoo_terminate(self):
        '''
        Terminates the evolutionary run (if yet in progress), saves parameters
        and data to disk, and cleanly returns the user to karoo_gp.py and the
        command line.

        TODO: Replace with a save() method, possibly call automatically when
        used via ContextManager, or manually.
        '''
        kernel = {
            RegressorGP: 'r', MultiClassifierGP: 'c', MatchingGP: 'm', BaseGP: 'p'
        }[type(self)]
        self.fx_data_params_write(kernel)
        self.fx_data_params_write_json(kernel)

        # save the final population
        self.loader.save([t.save() for t in self.population.trees], 'f')

        self.log(f'Your Trees and runtime parameters are archived in '
                 f'{self.loader.path}/population_f.csv')
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
        with open(self.loader.path + 'log_config.txt', 'w') as file:
            file.write('Karoo GP')
            file.write('\n launched: ' + str(self.datetime))
            file.write('\n dataset: ' + str(self.filename))
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
            file.write('\n genetic operator Branch Mutation: ' + str(self.evolve_branch))
            file.write('\n genetic operator Crossover: ' + str(self.evolve_cross))
            file.write('\n')
            file.write('\n tournament size: ' + str(self.tourn_size))
            file.write('\n population: ' + str(self.tree_pop_max))
            file.write('\n number of generations: ' + str(self.population.gen_id))
            file.write('\n\n')

        with open(self.loader.path + 'log_test.txt', 'w') as file:
            file.write('Karoo GP')
            file.write('\n launched: ' + str(self.datetime))
            file.write('\n dataset: ' + str(self.filename))
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
            dataset=str(self.filename),
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
        with open(self.loader.path + 'results.json', 'w') as f:
            json.dump(final_dict, f, indent=4)


# _____________________________________________________________________________
# KERNEL IMPLEMENTATIONS

class RegressorGP(BaseGP):
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
    def __init__(self, *args, **kwargs):
        """Add kernel-specific scoring functions, setup decoder"""

        def n_correct(y_true, y_pred):
            """Default classification fitness: number of correct predictions"""
            return float(np.sum(y_true == y_pred))

        def cls_report_zero_div(y_true, y_pred):
            # TODO: cli test expected output (from '..write_json()')
            # classification report dict keys are floats. Should use int.
            return skm.classification_report(
                y_true.astype(np.float32),
                y_pred.astype(np.float32),
                zero_division=0, output_dict=True)

        def conf_matrix_as_list(y_true, y_pred):
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
