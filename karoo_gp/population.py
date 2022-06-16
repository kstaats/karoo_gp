import ast, math
from functools import reduce
import operator as op

import numpy as np
import sklearn.metrics as skm
from sympy import sympify
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  # from https://www.tensorflow.org/guide/migrate

from . import Tree

operators = {
    ast.Add: tf.add,  # e.g., a + b
    ast.Sub: tf.subtract,  # e.g., a - b
    ast.Mult: tf.multiply,  # e.g., a * b
    ast.Div: tf.divide,  # e.g., a / b
    ast.Pow: tf.pow,  # e.g., a ** 2
    ast.USub: tf.negative,  # e.g., -a
    ast.And: tf.logical_and,  # e.g., a and b
    ast.Or: tf.logical_or,  # e.g., a or b
    ast.Not: tf.logical_not,  # e.g., not a
    ast.Eq: tf.equal,  # e.g., a == b
    ast.NotEq: tf.not_equal,  # e.g., a != b
    ast.Lt: tf.less,  # e.g., a < b
    ast.LtE: tf.less_equal,  # e.g., a <= b
    ast.Gt: tf.greater,  # e.g., a > b
    ast.GtE: tf.greater_equal,  # e.g., a >= 1
    'abs': tf.abs,  # e.g., abs(a)
    'sign': tf.sign,  # e.g., sign(a)
    'square': tf.square,  # e.g., square(a)
    'sqrt': tf.sqrt,  # e.g., sqrt(a)
    'pow': tf.pow,  # e.g., pow(a, b)
    'log': tf.log,  # e.g., log(a)
    'log1p': tf.log1p,  # e.g., log1p(a)
    'cos': tf.cos,  # e.g., cos(a)
    'sin': tf.sin,  # e.g., sin(a)
    'tan': tf.tan,  # e.g., tan(a)
    'acos': tf.acos,  # e.g., acos(a)
    'asin': tf.asin,  # e.g., asin(a)
    'atan': tf.atan,  # e.g., atan(a)
    }

# ----------------------------
# MAY REDESIGN
#
# The Population and Tree objects implement basic api methods, initially as
# wrappers for 'fx_..' functions.
#
# For now, they hold very little in state and take an obscene number of args.
# This is temporary, while we figure out what should be stored where.

class Population:

    def __init__(self, trees, gen_id=1, fitness_type=None, fittest_dict=None,
                 history=None):
        '''TODO'''
        if fittest_dict is None:
            fittest_dict = {}
        if history is None:
            history = []
        self.trees = trees                # The list of Trees
        self.gen_id = gen_id              # population_b if len > 0, else trees
        self.fitness_type = fitness_type  # Kernel uses 'max' or 'min' fitness
        self.fittest_dict = fittest_dict  # TODO: Fix or remove
        self.history = history            # The fittest tree from each gen
        self.population_b = []            # Evolved from self.trees
        self.gene_pool = None

    @classmethod
    def generate(cls, log, pause, error, gen_id=1, tree_type='r',
                 tree_depth_base=3, tree_depth_max=3, tree_pop_max=100,
                 functions=None, terminals=None, rng=None, fitness_type='max'):
        """Return a new Population of a type/amount trees"""
        trees = []
        if tree_type == 'r':
            # (r)amped 50/50:  Create 1 full- and 1 grow-tree with each level of
            # depth, from 2 to the max depth.

            # Do as many full cycles of (n=max-2) depth as possible..
            n_cycles = int(tree_pop_max/2/tree_depth_base)
            for i in range(n_cycles):
                for d in range(tree_depth_base):
                    for _type in ['f', 'g']:
                        trees.append(Tree.generate(log, pause, error,
                                                   len(trees)+1, _type, d+1,
                                                   functions, terminals, rng))

            # ..and add ramped trees for the remainder.
            extras = tree_pop_max - len(trees)
            for i in range(extras):
                trees.append(Tree.generate(log, pause, error, len(trees)+1,
                                           'g', tree_depth_base,
                                           functions, terminals, rng))
        else:
            # (f)ull: Fill-in all nodes to the maximum depth
            # (g)row: Add nodes or terminals at random up to max depth
            for i in range(tree_pop_max):
                trees.append(Tree.generate(log, pause, error, i+1, tree_type,
                                           tree_depth_base,
                                           functions, terminals, rng))
        return cls(trees, gen_id, fitness_type)

    def fitness_compare(self, a, b, precision=6):
        """Return b if fitness of b is equal or better than a"""
        _fit = lambda t: round(float(t.fitness), precision)
        compare_func = dict(
            max=lambda a, b: a if _fit(a) > _fit(b) else b,
            min=lambda a, b: a if _fit(a) < _fit(b) else b,
        )[self.fitness_type]
        return compare_func(a, b)

    def fittest(self):
        '''Return the fittest tree of the population.

        TODO: cache'''
        return reduce(self.fitness_compare, self.trees)

    def evaluate(self, log=None, pause=None, error=None, data_train=[],
                 kernel='m', data_train_rows=0, tf_device_log=None,
                 class_labels=0, tf_device=None, terminals=[], precision=6,
                 savefile={}, fx_data_tree_write=None):
        '''Test all trees against the training data, log results'''
        # TODO: This is a workaround, explaned more in 'fx_.._maker'
        fx_fitness_labels_map = fx_fitness_labels_map_maker(class_labels)

        # Evaluate trees, log the results to each tree, and return them.
        new_trees = fx_eval_generation(
            self.trees, data_train, kernel, data_train_rows, tf_device_log,
            class_labels, tf_device, terminals, precision, savefile,
            self.gen_id, log, pause, error, self.evaluate_tree,
            fx_data_tree_write, fx_fitness_labels_map
        )

        # Replace current trees with evaluated trees
        self.trees = new_trees

        # MAY REDESIGN: relocated from `fx_fitness_gym`
        # Build the fittness dict
        self.fittest_dict = {}
        max_fitness = 0
        for tree in self.trees:
            fitness_test = dict(max=op.gt, min=op.lt)
            tree_fitness = round(float(tree.fitness), precision)
            if kernel in ['c', 'r']:
                fitter = (fitness_test[self.fitness_type](tree_fitness, max_fitness) or
                          op.eq(tree_fitness, max_fitness) or
                          max_fitness == 0)
            elif kernel == 'm':
                fitter = op.eq(float(tree_fitness), data_train_rows)
            if fitter:
                max_fitness = tree_fitness
                self.fittest_dict[tree.id] = tree.expression
        log(f'\n\033[36m {len(self.fittest_dict)} '
            f'trees\033[1m {np.sort(list(self.fittest_dict.keys()))} '
            f'\033[0;0m\033[36moffer the highest fitness scores.\033[0;0m')
        pause(display=['g'])


        # Add the fittest of this generation to history
        if len(self.history) < self.gen_id:
            self.history.append(self.fittest())

    def evaluate_tree(self, tree, data_train, tf_device_log, kernel,
                      class_labels, tf_device, terminals, precision,
                      log, fx_fitness_labels_map):
        '''Test a single tree against training data, log results'''
        expr = tree.expression
        log(f'\t\033[36mTree {tree.id} yields (sym):\033[1m {tree.expression}'
            f' \033[0;0m')
        result = fx_fitness_eval(expr, data_train, tf_device_log, kernel,
                                 class_labels, tf_device, terminals,
                                 fx_fitness_labels_map, get_pred_labels=True)
        log(f'\t \033[36m with fitness sum:\033[1m {result["fitness"]}'
            f'\033[0;0m\n', display=['i'])
        tree = fx_fitness_store(tree, result, kernel, precision)
        return tree

    def evolve(self, swim, tree_depth_min, functions, terminals, evolve_repro,
               evolve_point, evolve_branch, evolve_cross, tree_pop_max,
               tourn_size, precision, fitness_type, tree_depth_max, data_train,
               kernel, data_train_rows, tf_device_log, class_labels, tf_device,
               savefile, rng, log, pause, error, fx_data_tree_write):
        '''Return a new population evolved from self'''

        # Calculte number of new trees per evolution type
        evolve_ratios = dict(repro=evolve_repro, point=evolve_point,
                             branch=evolve_branch, cross=evolve_cross)
        if sum(evolve_ratios.values()) != 1.0:
            raise ValueError(f'Evolution parameters must sum to 1')
        # TODO: int(3.5) rounds down to 3. For test 'test_cli[m-g-3-1000]',
        # with `tree_pop_max=35`, this results in a population of 33 trees
        # instead of 35. This should be fixed but will change test result.
        evolve_amounts = {k: int(v * tree_pop_max)
                          for k, v in evolve_ratios.items()}

        # Create the list of eligible trees
        log('\n Prepare a viable gene pool ...', display=['i'])
        pause(display=['i'])
        self.fitness_gene_pool(log, swim, tree_depth_min, terminals)
        # Initialize new population and begin evolving new trees
        self.population_b = []
        for evolve_type, amount in evolve_amounts.items():
            verb = dict(repro='Reproductions', point='Point Mutations',
                        branch='Branch Mutations', cross='Crossovers')
            log(f'  Perform {amount} {verb[evolve_type]} ...')
            pause(display=['i'])
            amount = amount // 2 if evolve_type == 'cross' else amount
            for _ in range(amount):
                # Create offspring from first parent
                parent = self.tournament(rng, tourn_size, log)
                log(f'\n\t\033[36mThe winner of the tournament is '
                    f'Tree:\033[1m{parent.id} \033[0;0m', display=['i'])

                offspring = parent.copy(id=len(self.population_b) + 1)
                # Reproduce: add to new population as-is
                if evolve_type == 'repro':
                    self.population_b.append(offspring)
                # Point Mutate: replace a single node
                elif evolve_type == 'point':
                    offspring.point_mutate(rng, functions, terminals, log)
                    self.population_b.append(offspring)
                # Branch Mutate: replace a random subtree
                elif evolve_type == 'branch':
                    offspring.branch_mutate(rng, functions, terminals,
                                            tree_depth_max, log)
                    self.population_b.append(offspring)
                # Crossover: create 2 unique offspring by splicing 2 parents
                elif evolve_type == 'cross':
                    # Select parent a, clone, select random i_a, repeat with b
                    # TODO: This could move to Tree.crossover(), but it changes
                    # the order of calls to rng, so would need to update tests.
                    parent_a = parent  # Renamed for clarity
                    offspring_a = offspring
                    offspring_a.id += 1
                    i_mutate_a = rng.integers(1, parent_a.n_children + 1)
                    parent_b = self.tournament(rng, tourn_size, log)
                    offspring_b = parent_b.copy(id=len(self.population_b) + 1)
                    i_mutate_b = rng.integers(1, parent_b.n_children + 1)

                    for from_id, to_id, to_i in [
                        (parent_a.id, parent_b.id, i_mutate_b),
                        (parent_b.id, parent_a.id, i_mutate_a)]:
                        log(f'\t\033[36m crossover from \033[1mparent '
                            f'{from_id} \033[0;0m\033[36mto \033[1moffspring '
                            f'{to_id} \033[0;0m\033[36mat node\033[1m '
                            f'{to_i} \033[0;0m', display=['i'])

                    # Replace b's branch i_b with a's branch i_a & vice versa
                    offspring_b.crossover(i_mutate_b, parent_a, i_mutate_a,
                                          rng, terminals, tree_depth_max,
                                          log, pause)
                    self.population_b.append(offspring_b)
                    offspring_a.crossover(i_mutate_a, parent_b, i_mutate_b,
                                          rng, terminals, tree_depth_max,
                                          log, pause)
                    self.population_b.append(offspring_a)

        # Return a Population with new trees with fitness calculated
        new_population = Population(
            trees=self.population_b, gen_id=self.gen_id + 1,
            fitness_type=self.fitness_type, history=self.history)
        new_population.evaluate(
            log, pause, error, data_train, kernel, data_train_rows, tf_device_log,
            class_labels, tf_device, terminals, precision, savefile, fx_data_tree_write
        )
        return new_population

    #++++++++++++++++++++++++++++
    #   Evolution               |
    #++++++++++++++++++++++++++++

    def fitness_gene_pool(self, log, swim='p', tree_depth_min=None, terminals=None):
        self.gene_pool = []
        for tree in self.trees:
            if swim == 'p':
                # each tree must have the min number of nodes defined by user
                if tree.n_children + 1 >= tree_depth_min and tree.expression != '1':
                    log(f'\t\033[36m Tree {tree.id} has >= '
                             f'{tree_depth_min} nodes and is added to the gene'
                             f'pool\033[0;0m', display=['i'])
                    self.gene_pool.append(tree.id)
            elif swim == 'f':
                # each tree must contain at least one instance of each feature
                saved = tree.save()
                missing = sum([1 for t in terminals.get()
                               if f'({t.symbol})' not in saved])
                if not missing:
                    log(f'\t\033[36m Tree {tree.id} includes at least one'
                        f' of each feature and is added to the gene '
                        f'pool\033[0;0m', display=['i'])
                    self.gene_pool.append(tree.id)
        log(f'\n\t The total population of the gene pool is '
            f'{len(self.gene_pool)}', display=['i'])

    def tournament(self, rng, tournament_size=7, log=None):
        log('\n\tEnter the tournament ...', display=['i'])
        if not self.gene_pool:
            raise ValueError('Cannot conduct tournament: gene pool is empty')
        t_ids = [rng.choice(self.gene_pool) for _ in range(tournament_size)]
        trees = [self.trees[tree_id-1] for tree_id in t_ids]
        for t in trees:
            log(f'\t\033[36m Tree {t.id} has fitness {t.fitness}\033[0;0m',
                display=['i'])
        return reduce(self.fitness_compare, trees)

# __________________________________________________________________
# OLD METHODS

# used by: Population, Tree
def fx_data_tree_clean(tree):

    '''
    This method aesthetically cleans the Tree array, removing redundant data.

    Called by: fx_data_tree_append, fx_evolve_branch_copy

    Arguments required: tree
    '''

    tree.root[0][2:] = ''  # A little clean-up to make things look pretty :)
    tree.root[1][2:] = ''  # Ignore the man behind the curtain!
    tree.root[2][2:] = ''  # Yes, I am a bit OCD ... but you *know* you appreciate clean arrays.

    return tree

# used by: Population
def fx_eval_id(tree, node_id):

    '''
    Evaluate all or part of a Tree and return a list of all 'NODE_ID's.

    This method generates a list of all 'NODE_ID's from the given Node
    and below. It is used primarily to generate 'branch' for
    the multi-generational mutation of Trees.

    Pass the starting node for recursion via the local variable
    'node_id' where the local variable 'tree' is a copy of the Tree
    you desire to evaluate.

    Called by: fx_eval_id (recursively), fx_evolve_branch_select

    Arguments required: tree, node_id
    '''

    node_id = int(node_id)

    if tree.root[8, node_id] == '0':  # arity of 0 for the pattern '[NODE_ID]'
        return tree.root[3, node_id]  # 'NODE_ID'

    else:
        # arity of 1 for the pattern '[NODE_ID], [NODE_ID]'
        if tree.root[8, node_id] == '1':
            return (tree.root[3, node_id] + ', ' +
                    fx_eval_id(tree, tree.root[9, node_id]))

        # arity of 2 for the pattern '[NODE_ID], [NODE_ID], [NODE_ID]'
        elif tree.root[8, node_id] == '2':
            return (tree.root[3, node_id] + ', ' +
                    fx_eval_id(tree, tree.root[9, node_id]) + ', ' +
                    fx_eval_id(tree, tree.root[10, node_id]))

        # arity of 3 for the pattern '[NODE_ID], [NODE_ID], [NODE_ID], [NODE_ID]'
        elif tree.root[8, node_id] == '3':
            return (tree.root[3, node_id] + ', ' +
                    fx_eval_id(tree, tree.root[9, node_id]) + ', ' +
                    fx_eval_id(tree, tree.root[10, node_id]) + ', ' +
                    fx_eval_id(tree, tree.root[11, node_id]))

# used by: Population
def fx_eval_generation(trees, train_data, kernel, data_train_rows,
                       tf_device_log, class_labels, tf_device, terminals,
                       precision, savefile, gen_id, log, pause, error,
                       evaluate_tree, fx_data_tree_write,
                       fx_fitness_labels_map):

    '''
    This method invokes the evaluation of an entire generation of Trees.
    It automatically evaluates population_b before invoking
    the copy of _b to _a.

    Called by: fx_karoo_gp

    Arguments required: none
    '''

    log(f'\n Evaluate all Trees in Generation {gen_id}')
    pause(display=['i'])

    # renumber all Trees in given population - merged fx_evolve_tree_renum 2018 04/12
    for i, tree in enumerate(trees):
        tree.id = i+1

    # run fx_eval(), fx_fitness(), fx_fitness_store(), and fitness record
    trees = fx_fitness_gym(trees, train_data, kernel,
                           data_train_rows, tf_device_log,
                           class_labels, tf_device,
                           terminals, precision, log, pause,
                           error, evaluate_tree,
                           fx_fitness_labels_map)
    # archive current population as foundation for next generation

    # MAY REDESIGN - Switch to parent method callback
    fx_data_tree_write(trees, 'a')

    return trees

#+++++++++++++++++++++++++++++++++++++++++++++
#   Methods to Train and Test a Tree         |
#+++++++++++++++++++++++++++++++++++++++++++++
# used by: Population
def fx_fitness_gym(trees, data_train, kernel, data_train_rows,
                   tf_device_log, class_labels, tf_device, terminals,
                   precision, log, pause, error, evaluate_tree,
                   fx_fitness_labels_map):
    '''
    Part 1 evaluates each expression against the data, line for line.
    This is the most time consuming and computationally expensive part of
    genetic programming. When GPUs are available, the performance can increase
    by many orders of magnitude for datasets measured in millions of data.

    Part 2 evaluates every Tree in each generation to determine which have
    the best, overall fitness score. This could be the highest or lowest
    depending upon if the fitness function is maximising (higher is better)
    or minimising (lower is better). The total fitness score is then saved
    with each Tree in the external .csv file.

    Part 3 compares the fitness of each Tree to the prior best fit in order
    to track those that improve with each comparison. For matching
    functions, all the Trees will have the same fitness score, but they
    may present more than one solution. For minimisation and maximisation
    functions, the final Tree should present the best overall fitness for
    that generation. It is important to note that Part 3 does *not* in any
    way influence the Tournament Selection which is a stand-alone process.

    Called by: fx_karoo_gp, fx_eval_generations

    Arguments required: trees
    '''

    new_trees = []
    for tree in trees:

        ### PART 1 - GENERATE MULTIVARIATE EXPRESSION FOR EACH TREE ###
        log(f'\t\033[36mTree {tree.id} '
            f'yields (sym):\033[1m {tree.expression} \033[0;0m')

        # get sympified expression and process it with TF - tested 2017 02/02
        new_trees.append(evaluate_tree(
            tree, data_train, tf_device_log, kernel, class_labels, tf_device,
            terminals, precision, log, fx_fitness_labels_map
        ))

        # MAY REDESIGN: Moved fitness_dict calculation elsewhere

    return trees

# used by: Population
def fx_fitness_eval(expr, data, tf_device_log, kernel, class_labels,
                    tf_device, terminals, fx_fitness_labels_map,
                    get_pred_labels=False):

    '''
    Computes tree expression using TensorFlow (TF) returning results and
    fitness scores.

    This method orchestrates most of the TF routines by parsing input
    string 'expression' and converting it into a TF operation graph
    which is then processed in an isolated TF session to compute the
    results and corresponding fitness values.

        'self.tf_device': controls which device will be used
            for computations (CPU or GPU).
        'self.tf_device_log': controls device placement logging (debug only).

    Args:
        'expr': a string containing math expression to be computed on
            the data. Variable names should match corresponding terminal
            names in 'self.terminals'.

        'data': an 'n by m' matrix of the data points containing n
            observations and m features per observation. Variable order
            should match corresponding order of terminals in 'self.terminals'.

        'get_pred_labels': a boolean flag which controls whether the
            predicted labels should be extracted from the evolved results.
            This applies only to the CLASSIFY kernel and defaults to 'False'.

    Returns:
        A dict mapping keys to the following outputs:
            'result': an array of the results of applying given
                expression to the data
            'pred_labels': an array of the predicted labels extracted from
                the results; defined only for CLASSIFY kernel, else None
            'solution': an array of the solution values extracted from
                the data (variable 's' in the dataset)
            'pairwise_fitness': an array of the element-wise results of
                applying corresponding fitness kernel function
            'fitness' - aggregated scalar fitness score

    Called by: fx_karoo_pause, fx_data_params_write, fx_fitness_gym

    Arguments required: expr, data
    '''

    # Initialize TensorFlow session
    # Reset TF internal state and cache (after previous processing)
    tf.reset_default_graph()
    config = tf.ConfigProto(log_device_placement=tf_device_log,
                            allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        with sess.graph.device(tf_device):

            # 1 - Load data into TF vectors
            tensors = {}
            terminal_symbols = [t.symbol for t in terminals.get()]
            terminal_symbols.append('s')  # Add column for solution
            for i, term in enumerate(terminal_symbols):
                # converts data into vectors
                tensors[term] = tf.constant(data[:, i], dtype=tf.float32)

            # 2- Transform string expression into TF operation graph
            result = fx_fitness_expr_parse(expr, tensors)
            # a placeholder, applies only to CLASSIFY kernel
            pred_labels = tf.no_op()
            # solution value is assumed to be stored in 's' terminal
            solution = tensors['s']

            # 3- Add fitness computation into TF graph
            if kernel == 'c':  # CLASSIFY kernel

                '''
                Creates element-wise fitness computation TensorFlow (TF)
                sub-graph for CLASSIFY kernel.

                This method uses the 'sympified' (SymPy) expression
                ('algo_sym') created in fx_eval_poly() and the data set
                loaded at run-time to evaluate the fitness of the selected
                kernel.

                This multiclass classifer compares each row of a given
                Tree to the known solution, comparing predicted labels
                generated by Karoo GP against the true classs labels.
                This method is able to work with any number of class
                labels, from 2 to n. The left-most bin includes -inf.
                The right-most bin includes +inf. Those inbetween are by
                default confined to the spacing of 1.0 each, as defined by:

                    (solution - 1) < result <= solution

                The skew adjusts the boundaries of the bins such that they
                fall on both the negative and positive sides of the
                origin. At the time of this writing, an odd number of
                class labels will generate an extra bin on the positive
                side of origin as it has not yet been determined the
                effect of enabling the middle bin to include both a
                negative and positive result.
                '''

                # was breaking with upgrade from Tensorflow 1.1 to 1.3;
                #  fixed by Iurii by replacing [] with () as of 20171026
                # if get_pred_labels:
                #      pred_labels = tf.map_fn(self.fx_fitness_labels_map,
                #                              result, dtype=[tf.int32, tf.string],
                #                              swap_memory = True)
                if get_pred_labels:
                    pred_labels = tf.map_fn(fx_fitness_labels_map,
                                            result, dtype=(tf.int32, tf.string),
                                            swap_memory=True)

                skew = (class_labels / 2) - 1

                rule11 = tf.equal(solution, 0)
                rule12 = tf.less_equal(result, 0 - skew)
                rule13 = tf.logical_and(rule11, rule12)

                rule21 = tf.equal(solution, class_labels - 1)
                rule22 = tf.greater(result, solution - 1 - skew)
                rule23 = tf.logical_and(rule21, rule22)

                rule31 = tf.less(solution - 1 - skew, result)
                rule32 = tf.less_equal(result, solution - skew)
                rule33 = tf.logical_and(rule31, rule32)

                pairwise_fitness = tf.cast(
                    tf.logical_or(tf.logical_or(rule13, rule23), rule33),
                    tf.int32
                )


            elif kernel == 'r':  # REGRESSION kernel

                '''
                A very, very basic REGRESSION kernel which is not designed
                to perform well in the real world. It requires that you
                raise the minimum node count to keep it from converging
                on the value of '1'. Consider writing or integrating a
                more sophisticated kernel.
                '''

                pairwise_fitness = tf.abs(solution - result)


            elif kernel == 'm':  # MATCH kernel

                '''
                This is used for demonstration purposes only.
                '''

                # pairwise_fitness = tf.cast(tf.equal(solution, result),
                #     tf.int32)  # breaks due to floating points
                # fixes above issue by checking if a float value lies
                # within a range of values
                RTOL, ATOL = 1e-05, 1e-08
                pairwise_fitness = tf.cast(
                    tf.less_equal(tf.abs(solution - result),
                                  ATOL + RTOL * tf.abs(result)),
                    tf.int32
                )

            # elif self.kernel == '[other]':  # use others as a template

            else:
                raise ValueError(f'Kernel type is wrong or missing. '
                                 f'You entered {kernel}')

            fitness = tf.reduce_sum(pairwise_fitness)

            # Process TF graph and collect the results
            result, pred_labels, solution, fitness, pairwise_fitness = (
                sess.run([result, pred_labels, solution,
                          fitness, pairwise_fitness]))

    return {'result': result, 'pred_labels': pred_labels,
            'solution': solution, 'fitness': float(fitness),
            'pairwise_fitness': pairwise_fitness}

# used by: Population
def fx_fitness_expr_parse(expr, tensors):

    '''
    Extract expression tree from the string algo_sym and
    transform into TensorFlow (TF) graph.

    Called by: fx_fitness_eval

    Arguments required: expr, tensors
    '''

    tree = ast.parse(expr, mode='eval').body

    return fx_fitness_node_parse(tree, tensors)

# used by: Population
def fx_fitness_chain_bool(values, operation, tensors):

    '''
    Chains a sequence of boolean operations (e.g. 'a and b and c')
    into a single TensorFlow (TF) sub graph.

    Called by: fx_fitness_node_parse

    Arguments required: values, operation, tensors
    '''

    x = tf.cast(fx_fitness_node_parse(values[0], tensors), tf.bool)
    if len(values) > 1:
        return operation(x, fx_fitness_chain_bool(values[1:],
                                                  operation, tensors))
    else:
        return x

# used by: Population
def fx_fitness_chain_compare(comparators, ops, tensors):

    '''
    Chains a sequence of comparison operations (e.g. 'a > b < c')
    into a single TensorFlow (TF) sub graph.

    Called by: fx_fitness_node_parse

    Arguments required: comparators, ops, tensors
    '''

    x = fx_fitness_node_parse(comparators[0], tensors)
    y = fx_fitness_node_parse(comparators[1], tensors)
    if len(comparators) > 2:
        return tf.logical_and(operators[type(ops[0])](x, y), fx_fitness_chain_compare(comparators[1:], ops[1:], tensors))
    else:
        return operators[type(ops[0])](x, y)

# used by: Population
def fx_fitness_node_parse(node, tensors):

    '''
    Recursively transforms parsed expression tree into TensorFlow (TF) graph.

    Called by: fx_fitness_expr_parse, fx_fitness_chain_bool,
               fx_fitness_chain_compare

    Arguments required: node, tensors
    '''

    if isinstance(node, ast.Name):  # <tensor_name>
        return tensors[node.id]

    elif isinstance(node, ast.Num):  # <number>
        #shape = tensors[tensors.keys()[0]].get_shape()  # Python 2.7
        shape = tensors[list(tensors.keys())[0]].get_shape()
        return tf.constant(node.n, shape=shape, dtype=tf.float32)

    elif isinstance(node, ast.BinOp):  # <left> <operator> <right>, e.g., x + y
        return operators[type(node.op)](
            fx_fitness_node_parse(node.left, tensors),
            fx_fitness_node_parse(node.right, tensors)
        )

    elif isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., -1
        return operators[type(node.op)](
            fx_fitness_node_parse(node.operand, tensors)
        )

    elif isinstance(node, ast.Call):   # <function>(<arguments>) e.g., sin(x)
        return operators[node.func.id](
            *[fx_fitness_node_parse(arg, tensors) for arg in node.args]
        )

    elif isinstance(node, ast.BoolOp):   # <left> <bool_operator> <right> e.g. x or y
        return fx_fitness_chain_bool(
            node.values, operators[type(node.op)], tensors
        )

    elif isinstance(node, ast.Compare):   # <left> <compare> <right> e.g., a > z
        return fx_fitness_chain_compare(
            [node.left] + node.comparators, node.ops, tensors
        )

    else:
        raise TypeError(node)

# used by: None (defined inside fx_fitness_eval)
def fx_fitness_labels_map_maker(class_labels):
    def fx_fitness_labels_map(result):

        '''
        For the CLASSIFY kernel, creates a TensorFlow (TF) sub-graph defined
        as a sequence of boolean conditions based upon the quantity of true
        class labels provided in the data .csv. Outputs an array of tuples
        containing the predicted labels based upon the result and
        corresponding boolean condition triggered.

        For comparison, the original (pre-TensorFlow) cod follows:

            # '-1' keeps a binary classification splitting over the origin
            skew = (self.class_labels / 2) - 1
            # check for first class (the left-most bin)
            if solution == 0 and result <= 0 - skew:
                fitness = 1
            # check for last class (the right-most bin)
            elif solution == self.class_labels - 1 and result > solution - 1 - skew:
                fitness = 1
            # check for class bins between first and last
            elif solution - 1 - skew < result <= solution - skew:
                fitness = 1
            else:
                fitness = 0  # no class match

        Called by: fx_fitness_eval

        Arguments required: result
        '''

        skew = (class_labels / 2) - 1
        label_rules = {
            class_labels - 1: (tf.constant(class_labels - 1),
                                    tf.constant(' > {}'.format(
                                        class_labels - 2 - skew)))
        }

        for class_label in range(class_labels - 2, 0, -1):
            cond = ((class_label - 1 - skew < result) &
                    (result <= class_label - skew))
            label_rules[class_label] = tf.cond(
                cond,
                lambda: (tf.constant(class_label),
                          tf.constant(' <= {}'.format(class_label - skew))),
                lambda: label_rules[class_label + 1]
            )

        # Moved from fx_fitness_eval
        pred_label = tf.cond(
            result <= 0 - skew,
            lambda: (tf.constant(0), tf.constant(' <= {}'.format(0 - skew))),
            lambda: label_rules[1]
        )

        return pred_label
    return fx_fitness_labels_map

# used by: Population
def fx_fitness_store(tree, result, kernel, precision):

    '''
    Records the fitness and length of the raw algorithm (multivariate
    expression) to the Numpy array. Parsimony can be used to apply
    pressure to the evolutionary process to select from a set of trees
    with the same fitness function the one(s) with the simplest (shortest)
    multivariate expression.

    Called by: fx_fitness_gym

    Arguments required: tree, fitness
    '''

    fitness = float(result['fitness'])
    fitness = round(fitness, precision)  # TODO: Best to only round for display
    tree.result = {}
    tree.result['fitness'] = fitness

    # relocated from fx_fitness_test_classify/regress/match
    tree.result['result'] = list(result['result'])
    tree.result['solution'] = list(result['solution'])
    if kernel == 'c':
        tree.result['pred_labels'] = result['pred_labels']
        tree.result['precision_recall'] = skm.classification_report(
            result['solution'], result['pred_labels'][0],
            zero_division=0)
        tree.result['confusion_matrix'] = skm.confusion_matrix(
            result['solution'], result['pred_labels'][0])
    elif kernel == 'r':
        tree.result['mean_squared_error'] = skm.mean_squared_error(
            result['result'], result['solution'])
    elif kernel == 'm':
        pass

    return tree
