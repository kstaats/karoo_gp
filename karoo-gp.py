#!/bin/python3
# Karoo GP - Genetic Programming for Classification and Symbolic Regression

'''
A word to the newbie, expert, and brave--
Even if you are highly experienced in Genetic Programming, it is
recommended that you review the 'Karoo User Guide' before running
this application. While your computer will not burst into flames
nor will the sun collapse into a black hole if you do not, you will
likely find more enjoyment of this particular flavour of GP with
a little understanding of its intent and design.

Without any command line arguments, Karoo GP relies upon user
settings and the datasets located in karoo_gp/files/.

    $ python karoo_gp_main.py


If you include the path to an external dataset, it will auto-load at launch:

    $ python karoo_gp_main.py /[path]/[to_your]/[filename].csv


If you include one or more additional arguments, they will
override the default values, as follows:

    -ker [r,c,m]                 fitness function: (r)egression, (c)lassification, or (m)atching
    -typ [f,g,r]                 Tree type: (f)ull, (g)row, or (r)amped half/half
    -bas [3...10]                maximum Tree depth for initial population
    -max [3...10]                maximum Tree depth for entire run
    -min [3 to 2^(bas+1) - 1]    minimum number of nodes
    -pop [10...1000]             number of trees in each generational population
    -gen [1...100]               number of generations
    -tor [7 per 100]             number of trees selected for tournament
    -evr [0.0...1.0]             decimal percent of pop generated through Reproduction
    -evp [0.0...1.0]             decimal percent of pop generated through Point Mutation
    -evb [0.0...1.0]             decimal percent of pop generated through Branch Mutation
    -evc [0.0...1.0]             decimal percent of pop generated through Crossover

If you include any of the above flags, then you *must* also
include a flag to load an external dataset.

    -fil [path]/[to]/[data].csv  an external dataset


An example is given, as follows:

    $ python karoo_gp_server.py -ker c -typ r -bas 4 -fil [path]/[to]/[data].csv

'''

import os
import sys
import pathlib
import argparse

import numpy as np
import pandas as pd

from karoo_gp import pause as menu
from karoo_gp import __version__, MultiClassifierGP, RegressorGP, MatchingGP, BaseGP

#++++++++++++++++++++++++++++++++++++++++++
#   User Interface for Configuation       |
#++++++++++++++++++++++++++++++++++++++++++

# either no command line argument, or only a filename is provided
if len(sys.argv) < 3:

    os.system('clear')
    print('\n\033[36m\033[1m')
    print('\t **   **   ******    *****    ******    ******       ******    ******')
    print('\t **  **   **    **  **   **  **    **  **    **     **        **    **')
    print('\t ** **    **    **  **   **  **    **  **    **     **        **    **')
    print('\t ****     ********  ******   **    **  **    **     **   ***  *******')
    print('\t ** **    **    **  ** **    **    **  **    **     **    **  **')
    print('\t **  **   **    **  **  **   **    **  **    **     **    **  **')
    print('\t **   **  **    **  **   **  **    **  **    **     **    **  **')
    print('\t **    ** **    **  **    **  ******    ******       ******   **')
    print('\033[0;0m')
    print('\t\033[36m Genetic Programming in Python with TensorFlow - '
          'by Kai Staats, version {}\033[0;0m'.format(__version__))
    print()

    while True:
        try:
            query = input('\t Select (c)lassification, (r)egression, '
                          '(m)atching, or (p)lay (default m): ')
            if query in ['c', 'r', 'm', 'p', '']:
                kernel = query or 'm'
                break
            else:
                raise ValueError()
        except ValueError:
            print('\t\033[32m Select from the options given. '
                  'Try again ...\n\033[0;0m')
        except KeyboardInterrupt:
            sys.exit()

    if kernel == 'p':  # play mode
        while True:
            try:
                query = input('\t Select (f)ull or (g)row (default g): ')
                if query in ['f', 'g', '']:
                    tree_type = query or 'g'
                    break
                else:
                    raise ValueError()
            except ValueError:
                print('\t\033[32m Select from the options given. '
                      'Try again ...\n\033[0;0m')
            except KeyboardInterrupt:
                sys.exit()

        while True:
            try:
                query = input('\t Enter the depth of the Tree (default 1): ')
                if query == '':
                    tree_depth_base = 1
                    break
                elif int(query) in list(range(1, 11)):
                    tree_depth_base = int(query)
                    break
                else:
                    raise ValueError()
            except ValueError:
                print('\t\033[32m Enter a number from 1 including 10. '
                      'Try again ...\n\033[0;0m')
            except KeyboardInterrupt:
                sys.exit()

        tree_depth_max = tree_depth_base
        tree_depth_min = 3
        tree_pop_max = 1
        gen_max = 1
        tourn_size = 0
        display = 's'  # for play mode, initialize, print fittest tree and quit
        # evolve_repro, evolve_point, evolve_branch, evolve_cross,
        # tourn_size, precision, filename are not required

    else:  # if any other kernel is selected

        while True:
            try:
                query = input('\t Select (f)ull, (g)row, or '
                              '(r)amped 50/50 method (default r): ')
                if query in ['f', 'g', 'r', '']:
                    tree_type = query or 'r'
                    break
                else:
                    raise ValueError()
            except ValueError:
                print('\t\033[32m Select from the options given. '
                      'Try again ...\n\033[0;0m')
            except KeyboardInterrupt:
                sys.exit()

        while True:
            try:
                query = input('\t Enter depth of the \033[3minitial\033[0;0m '
                              'population of Trees (default 3): ')
                if query == '':
                    tree_depth_base = 3
                    break
                elif int(query) in list(range(1, 11)):
                    tree_depth_base = int(query)
                    break
                else:
                    raise ValueError()
            except ValueError:
                print('\t\033[32m Enter a number from 1 including 10. '
                      'Try again ...\n\033[0;0m')
            except KeyboardInterrupt:
                sys.exit()

        while True:
            try:
                query = input('\t Enter maximum Tree depth (default %s): ' %
                              str(tree_depth_base))
                if query == '':
                    tree_depth_max = tree_depth_base
                    break
                elif int(query) in list(range(tree_depth_base, 11)):
                    tree_depth_max = int(query)
                    break
                else:
                    raise ValueError()
            except ValueError:
                print('\t\033[32m Enter a number from %s including 10. '
                      'Try again ...\n\033[0;0m' % str(tree_depth_base))
            except KeyboardInterrupt:
                sys.exit()

        # calc the max number of nodes for the given depth
        max_nodes = 2**(tree_depth_base+1) - 1

        while True:
            try:
                query = input('\t Enter minimum number of nodes for any given '
                              'Tree (default 3; max %s): ' % str(max_nodes))
                if query == '':
                    tree_depth_min = 3
                    break
                elif int(query) in list(range(3, max_nodes+1)):
                    tree_depth_min = int(query)
                    break
                else:
                    raise ValueError()
            except ValueError:
                print('\t\033[32m Enter a number from 3 including %s. '
                      'Try again ...\n\033[0;0m' % str(max_nodes))
            except KeyboardInterrupt:
                sys.exit()

        #while True:
            #try:
                #query = input('\t Select (p)artial or (f)ull operator '
                              #'inclusion (default p): ')
                #if query == '':
                    #swim = 'p'
                    #break
                #elif query in ['p','f']:
                    #swim = query
                    #break
                #else:
                    #raise ValueError()
            #except ValueError:
                #print('\t\033[32m Select from the options given. '
                      #'Try again ...\n\033[0;0m')
            #except KeyboardInterrupt:
                #sys.exit()

        while True:
            try:
                query = input('\t Enter number of Trees in each population '
                              '(default 100): ')
                if query == '':
                    tree_pop_max = 100
                    break
                elif int(query) in list(range(1, 1001)):
                    tree_pop_max = int(query)
                    break
                else:
                    raise ValueError()
            except ValueError:
                print('\t\033[32m Enter a number from 1 including 1000. '
                      'Try again ...\n\033[0;0m')
            except KeyboardInterrupt:
                sys.exit()

        # calculate the tournament size
        # default 7% can be changed by selecting (g)eneration and then 'ts'
        tourn_size = int(tree_pop_max * 0.07)
        if tourn_size < 2:
            # forces some diversity for small populations
            tourn_size = 2
        if tree_pop_max == 1:
            # in theory, supports the evolution of a single Tree - NEED TO FIX 2018 04/19
            tourn_size = 1

        while True:
            try:
                query = input('\t Enter max number of generations (default 10): ')
                if query == '':
                    gen_max = 10
                    break
                elif int(query) in list(range(1, 101)):
                    gen_max = int(query)
                    break
                else:
                    raise ValueError()
            except ValueError:
                print('\t\033[32m Enter a number from 1 including 100. '
                      'Try again ...\n\033[0;0m')
            except KeyboardInterrupt:
                sys.exit()

        if gen_max > 1:
            while True:
                try:
                    query = input('\t Display (i)nteractive, (g)eneration, '
                                  '(m)iminal, (s)ilent, or (d)e(b)ug (default m): ')
                    if query in ['i', 'g', 'm', 's', 'db', '']:
                        display = query or 'm'
                        break
                    else:
                        raise ValueError()
                except ValueError:
                    print('\t\033[32m Select from the options given. '
                          'Try again ...\n\033[0;0m')
                except KeyboardInterrupt:
                    sys.exit()

        else:
            display = 's'  # display mode is not used, but a value must be passed

    ### additional configuration parameters ###

    # quantity of a population generated through Reproduction
    evolve_repro = 0.1
    # quantity of a population generated through Point Mutation
    evolve_point = 0.1
    # quantity of a population generated through Branch Mutation
    evolve_branch = 0.2
    # quantity of a population generated through Crossover
    evolve_cross = 0.6
    # not required unless an external file is referenced
    filename = ''
    # not required unless saving to a specific dir in runs/
    output_dir = ''
    # number of floating points for the round function in 'fx_fitness_eval'
    precision = 6
    # require (p)artial or (f)ull set of features (operators)
    # for each Tree entering the gene_pool
    swim = 'p'
    # pause at the (d)esktop when complete, awaiting further
    # user interaction; or terminate in (s)erver mode
    mode = 'd'
    # random seed for reproducibility
    seed = None


#++++++++++++++++++++++++++++++++++++++++++
#   Command Line for Configuation         |
#++++++++++++++++++++++++++++++++++++++++++

else:  # 2 or more command line arguments are provided

    ap = argparse.ArgumentParser(description='Karoo GP Server')
    ap.add_argument('-ker', action='store', dest='kernel', default='c',
                    help='[c,r,m] fitness function: (r)egression, '
                         '(c)lassification, or (m)atching')
    ap.add_argument('-typ', action='store', dest='type', default='r',
                    help='[f,g,r] Tree type: (f)ull, (g)row, or (r)amped half/half')
    ap.add_argument('-bas', action='store', dest='depth_base', type=int, default=4,
                    help='[3...10] maximum Tree depth for the initial population')
    ap.add_argument('-max', action='store', dest='depth_max', type=int, default=None,
                    help='[3...10] maximum Tree depth for the entire run')
    ap.add_argument('-min', action='store', dest='depth_min', default=3,
                    help='minimum nodes, from 3 to 2^(base_depth +1) - 1')
    ap.add_argument('-pop', action='store', dest='pop_max', default=100,
                    help='[10...1000] number of trees per generation')
    ap.add_argument('-gen', action='store', dest='gen_max', default=10,
                    help='[1...100] number of generations')
    ap.add_argument('-tor', action='store', dest='tor_size', default=7,
                    help='[7 for each 100] recommended tournament size')
    ap.add_argument('-evr', action='store', dest='evo_r', default=0.1,
                    help='[0.0-1.0] decimal percent of pop generated '
                         'through Reproduction')
    ap.add_argument('-evp', action='store', dest='evo_p', default=0.1,
                    help='[0.0-1.0] decimal percent of pop generated '
                         'through Point Mutation')
    ap.add_argument('-evb', action='store', dest='evo_b', default=0.2,
                    help='[0.0-1.0] decimal percent of pop generated '
                         'through Branch Mutation')
    ap.add_argument('-evc', action='store', dest='evo_c', default=0.6,
                    help='[0.0-1.0] decimal percent of pop generated '
                         'through Crossover')
    ap.add_argument('-fil', action='store', dest='filename', default='',
                    help='/path/to_your/[data].csv')
    ap.add_argument('-out', action='store', dest='output_dir', default='',
                    help='/path/to_your/output_dir/')
    ap.add_argument('-rsd', action='store', dest='seed', default=None,
                    help='seed for the random number generator')

    args = ap.parse_args()

    # pass the argparse defaults and/or user inputs to the required variables
    kernel = str(args.kernel)
    tree_type = str(args.type)
    tree_depth_base = args.depth_base
    tree_depth_max = args.depth_max
    tree_depth_min = int(args.depth_min)
    tree_pop_max = int(args.pop_max)
    gen_max = int(args.gen_max)
    tourn_size = int(args.tor_size)
    evolve_repro = float(args.evo_r)
    evolve_point = float(args.evo_p)
    evolve_branch = float(args.evo_b)
    evolve_cross = float(args.evo_c)
    filename = str(args.filename)
    output_dir = str(args.output_dir)
    seed = None if args.seed is None else int(args.seed)

    # display mode is set to (s)ilent
    display = 's'
    # number of floating points for the round function in 'fx_fitness_eval'
    precision = 6
    # require (p)artial or (f)ull set of features (operators)
    # for each Tree entering the gene_pool
    swim = 'p'
    # pause at the (d)esktop when complete, awaiting further user interaction;
    # or terminate in (s)erver mode
    mode = 's'


#++++++++++++++++++++++++++++++++++++++++++
#   Define pause callback                 |
#++++++++++++++++++++++++++++++++++++++++++

# used by: karoo-gp interactive
def fx_karoo_pause_refer(model):

    '''
    Enables (g)eneration, (i)nteractive, and (d)e(b)ug display modes
    to offer the (pause) menu at each prompt.

    See fx_karoo_pause() for an explanation of the value being passed.

    Called by: the functions called by PART 4 of fx_karoo_gp()

    Arguments required: none
    '''

    menu = 1
    while menu == 1:
        menu = fx_karoo_pause(model)

# used by: karoo-gp interactive
def fx_karoo_pause(model):

    '''
    Pause the program execution and engage the user, providing a number of options.

    Called by: fx_karoo_pause_refer

    Arguments required: [0,1,2] where (0) refers to an end-of-run;
    (1) refers to any use of the (pause) menu from within the run,
    and anticipates ENTER as an escape from the menu to continue the run;
    and (2) refers to an 'ERROR!' for which the user may want to archive
    data before terminating. At this point in time, (2) is associated
    with each error but does not provide any special options).
    '''

    ### PART 1 - reset and pack values to send to menu.pause ###
    menu_dict = {
        'input_a': '',
        'input_b': 0,
        'display': model.display,
        'tree_depth_max': model.tree_depth_max,
        'tree_depth_min': model.tree_depth_min,
        'tree_pop_max': model.tree_pop_max,
        'gen_id': 0,
        'gen_max': model.gen_max,
        'tourn_size': model.tourn_size,
        'evolve_repro': model.evolve_repro,
        'evolve_point': model.evolve_point,
        'evolve_branch': model.evolve_branch,
        'evolve_cross': model.evolve_cross,
        'fittest_dict': {},
        'population_len': 0,
        'next_gen_len':0,
        'path': '',
    }
    # So it doesn't break if called before population is initialized
    if model.population is not None:
        pop_dict = {
            'gen_id': model.population.gen_id,
            'fittest_dict': model.population.fittest_dict,
            'population_len': len(model.population.trees),
            'next_gen_len': len(model.population.next_gen_trees),
            'path': model.path,
        }
        menu_dict = {**menu_dict, **pop_dict}

    # call the external function menu.pause
    menu_dict = menu(menu_dict)

    ### PART 2 - unpack values returned from menu.pause ###
    input_a = menu_dict['input_a']
    input_b = menu_dict['input_b']
    model.display = menu_dict['display']
    model.tree_depth_min = menu_dict['tree_depth_min']
    model.gen_max = menu_dict['gen_max']
    model.tourn_size = menu_dict['tourn_size']
    model.evolve_repro = menu_dict['evolve_repro']
    model.evolve_point = menu_dict['evolve_point']
    model.evolve_branch = menu_dict['evolve_branch']
    model.evolve_cross = menu_dict['evolve_cross']

    ### PART 3 - execute the user queries returned from menu.pause ###
    if input_a == 'esc':
        # breaks out of the fx_karoo_gp() or fx_karoo_pause_refer() loop
        return 2

    elif input_a == 'eval':  # evaluate a Tree against the TEST data
        if menu_dict['next_gen_len'] > 0:
            tree = model.population.next_gen_trees[input_b - 1]
        else:
            # TODO: Tell user which population is being used
            tree = model.population.trees[input_b - 1]
        model.log(f'Tree {tree.id} yields (raw): {tree.raw_expression}')
        model.log(f'Tree {tree.id} yields (sym): {tree.expression}')

        # Predict X_test and show predictions vs actual
        predictions = model.tree_predict(tree, model.X_test)
        for i, (y_pred, y_true) in enumerate(zip(predictions, model.y_test)):
            model.log(f'Data row {i} predicts: {y_pred}, actual: {y_true}')

        # Score the predictions and display result for each scoring parameter
        score = model.calculate_score(predictions, model.y_test)
        for k, v in score.items():
            model.log(f'{k.replace("_", " ").title()}: {v}')

    elif input_a in ['print_a', 'print_b']:  # print a Tree from population_a
        population = (model.population.trees if input_a == 'print_a'
                      else model.population.next_gen_trees)
        tree = population[input_b - 1]
        model.log(f'\nTree ID {input_b}\n'
                  f'Expression: {tree.expression}\n'
                  f'Raw: {tree.raw_expression}\n'
                  f'Fitness: {tree.fitness}'
                  f'\n{tree.display(method="list")}')

    elif input_a == 'population':  # list all Trees in population_a
        for tree in model.population.trees:
            model.log(f'Tree {tree.id} yields (sym): {tree.expression}')

    elif input_a == 'next_gen':  # list all Trees in next_gen_trees
        for tree in model.population.next_gen_trees:
            model.log(f'Tree {tree.id} yields (sym): {tree.expression}')

    elif input_a == 'load':  # load population_s to replace population_a
        model.load_population()
        model.log(f'\n\t Replacing population_a with population_s.csv')

    elif input_a == 'write':  # write the evolving next_gen_trees to disk
        path = model.save_population('b')
        model.log(f'\n\t All current members of the evolving next_gen_trees '
                  f'saved to {path}')

    elif input_a == 'add':
        # check for added generations, then exit fx_karoo_pause
        # and continue the run
        # if input_b > 0: self.gen_max = self.gen_max + input_b - REMOVED 2019 06/05
        model.gen_max = model.gen_max + input_b

    elif input_a == 'quit':
        model.fx_karoo_terminate()  # archive populations and exit

    return 1

#++++++++++++++++++++++++++++++++++++++++++
#   Load Data                             |
#++++++++++++++++++++++++++++++++++++++++++
karoo_dir = pathlib.Path(__file__).resolve().parent
suffix = dict(r='REGRESS', c='CLASSIFY', m='MATCH', p='PLAY')[kernel]
func_path = karoo_dir / 'karoo_gp' / 'files' / f'operators_{suffix}.csv'
filename = filename or karoo_dir / 'karoo_gp' / 'files' / f'data_{suffix}.csv'

functions = np.loadtxt(func_path, delimiter=',', skiprows=1, dtype=str)
functions = [f[0] for f in functions]  # Arity is now hard-coded by label
dataset = pd.read_csv(filename)
y = dataset.pop('s')
terminals = list(dataset.keys())
X, y = dataset.to_numpy(), y.to_numpy()

#++++++++++++++++++++++++++++++++++++++++++
#   Conduct the GP run                    |
#++++++++++++++++++++++++++++++++++++++++++

# Select the correct class for kernel
cls = {
    'c': MultiClassifierGP, 'r': RegressorGP, 'm': MatchingGP, 'p': BaseGP
}[kernel]

# Initialize the model
gp = cls(
    tree_type=tree_type,
    tree_depth_base=tree_depth_base,
    tree_depth_max=tree_depth_max,
    tree_depth_min=tree_depth_min,
    tree_pop_max=tree_pop_max,
    gen_max=gen_max,
    tourn_size=tourn_size,
    filename=filename,
    output_dir=output_dir,
    evolve_repro=evolve_repro,
    evolve_point=evolve_point,
    evolve_branch=evolve_branch,
    evolve_cross=evolve_cross,
    display=display,
    precision=precision,
    swim=swim,
    mode=mode,
    random_state=seed,
    pause_callback=fx_karoo_pause_refer,
    functions=functions,
    terminals=terminals,
)

# Fit to the data
gp.fit(X, y)

if kernel == 'p':
    tree = gp.population.trees[0]
    print(f'\nTree ID {tree.id}')
    print(f'  yields (raw): {tree.raw_expression}')
    print(f'  yields (sym): {tree.expression}\n')
    print(gp.population.trees[0].display(method='viz'))
    print(gp.population.trees[0].display(method='list'))
    # self.fx_data_tree_write(self.population.trees, 'a')

# Save files and exit
gp.fx_karoo_terminate()
