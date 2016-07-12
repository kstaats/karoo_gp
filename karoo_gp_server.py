# Karoo GP Server
# Use Genetic Programming for Classification and Symbolic Regression
# by Kai Staats, MSc UCT / AIMS
# Much thanks to Emmanuel Dufourq and Arun Kumar for their support, guidance, and free psychotherapy sessions
# version 0.9.1.5

'''
A NOTE TO THE NEWBIE, EXPERT, AND BRAVE
Even if you are highly experienced in Genetic Programming, it is recommended that you review the 'Karoo Quick Start' before running 
this application. While your computer will not burst into flames nor will the sun collapse into a black hole if you do not, you will 
likely find more enjoyment of this particular flavour of GP with a little understanding of its intent and design.
'''

import sys # sys.path.append('modules/') # add the directory 'modules' to the current path
import karoo_gp_base_class; gp = karoo_gp_base_class.Base_GP()

# parameters configuration
gp.kernel = 'c' # ['r','c','m'] fitness function: (r)egression, (c)lassification, or (m)atching
gp.class_labels = 3 # [2,3, ...] number of class labels in the feature set
tree_type = 'r' # ['f','g','r'] Tree (t)ype: (f)ull, (g)row, or (r)amped half/half
tree_depth_base = 3 # [3,10] maximum Tree depth for the initial population, where nodes = 2^(depth + 1) - 1
gp.tree_depth_max = 3 # [3,10] maximum Tree depth for the entire run; introduces potential bloat
gp.tree_depth_min = 3 # [3,100] minimum number of nodes
gp.tree_pop_max = 100 # [10,1000] maximum population
gp.generation_max = 10 # [1,100] number of generations
gp.display = 'm' # ['i','m','g','s'] display mode: (i)nteractive, (m)inimal, (g)enerational, or (s)erver

gp.evolve_repro = int(0.1 * gp.tree_pop_max) # percentage of subsequent population to be generated through Reproduction
gp.evolve_point = int(0.1 * gp.tree_pop_max) # percentage of subsequent population to be generated through Point Mutation
gp.evolve_branch = int(0.2 * gp.tree_pop_max) # percentage of subsequent population to be generated through Branch Mutation
gp.evolve_cross = int(0.6 * gp.tree_pop_max) # percentage of subsequent population to be generated through Crossover Reproduction

gp.tourn_size = 10 # qty of individuals entered into each tournament (standard 10); can be adjusted in 'i'nteractive mode
gp.cores = 1 # replace '1' with 'int(gp.core_count)' to auto-set to max; can be adjusted in 'i'nteractive mode
gp.precision = 4 # the number of floating points for the round function in 'fx_fitness_eval'; hard coded

# run Karoo GP
gp.karoo_gp('server', tree_type, tree_depth_max)
