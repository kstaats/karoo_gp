# Karoo GP Server
# Use Genetic Programming for Classification and Symbolic Regression
# by Kai Staats, MSc UCT / AIMS
# Much thanks to Emmanuel Dufourq and Arun Kumar for their support, guidance, and free psychotherapy sessions
# version 0.9

import sys # sys.path.append('modules/') # add the directory 'modules' to the current path
import karoo_gp_base_class; gp = karoo_gp_base_class.Base_GP()

# parameters configuration
gp.kernel = 'c' # ['a','c','m'] fitness function: ABS Value, Classification, or Matching
gp.class_labels = 3 # number of class labels in the feature set
tree_type = 'r' # ['f','g','r'] Tree type: full, grow, or ramped half/half
tree_depth_max = 3 # [3,10] maximum tree depth
gp.tree_depth_min = 3 #[3,100] minimum number of nodes
gp.tree_pop_max = 100 # [10,1000] maximum population
gp.generation_max = 10 # [1,1000] number of generations
gp.display = 'm' # ['i','m','g','s','db','t'] display mode: Interactive, Minimal, Generational, Server, Debug, or Timer

gp.evolve_repro = int(0.1 * gp.tree_pop_max) # percentage of subsequent population to be generated through Reproduction
gp.evolve_point = int(0.1 * gp.tree_pop_max) # percentage of subsequent population to be generated through Point Mutation
gp.evolve_branch = int(0.2 * gp.tree_pop_max) # percentage of subsequent population to be generated through Branch Mutation
gp.evolve_cross = int(0.6 * gp.tree_pop_max) # percentage of subsequent population to be generated through Crossover Reproduction

gp.tourn_size = 10 # qty of individuals entered into each tournament (standard 10); can be adjusted in 'i'nteractive mode
gp.cores = 1 # replace '1' with 'int(gp.core_count)' to auto-set to max; can be adjusted in 'i'nteractive mode
gp.precision = 4 # the number of floating points for the round function in 'fx_fitness_eval'; hard coded

# run Karoo GP
gp.karoo_gp('server', tree_type, tree_depth_max)
