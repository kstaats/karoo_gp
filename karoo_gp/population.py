from functools import reduce

import numpy as np

from . import Tree

class Population:

    def __init__(self, model, trees, gen_id=1, history=None):
        """TODO"""
        self.model = model
        self.trees = trees
        self.gen_id = gen_id
        self.evaluated = False
        self.fittest_dict = {}
        self.history = history or []
        self.next_generation = []
        self.gene_pool = []

    @classmethod
    def generate(cls, model=None, functions=None, terminals=None,
                 tree_type='r', tree_depth_base=3, tree_pop_max=100):
        """Return a new Population of a type/amount trees"""
        trees = []
        args = (functions, terminals, model.rng)
        if tree_type == 'r':
            # (r)amped 50/50:  Create 1 full- and 1 grow-tree with each level of
            # depth, from 2 to the max depth.

            # Do as many full cycles of (n=max-2) depth as possible..
            n_cycles = int(tree_pop_max/2/tree_depth_base)
            for i in range(n_cycles):
                for d in range(tree_depth_base):
                    for _type in ['f', 'g']:
                        trees.append(Tree.generate(len(trees)+1, _type, d+1,
                                                   *args))

            # ..and add grow trees of base depth for the remainder.
            extras = tree_pop_max - len(trees)
            for i in range(extras):
                trees.append(Tree.generate(len(trees)+1,
                                           'g', tree_depth_base, *args))
        else:
            # (f)ull: Fill-in all nodes to the maximum depth
            # (g)row: Add nodes or terminals at random up to max depth
            for i in range(tree_pop_max):
                trees.append(Tree.generate(i+1, tree_type,
                                           tree_depth_base, *args))
        return cls(model, trees)

    def fittest(self):
        """Return the fittest tree of the population."""
        if not self.evaluated:
            self.model.log('Population has not yet been evaluated, returning'
                           'a random tree.')
            return self.trees[0]
        return reduce(self.model.fitness_compare, self.trees)

    def evaluate(self, X, y, X_hash=None):
        """Score all trees; use cached (by X_hash & expr) or calculate

        Uses model methods .predict and .score and manages model cache
        """
        predictions = self.model.batch_predict(X, self.trees, X_hash)
        for tree, y_pred in zip(self.trees, predictions):
            if X_hash is not None:
                cached_score = self.model.cache[X_hash].get(tree.expression)
                if cached_score:
                    tree.score = cached_score
                    continue
            tree.score = self.model.calculate_score(y_pred, y)
            if X_hash is not None:
                self.model.cache[X_hash][tree.expression] = tree.score
        self.history.append(self.fittest().save())
        self.evaluated = True

        self.fittest_dict = self.model.build_fittest_dict(self.trees)
        self.model.log(f'\n\033[36m {len(self.fittest_dict)} '
            f'trees\033[1m {np.sort(list(self.fittest_dict.keys()))} '
            f'\033[0;0m\033[36moffer the highest fitness scores.\033[0;0m')
        self.model.pause(display=['g'])

    def evolve(self, tree_pop_max, functions, terminals, swim='p',
               tree_depth_min=None, tree_depth_max=5, tourn_size=7,
               evolve_repro=0.1, evolve_point=0.1, evolve_branch=0.2,
               evolve_cross=0.6):
        """Return a new population evolved from self"""

        # Calculte number of new trees per evolution type
        evolve_ratios = dict(repro=evolve_repro, point=evolve_point,
                             branch=evolve_branch, cross=evolve_cross)
        if sum(evolve_ratios.values()) != 1.0:
            raise ValueError(f'Evolution parameters must sum to 1')
        # TODO: Using int(v * pop_max) is not optimal, but would cause the
        # tests to break if changed. Should be guaranteed to return a list
        # of length tree_pop_max.
        evolve_amounts = {k: int(v * tree_pop_max)
                          for k, v in evolve_ratios.items()}
        log = self.model.log
        pause = self.model.pause
        rng = self.model.rng
        # Create the list of eligible trees
        log('\n Prepare a viable gene pool ...', display=['i'])
        pause(display=['i'])
        self.fitness_gene_pool(swim, tree_depth_min, terminals)
        # Initialize new population and begin evolving new trees
        self.next_generation = []
        for evolve_type, amount in evolve_amounts.items():
            verb = dict(repro='Reproductions', point='Point Mutations',
                        branch='Branch Mutations', cross='Crossovers')
            log(f'  Perform {amount} {verb[evolve_type]} ...')
            pause(display=['i'])
            amount = amount // 2 if evolve_type == 'cross' else amount
            for _ in range(amount):
                # Create offspring from first parent
                parent = self.tournament(rng, tourn_size)
                log(f'\n\t\033[36mThe winner of the tournament is '
                    f'Tree:\033[1m{parent.id} \033[0;0m', display=['i'])

                offspring = parent.copy(id=len(self.next_generation) + 1)
                # Reproduce: add to new population as-is
                if evolve_type == 'repro':
                    self.next_generation.append(offspring)
                # Point Mutate: replace a single node
                elif evolve_type == 'point':
                    offspring.point_mutate(rng, functions, terminals, log)
                    self.next_generation.append(offspring)
                # Branch Mutate: replace a random subtree
                elif evolve_type == 'branch':
                    offspring.branch_mutate(rng, functions, terminals,
                                            tree_depth_max, log)
                    self.next_generation.append(offspring)
                # Crossover: create 2 unique offspring by splicing 2 parents
                elif evolve_type == 'cross':
                    # Select parent a, clone, select random i_a, repeat with b
                    # TODO: This could move to Tree.crossover(), but it changes
                    # the order of calls to rng, so would need to update tests.
                    parent_a = parent  # Renamed for clarity
                    offspring_a = offspring
                    offspring_a.id += 1  # Swap IDs, offspring_b added first
                    i_mutate_a = rng.integers(1, parent_a.n_children + 1)
                    parent_b = self.tournament(rng, tourn_size)
                    offspring_b = parent_b.copy(id=len(self.next_generation) + 1)
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
                    self.next_generation.append(offspring_b)
                    offspring_a.crossover(i_mutate_a, parent_b, i_mutate_b,
                                          rng, terminals, tree_depth_max,
                                          log, pause)
                    self.next_generation.append(offspring_a)

        # Return next generation as a Population
        next_gen = Population(model=self.model, trees=self.next_generation,
                              gen_id=self.gen_id + 1, history=self.history)
        return next_gen

    #++++++++++++++++++++++++++++
    #   Evolution               |
    #++++++++++++++++++++++++++++

    def fitness_gene_pool(self, swim='p', tree_depth_min=None, terminals=None):
        self.gene_pool = []
        for tree in self.trees:
            if swim == 'p':
                # each tree must have the min number of nodes defined by user
                if (tree.n_children + 1 >= tree_depth_min and
                    tree.expression != '1'):
                    self.model.log(f'\t\033[36m Tree {tree.id} has >= '
                                   f'{tree_depth_min} nodes and is added to '
                                   f'the gene pool\033[0;0m', display=['i'])
                    self.gene_pool.append(tree.id)
            elif swim == 'f':
                # each tree must contain at least one instance of each feature
                saved = tree.save()
                missing = sum([1 for t in terminals.get()
                               if f'({t.symbol})' not in saved])
                if not missing:
                    self.model.log(
                        f'\t\033[36m Tree {tree.id} includes at least one'
                        f' of each feature and is added to the gene '
                        f'pool\033[0;0m', display=['i'])
                    self.gene_pool.append(tree.id)
        self.model.log(f'\n\t The total population of the gene pool is '
                       f'{len(self.gene_pool)}', display=['i'])

    def tournament(self, rng, tournament_size=7):
        self.model.log('\n\tEnter the tournament ...', display=['i'])
        if not self.gene_pool:
            raise ValueError('Cannot conduct tournament: gene pool is empty')
        t_ids = [rng.choice(self.gene_pool) for _ in range(tournament_size)]
        trees = [self.trees[tree_id-1] for tree_id in t_ids]
        for t in trees:
            self.model.log(f'\t\033[36m Tree {t.id} has fitness {t.fitness}'
                           f'\033[0;0m', display=['i'])
        return reduce(self.model.fitness_compare, trees)
