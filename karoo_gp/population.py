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
        self.next_gen_trees = []
        self.gene_pool = []

    @classmethod
    def generate(cls, model=None, tree_type='r', tree_depth_base=3,
                 tree_pop_max=100, force_types=None):
        """Return a new Population of a type/amount trees"""
        trees = []
        kwargs = dict(get_nodes=model.get_nodes, rng=model.rng,
                      force_types=force_types)
        if tree_type == 'r':
            # (r)amped 50/50:  Create 1 full- and 1 grow-tree with each level of
            # depth, from 2 to the max depth.

            # Do as many full cycles of (n=max-2) depth as possible..
            n_cycles = int(tree_pop_max/2/tree_depth_base)
            for i in range(n_cycles):
                for d in range(tree_depth_base):
                    for _type in ['f', 'g']:
                        trees.append(Tree.generate(len(trees)+1, _type, d+1,
                                                   **kwargs))

            # ..and add grow trees of base depth for the remainder.
            extras = tree_pop_max - len(trees)
            for i in range(extras):
                trees.append(Tree.generate(len(trees)+1, 'g', tree_depth_base,
                                           **kwargs))
        else:
            # (f)ull: Fill-in all nodes to the maximum depth
            # (g)row: Add nodes or terminals at random up to max depth
            for i in range(tree_pop_max):
                trees.append(Tree.generate(i+1, tree_type, tree_depth_base,
                                           **kwargs))
        return cls(model, trees)

    def fittest(self):
        """Return the fittest tree of the population."""
        if not self.evaluated:
            raise ValueError('Population has not been evaluated yet')
        return reduce(self.model.fitness_compare, self.trees)

    def evaluate(self, X, y, X_hash=None):
        """Score all trees; use cached (by X_hash & expr) or calculate

        Uses model methods .predict and .score and manages model cache
        """
        self.model.log(f'\nEvaluate all Trees in Generation {self.gen_id}')
        self.model.pause(display=['i'])

        predictions = self.model.batch_predict(X, self.trees, X_hash)
        for tree, y_pred in zip(self.trees, predictions):
            cached = False
            if X_hash is not None:
                cached_score = self.model.cache_[X_hash].get(tree.expression)
                if cached_score:
                    tree.score = cached_score
                    cached = True
            if not cached:
                tree.score = self.model.calculate_score(y_pred, y)
                if X_hash is not None:
                    self.model.cache_[X_hash][tree.expression] = tree.score
            self.model.log(f'\nTree {tree.id} yields (sym): {tree.expression}'
                           f'\nwith fitness sum: {tree.fitness}')
        self.evaluated = True
        self.history.append(self.fittest().save())

        self.fittest_dict = self.model.build_fittest_dict(self.trees)
        self.model.log(f'\n{len(self.fittest_dict)} trees '
                       f'{np.sort(list(self.fittest_dict.keys()))} offer the '
                       f'highest fitness scores.')
        self.model.pause(display=['g'])

    def evolve(self, tree_pop_max, swim='p', tree_depth_min=None,
               tree_depth_max=5, tourn_size=7, evolve_repro=0.1,
               evolve_point=0.1, evolve_branch=0.2, evolve_cross=0.6):
        """Return a new population evolved from self"""
        log = self.model.log
        pause = self.model.pause
        rng = self.model.rng
        log(f'\nEvolve a population for Generation {self.gen_id + 1} ...')

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
        # Create the list of eligible trees
        log('\nPrepare a viable gene pool ...', display=['i'])
        pause(display=['i'])
        self.fitness_gene_pool(swim, tree_depth_min)
        # Initialize new population and begin evolving new trees
        self.next_gen_trees = []
        for evolve_type, amount in evolve_amounts.items():
            verb = dict(repro='Reproductions', point='Point Mutations',
                        branch='Branch Mutations', cross='Crossovers')
            log(f'\nPerform {amount} {verb[evolve_type]} ...')
            pause(display=['i'])
            amount = amount // 2 if evolve_type == 'cross' else amount
            for _ in range(amount):
                # Create offspring from first parent
                parent = self.tournament(rng, tourn_size)
                offspring = parent.copy(id=len(self.next_gen_trees) + 1)
                # Reproduce: add to new population as-is
                if evolve_type == 'repro':
                    self.next_gen_trees.append(offspring)
                # Point Mutate: replace a single node
                elif evolve_type == 'point':
                    offspring.point_mutate(rng, self.model.get_nodes, log)
                    self.next_gen_trees.append(offspring)
                # Node Mutate: replace a random subtree
                elif evolve_type == 'branch':
                    offspring.branch_mutate(rng, self.model.get_nodes,
                                            self.model.force_types,
                                            tree_depth_max, log)
                    self.next_gen_trees.append(offspring)
                # Crossover: create 2 unique offspring by splicing 2 parents
                elif evolve_type == 'cross':
                    # Select parent a, clone, select random i_a, repeat with b
                    # TODO: This could move to Tree.crossover(), but it changes
                    # the order of calls to rng, so would need to update tests.
                    parent_a = parent  # Renamed for clarity
                    offspring_a = offspring
                    offspring_a.id += 1  # Swap IDs, offspring_b added first
                    i_mutate_a = rng.randint(1, parent_a.n_children + 1)
                    parent_b = self.tournament(rng, tourn_size)
                    offspring_b = parent_b.copy(id=len(self.next_gen_trees) + 1)
                    i_mutate_b = rng.randint(1, parent_b.n_children + 1)

                    log('', display=['i'])  # Extra line to separate from tourn
                    for from_id, to_id, to_i in [
                        (parent_a.id, parent_b.id, i_mutate_b),
                        (parent_b.id, parent_a.id, i_mutate_a)]:
                        log(f'Crossover from parent {from_id} to offspring '
                            f'{to_id} at node {to_i}', display=['i'])

                    # Replace b's node i_b with a's node i_a & vice versa
                    offspring_b.crossover(i_mutate_b, parent_a, i_mutate_a,
                                          rng, self.model.get_nodes,
                                          tree_depth_max, log, pause)
                    self.next_gen_trees.append(offspring_b)
                    offspring_a.crossover(i_mutate_a, parent_b, i_mutate_b,
                                          rng, self.model.get_nodes,
                                          tree_depth_max, log, pause)
                    self.next_gen_trees.append(offspring_a)

        # Return next generation as a Population
        next_gen = Population(model=self.model, trees=self.next_gen_trees,
                              gen_id=self.gen_id + 1, history=self.history)
        return next_gen

    #++++++++++++++++++++++++++++
    #   Evolution               |
    #++++++++++++++++++++++++++++

    def fitness_gene_pool(self, swim='p', tree_depth_min=None):
        self.gene_pool = []
        for tree in self.trees:
            if swim == 'p':
                # each tree must have the min number of nodes defined by user
                if (tree.n_children + 1 >= tree_depth_min and
                    tree.expression != '1'):
                    self.model.log(f'Tree {tree.id} has >= {tree_depth_min} '
                                   f'nodes and is added to the gene pool',
                                   display=['i'])
                    self.gene_pool.append(tree.id)
            elif swim == 'f':
                # each tree must contain at least one instance of each feature
                saved = tree.save()
                missing = sum([1 for t in self.model.get_nodes(('terminal'))
                               if f'({t.label})' not in saved])
                if not missing:
                    self.model.log(
                        f'Tree {tree.id} includes at least one of each feature'
                        f' and is added to the gene pool', display=['i'])
                    self.gene_pool.append(tree.id)
        self.model.log(f'\nThe total population of the gene pool is '
                       f'{len(self.gene_pool)}', display=['i'])
        self.model.pause(display=['i'])

    def tournament(self, rng, tournament_size=7):
        self.model.log('\nEnter the tournament ...', display=['i'])
        if not self.gene_pool:
            raise ValueError('Cannot conduct tournament: gene pool is empty')
        t_ids = [rng.choice(self.gene_pool) for _ in range(tournament_size)]
        trees = [self.trees[tree_id-1] for tree_id in t_ids]
        for t in trees:
            self.model.log(f'Tree {t.id} has fitness {t.fitness}',
                           display=['i'])
        winner = reduce(self.model.fitness_compare, trees)
        self.model.log(f'The winner of the tournament is Tree: {winner.id}',
                       display=['i'])
        return winner
