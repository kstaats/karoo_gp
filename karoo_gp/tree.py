import numpy as np
from sympy import sympify

from . import Branch, Terminal, Function

class Tree:

    #++++++++++++++++++++++++++++
    #   Initialize              |
    #++++++++++++++++++++++++++++

    def __init__(self, id, root, tree_type='g', score=None):
        """Initialize a Tree from an id and Branch"""
        self.id = id        # The tree's position within population
        self.root = root    # The top Branch (depth = 0)
        self.tree_type = tree_type
        self.score = score or {}
        self.bfs_ref = None

    @classmethod
    def load(cls, id, expr, tree_type='f'):
        if expr[0] in ['f', 'g']:
            tree_type = expr[0]
            expr = expr[1:]
        elif expr[0] != '(':
            raise ValueError('Load-from expressions must start with tree type'
                             '("f"/"g") or "(".')
        root = Branch.load(expr, tree_type)
        tree = cls(id, root, tree_type)
        return tree

    #++++++++++++++++++++++++++++
    #   Generate Random         |
    #++++++++++++++++++++++++++++

    @classmethod
    def generate(cls, id, tree_type, tree_depth_base,
                 functions, terminals, rng, method='BFS'):
        '''Generate a new Tree object given starting parameters.'''
        root = Branch.generate(rng, functions, terminals, tree_type,
                               tree_depth_base, parent=None, method=method)
        return cls(id, root, tree_type)

    def copy(self, id=None, include_score=False):
        '''Return a duplicate, all attributes/state'''
        args = (id if id is not None else self.id,
                self.root.copy(),
                self.tree_type,
                self.score if include_score else None)
        return Tree(*args)

    #++++++++++++++++++++++++++++
    #   Display                 |
    #++++++++++++++++++++++++++++

    def __repr__(self):
        fit_repr = '' if self.fitness is None else f" fitness: {self.fitness}"
        expr = str(self.expression)
        if len(expr) > 16:
            expr = expr[:13] + "..."
        return f"<Tree {self.id}: '{expr}'{fit_repr}>"

    @property
    def raw_expression(self):
        """Return the raw (un-sympified) expression"""
        return self.root.parse()

    @property
    def expression(self):
        """Return the sympified expression"""
        return str(sympify(self.raw_expression))

    def save(self):
        return f'{self.tree_type}{self.root.save()}'

    def display(self, *args, **kwargs):
        return self.root.display(*args, **kwargs)

    #++++++++++++++++++++++++++++
    #   Query                   |
    #++++++++++++++++++++++++++++

    @property
    def depth(self):
        return self.root.depth

    @property
    def n_children(self):
        return self.root.n_children

    @property
    def fitness(self):
        '''Return fitness or -1 if not yet evaluated'''
        fitness = self.score.get('fitness')
        return None if fitness is None else float(fitness)

    def get_child(self, i_child, **kwargs):
        n_ch = self.n_children
        if i_child > n_ch:
            raise ValueError(f'Index "{i_child}" out of range ({n_ch}')
        return self.root.get_child(i_child, **kwargs)

    #++++++++++++++++++++++++++++
    #   Modify                  |
    #++++++++++++++++++++++++++++

    def set_child(self, i_child, branch, **kwargs):
        if i_child == 0:
            self.root = branch
        n_ch = self.n_children
        if i_child > n_ch:
            raise ValueError(f'Index "{i_child}" out of range ({n_ch}')
        self.root.set_child(i_child, branch, **kwargs)
        self.bfs_ref = None

    def point_mutate(self, rng, functions, terminals, log):
        """Replace a node (including root) with random node of same type"""
        i_mutate = rng.integers(0, self.n_children + 1)
        log(f'\t \033[36mwith node\033[1m {i_mutate} \033[0;0m\033[36mchosen '
            f'for mutation\033[0;0m', display=['i'])
        branch = self.get_child(i_mutate)
        _type = type(branch.node)
        replace = {Terminal: terminals, Function: functions}[_type]
        branch.node = rng.choice(replace.get())

    def branch_mutate(self, rng, functions, terminals, tree_depth_max, log):
        """Replace a subtree (excluding root) with random subtree"""
        i_mutate = rng.integers(1, self.n_children + 1)
        branch = self.get_child(i_mutate)
        kids = f'and {branch.n_children} sub-nodes ' if branch.children else ''
        log(f'\t \033[36mwith node \033[1m {i_mutate} {kids}\033[0;0m\033[36m'
            f'chosen for mutation\033[0;0m', display=['i'])
        if self.tree_type == 'f':
            # Replace all subtree nodes with random node of same type
            for c in range(branch.n_children + 1):
                child = branch.get_child(c)
                _type = type(child.node)
                replace = {Terminal: terminals, Function: functions}[_type]
                child.node = rng.choice(replace.get())
        elif self.tree_type == 'g':
            # Replace subtree with new random subtree of same target depth
            depth = tree_depth_max - branch.height
            replacement = Branch.generate(rng, functions, terminals,
                                          self.tree_type, depth,
                                          force_function_root=False)
            self.set_child(i_mutate, replacement)

    def prune(self, rng, terminals, max_depth):
        """Shrink tree to a given depth."""
        if self.depth <= max_depth:
            return
        elif max_depth == 0 and type(self.root.node) != Terminal:  # Replace the root
            self.root = Branch(rng.choice(terminals.get()), self.tree_type)
        elif max_depth == 1:  # Prune the root
            self.root.prune(rng, terminals)
        else:  # Cycle through (BFS order), prune second-to-last depth
            last_depth_nodes = [self.root]
            for d in range(max_depth - 1, 0, -1):
                this_depth_nodes = []
                for parent in last_depth_nodes:
                    if parent.children:
                        this_depth_nodes.extend(parent.children)
                last_depth_nodes = this_depth_nodes
                if d == 1:  # second to last row
                    for node in this_depth_nodes:
                        node.prune(rng, terminals)

    def crossover(self, i, mate, i_mate, rng, terminals, tree_depth_max,
                  log, pause):
        """Replace node i (including subtree) with node i_mate from mate"""
        to_insert = mate.get_child(i_mate).copy()
        if to_insert.children:
            log(f'\n\033[36m From one parent:\033[0;0m\n {mate.display()} '
                f'\n\033[36m ... we copy branch\033[1m {i_mate} '
                f'\033[0;0m\033[36mas a new, sub-tree:\033[0;0m\n',
                display=['db'])
            log(to_insert.display(), display=['db'])
            log(f'\n\033[36m ... and insert it into a copy of the second '
                f'parent in place of the selected branch:\033[1m\n',
                display=['db'])
            log(self.get_child(i).display())
            pause(display=['db'])
        else:
            has_kids = self.get_child(i).n_children
            kids = f' and {has_kids} sub-nodes' if has_kids else ''
            log(f'\n\033[36m In a copy of one parent:\033[0;0m\n ',
                display=['db'])
            log(self.display(), display=['db'])
            log(f'\n\033[36m ... we remove node \033[1m '
                f'{i}{kids} \033[0;0m\033[36mand replace with a terminal from '
                f'branch_x:\033[0;0m\n', display=['db'])
            log(mate.get_child(i_mate).display(), display=['db'])
            pause(display=['db'])

        self.set_child(i, to_insert)
        initial_depth = self.depth
        self.prune(rng, terminals, tree_depth_max)
        pruned_depth = self.depth
        if initial_depth != pruned_depth:
            log(f'\n\033[36m ... and prune to depth \033[1m {pruned_depth}:',
                display=['db'])
        log(f'\n\033[36m This is the resulting offspring:\033[0;0m\n'
            f'{self.display()}', display=['db'])
        pause(display=['db'])
