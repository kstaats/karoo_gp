import numpy as np
from sympy import sympify

from . import Branch, Terminal, Function

class Tree:

    #++++++++++++++++++++++++++++
    #   Initialize              |
    #++++++++++++++++++++++++++++

    def __init__(self, id, root, tree_type='g', score=None):
        """Initialize a Tree from an id and Branch"""
        # TODO: start from 0
        self.id = id        # The tree's position within population
        self.root = root    # The top Branch (depth = 0)
        self.tree_type = tree_type
        self.score = score or {}
        self.renumber()

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

    def renumber(self, method='BFS'):
        """Set the id of each branch of the subtree"""
        self.root.bfs_ref = None
        for i in range(0, self.n_children + 1):
            self.get_child(i, method=method).id = i

    def set_child(self, i_child, branch, **kwargs):
        if i_child == 0:
            self.root = branch
        n_ch = self.n_children
        if i_child > n_ch:
            raise ValueError(f'Index "{i_child}" out of range ({n_ch}')
        self.root.set_child(i_child, branch, **kwargs)
        self.renumber()

    def point_mutate(self, rng, functions, terminals, log):
        """Replace a node (including root) with random node of same type"""
        i_mutate = rng.randint(0, self.n_children + 1)
        log(f'Node {i_mutate} chosen for mutation', display=['i'])
        branch = self.get_child(i_mutate)
        _type = type(branch.node)
        replace = {Terminal: terminals, Function: functions}[_type]
        branch.node = rng.choice(replace.get())

    def branch_mutate(self, rng, functions, terminals, tree_depth_max, log):
        """Replace a subtree (excluding root) with random subtree"""
        i_mutate = rng.randint(1, self.n_children + 1)
        branch = self.get_child(i_mutate)
        from_type = {Terminal: 'term', Function: 'func'}[type(branch.node)]
        kids = f' and {branch.n_children} sub-nodes' if branch.children else ''
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
        to_type = {
            Terminal: 'term', Function: 'func'
        }[type(self.get_child(i_mutate).node)]
        log(f'Node {i_mutate}{kids} chosen for mutation, from {from_type} '
            f'to {to_type}', display=['i'])

    def prune(self, rng, terminals, max_depth):
        """Shrink tree to a given depth."""
        if self.depth <= max_depth:
            return
        elif max_depth == 0 and type(self.root.node) != Terminal:  # Replace the root
            self.root = Branch(rng.choice(terminals.get()), self.tree_type)
            self.renumber()
        elif max_depth == 1:  # Prune the root
            self.root.prune(rng, terminals)
            self.renumber()
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
            self.renumber()

    def crossover(self, i, mate, i_mate, rng, terminals, tree_depth_max,
                  log, pause):
        """Replace node i (including subtree) with node i_mate from mate"""
        to_insert = mate.get_child(i_mate).copy()

        # Get display fields before modifying
        from_disp = self.display()
        from_expr = self.get_child(i).parse()
        with_expr = to_insert.parse()

        self.set_child(i, to_insert)
        initial_depth = self.depth
        self.prune(rng, terminals, tree_depth_max)
        prune = (f' and prune to depth {self.depth}'
                 if self.depth != initial_depth else '')
        log(f'\nIn a copy of the first parent: \n{from_disp}\n\n '
            f'...we replace node {i} ({from_expr}) with node {i_mate} '
            f'from the second parent: {with_expr}{prune}\n'
            f'The resulting offspring is: \n{self.display()}',
            display=['db'])
        pause(display=['db'])
