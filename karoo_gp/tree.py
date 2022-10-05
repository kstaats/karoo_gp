import numpy as np

from . import Node

class Tree:

    #++++++++++++++++++++++++++++
    #   Initialize              |
    #++++++++++++++++++++++++++++

    def __init__(self, id, root, tree_type='g', score=None):
        """Initialize a Tree from an id and Node"""
        # TODO: start from 0
        self.id = id        # The tree's position within population
        self.root = root    # The top Node (depth = 0)
        self.tree_type = tree_type
        self.score = score or {}
        self.renumber()

        # This flag designates a tree which results in an error.
        # Some allowed operators will return 'nan' or 'inf' values, e.g.:
        #   '2**(1/0.0001) -> inf'  |  'arcsin(2) -> nan'
        self.is_unfit = False

    @classmethod
    def load(cls, id, expr, tree_type='f'):
        if expr[0] in ['f', 'g']:
            tree_type = expr[0]
            expr = expr[1:]
        elif expr[0] != '(':
            raise ValueError(f'Load-from expressions must start with tree type'
                             f' (f/g) or parenthesis; got {expr[0]!r}')
        root = Node.load(expr, tree_type)
        tree = cls(id, root, tree_type)
        return tree

    #++++++++++++++++++++++++++++
    #   Generate Random         |
    #++++++++++++++++++++++++++++

    @classmethod
    def generate(cls, id=None, tree_type='g', tree_depth_base=3,
                 get_nodes=None, rng=None, force_types=None):
        """Generate a new Tree object given starting parameters."""
        root = Node.generate(rng, get_nodes, tree_type, tree_depth_base,
                             parent=None, force_types=force_types)
        return cls(id, root, tree_type)

    def copy(self, id=None, include_score=False):
        """Return a duplicate, all attributes/state"""
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
        """Return the raw (full) expression"""
        return self.root.parse()

    @property
    def expression(self):
        """Return the simplified expression"""
        return self.root.parse(simplified=True)

    def save(self):
        """Return a re-loadable string of the tree and key attributes"""
        return f'{self.tree_type}{self.raw_expression}'

    def display(self, *args, **kwargs):
        """Return a printable string of all nodes"""
        return self.root.display(*args, **kwargs)

    #++++++++++++++++++++++++++++
    #   Query                   |
    #++++++++++++++++++++++++++++

    @property
    def depth(self):
        """Return the maximum depth (distance from root) of any node"""
        return self.root.depth

    @property
    def n_children(self):
        """Return the total number of nodes in subtree"""
        return self.root.n_children

    @property
    def fitness(self):
        """Return fitness or None if not yet evaluated"""
        fitness = self.score.get('fitness')
        return None if fitness is None else float(fitness)

    def get_child(self, i_child, **kwargs):
        """Return the child node in the ith position"""
        n_ch = self.n_children
        if i_child > n_ch:
            raise ValueError(f'Index "{i_child}" out of range ({n_ch}')
        return self.root.get_child(i_child, **kwargs)

    #++++++++++++++++++++++++++++
    #   Modify                  |
    #++++++++++++++++++++++++++++

    def renumber(self):
        """Set the id of each node of the subtree"""
        for i in range(0, self.n_children + 1):
            self.get_child(i).id = i

    def set_child(self, i_child, node, **kwargs):
        """Replace the nth child with the given node"""
        if i_child == 0:
            self.root = node
        n_ch = self.n_children
        if i_child > n_ch:
            raise ValueError(f'Index "{i_child}" out of range ({n_ch}')
        self.root.set_child(i_child, node, **kwargs)
        self.renumber()

    # In mutation, node types within each tuple can be swapped
    # TODO: Add a Node.inherited_type() which gets parent.child_type
    # for its index, replace swappable with
    # - get_nodes(node.inherited_type(), arity=node.arity)
    swappable = [['terminal', 'constant'], ['operator'], ['cond'], ['bool']]
    def point_mutate(self, rng, get_nodes, log):
        """Replace a node (including root) with random node of same type"""
        i_mutate = rng.randint(0, self.n_children + 1)
        log(f'Node {i_mutate} chosen for mutation', display=['i'])
        node = self.get_child(i_mutate)
        for group in self.swappable:
            if node.node_type in group:
                types = group
        same_arity_types = [n for n in get_nodes(types) if n.arity == node.arity]
        node.node_data = rng.choice(same_arity_types)

    def branch_mutate(self, rng, get_nodes, force_types, tree_depth_max, log):
        """Replace a subtree (excluding root) with random subtree"""
        i_mutate = rng.randint(1, self.n_children + 1)
        node = self.get_child(i_mutate)
        from_type = f'{node.node_type}'
        kids = f' and {node.n_children} sub-nodes' if node.children else ''
        if self.tree_type == 'f':
            # Replace all subtree nodes with random node of same type
            for c in range(node.n_children + 1):
                child = node.get_child(c)
                for group in self.swappable:
                    if child.node_type in group:
                        child.node_data = rng.choice(get_nodes(group))
                        break
        elif self.tree_type == 'g':
            # Replace subtree with new random subtree of same target depth
            height = node.height
            depth = tree_depth_max - height
            force_types_ = force_types[height:]
            if not force_types_:
                # TODO COnvert this to Node.inherited_types to replace swappable
                # Without force_types, Node.generate below will check
                node_child_i = next(i for i, c in enumerate(node.parent.children)
                                    if c is node)
                force_types_ = [node.parent.child_type[node_child_i]]
            replacement = Node.generate(
                rng, get_nodes, self.tree_type, depth, force_types=force_types_)
            self.set_child(i_mutate, replacement)
        to_type = f'{node.node_type}'
        log(f'Node {i_mutate}{kids} chosen for mutation, from {from_type} '
            f'to {to_type}', display=['i'])

    def prune(self, rng, get_nodes, max_depth):
        """Shrink tree to a given depth."""
        if self.depth <= max_depth:
            return
        elif (max_depth == 0 and self.root.arity > 0):  # Replace root
            self.root = Node(rng.choice(get_nodes(['terminal', 'constant'])),
                             self.tree_type)
            self.renumber()
        self.root.prune(rng, get_nodes, max_depth)
        self.renumber()

    def crossover(self, i, mate, i_mate, rng, get_nodes, tree_depth_max,
                  log, pause):
        """Replace node i (including subtree) with node i_mate from mate"""
        to_insert = mate.get_child(i_mate).copy()

        # Get display fields before modifying
        from_disp = self.display()
        from_expr = self.get_child(i).parse()
        with_expr = to_insert.parse()

        self.set_child(i, to_insert)
        initial_depth = self.depth
        self.prune(rng, get_nodes, tree_depth_max)
        prune = (f' and prune to depth {self.depth}'
                 if self.depth != initial_depth else '')
        log(f'\nIn a copy of the parent: \n{from_disp}\n\n '
            f'...we replace node {i}: {from_expr} with node {i_mate} '
            f'from the second parent: {with_expr}{prune}.'
            f'The resulting offspring is: \n{self.display()}',
            display=['db'])
        pause(display=['db'])
