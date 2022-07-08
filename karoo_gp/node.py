import ast
import math
from collections import defaultdict
from sympy import sympify
from . import NodeData, get_function_node

# Used by load, i.e. recreate node from label strings
operators_ref = {
    ast.Add: '+',
    ast.Sub: '-',
    ast.Mult: '*',
    ast.Div: '/',
    ast.Pow: '**',
    ast.USub: '-',
    ast.Gt: '>',
    ast.Lt: '<',
    ast.Eq: '==',
    ast.NotEq: '!=',
    ast.LtE: '<=',
    ast.GtE: '>=',
    ast.And: 'and',
    ast.Or: 'or',
    ast.Not: 'not',
    ast.IfExp: 'if',
    'abs': 'abs',  # These 3 not supported by ast so appears as type ast.Call
    'square': 'square',
    'sqrt': 'sqrt',
}

class Node:
    """An recursive tree element with a node, parent and children"""

    #++++++++++++++++++++++++++++
    #   Initialize              |
    #++++++++++++++++++++++++++++
    def __init__(self, node_data, tree_type, parent=None):
        self.node_data = node_data  # Sets label, arity.. via setter (below)
        self.tree_type = tree_type
        self.parent = parent
        self.children = None
        self.bfs_ref = None
        self.id = None

    @classmethod
    def load(cls, expr: str, tree_type, parent=None):
        """Load from a string expression, e.g. 'g((a)+(1))' (recursive)

        Accepts a string (when called from Tree or from breadth_wise_generate)
        or ast.Expression (recursive calls)
        """
        expr = ast.parse(expr, mode='eval')
        return cls.recursive_load(expr, tree_type, parent)

    @classmethod
    def recursive_load(cls, expr, tree_type, parent=None):
        if isinstance(expr, ast.Expression):
            expr = expr.body
        if isinstance(expr, ast.Name):  # Terminal
            node_data = NodeData(expr.id, 'terminal')
            node = Node(node_data, tree_type, parent)
        elif isinstance(expr, ast.Num):  # Constant
            node_data = NodeData(expr.value, 'constant')
            node = Node(node_data, tree_type, parent)
        else:
            # Parse the label
            if isinstance(expr, ast.IfExp):  # Cond
                label = operators_ref[type(expr)]
            elif isinstance(expr, ast.Compare):  # Gt, Lt..
                label = operators_ref[type(expr.ops[0])]
            # Operator or not/and
            elif isinstance(expr, (ast.UnaryOp, ast.BinOp, ast.BoolOp)):
                label = operators_ref[type(expr.op)]
            elif isinstance(expr, ast.Call):
                label = expr.func.id
            else:
                raise ValueError(f'Unsupported element in ast tree: {expr}')

            # Create a node from label
            node_data = get_function_node(label)
            node = Node(node_data, tree_type, parent)

            # Load children
            if isinstance(expr, ast.IfExp):  # Cond
                children = [expr.test, expr.body, expr.orelse]
            elif isinstance(expr, ast.Compare):  # Bool
                children = [expr.left] + expr.comparators
            elif isinstance(expr, ast.BoolOp):
                children = expr.values
            elif isinstance(expr, ast.UnaryOp):  # Operator, arity 1
                if isinstance(expr.op, ast.Not):
                    children = [expr.operand]
                else:
                    children = [expr.left]
            elif isinstance(expr, ast.BinOp):  # Operator, arity 2
                children = [expr.left, expr.right]
            else:
                children = expr.args
            node.children = [cls.recursive_load(c, tree_type, node) for c in children]
        return node

    #++++++++++++++++++++++++++++
    #   Generate Random         |
    #++++++++++++++++++++++++++++

    @classmethod
    def generate(cls, rng, get_nodes, tree_type, depth, parent=None,
                 method='BFS', force_function_root=True, node_types=None,
                 force_types=None):
        if method == 'BFS':
            return cls.breadth_first_generate(rng, get_nodes, tree_type,
                                              depth, parent,
                                              force_function_root, node_types,
                                              force_types)
        elif method == 'DFS':
            return cls.recursive_generate(rng, get_nodes, tree_type, depth,
                                          parent, force_function_root,
                                          node_types, force_types)

    @classmethod
    def breadth_first_generate(cls, rng, get_nodes, tree_type, tree_depth,
                               parent=None, force_function_root=True,
                               node_types=None, force_types=None):
        """Return a randomly-generated node and subtree breadth-first"""
        def fn(types=None, depth=tree_depth):  # Helper functions to save space
            types = types or ('operator', 'cond')
            return rng.choice(get_nodes(types, depth))
        def tm():
            return rng.choice(get_nodes(('terminal', 'constant')))
        # 1. Call the random functions in BFS order and save the output.
        # Generate a dict of lists, one entry for each level of depth, with
        # randomly-chosen node at each level.
        nodes_by_height = defaultdict(list)  # left-to-right lists of nodes
        if (tree_depth == 0 or
            (tree_type == 'g' and
             not force_function_root and
             rng.choice([False, True]))):
            root_node = tm()
        else:
            types = None
            if force_types:
                types = force_types[0]
                force_types = force_types[1:]
            if not types and node_types:
                types = node_types
            root_node = fn(types, tree_depth)
        nodes_by_height[0].append(root_node)
        for height in range(1, tree_depth + 1):
            for _parent in nodes_by_height[height-1]:
                for i in range(_parent.arity):
                    if (height == tree_depth or
                        (tree_type == 'g' and
                         _parent.min_depth <= 1 and
                         rng.choice([False, True]))):
                        node = tm()
                    else:
                        types = None
                        if force_types:
                            types = force_types[0]
                        if not types and _parent.child_type:
                            types = _parent.child_type[i]
                        node = fn(types, tree_depth - height)
                    nodes_by_height[height].append(node)
            if not nodes_by_height[height]:
                break
            if force_types:
                force_types = force_types[1:]

        # 2. Convert above into a string expr that's compatible with
        # Node.load().  Parse by recursively filling-in child nodes,
        # beginning with the root (depth 0). For node with arity n at depth d,
        # the children are the n left-most unused nodes from depth d + 1.
        used_index = {k: 0 for k in nodes_by_height}  # Last node-index used
        def next_from_depth(depth):  # Return left-most unused node
            output = nodes_by_height[depth][used_index[depth]]
            used_index[depth] += 1
            return output
        def build_expression(node, depth):  # Recursive function to build expr
            d = depth + 1
            if node.arity == 0:  # arity
                return f'({node.label})'
            elif node.arity == 1:
                return f'({node.label}{build_expression(next_from_depth(d), d)})'
            elif node.arity == 2:
                return (f'({build_expression(next_from_depth(d), d)}'
                        f'{node.label}'  # label
                        f'{build_expression(next_from_depth(d), d)})')
            elif node.label == 'if':
                return (f'({build_expression(next_from_depth(d), d)}'
                        f'if{build_expression(next_from_depth(d), d)}'
                        f'else{build_expression(next_from_depth(d), d)})')
            else:
                raise ValueError(f'Cannot build expression for {node}.')
        expr = build_expression(nodes_by_height[0][0], 0)
        return cls.load(expr, tree_type, parent=parent)

    @classmethod
    def recursive_generate(cls, rng, get_nodes, tree_type, depth,
                           parent=None, force_function_root=True,
                           node_types=None, force_types=None):
        """Return a randomly generated node and subtree depth-first (recursive)"""
        # Grow trees flip a coin for function/terminal (except root)
        if depth == 0:
            is_terminal = True
        elif tree_type == 'g' and (parent is not None or not force_function_root):
            is_terminal = rng.choice([False, True])
        else:
            is_terminal = False

        # Create terminal or function
        if is_terminal:
            node_data = rng.choice(get_nodes(('terminal', 'constant')))
            node = cls(node_data, tree_type, parent=parent)
        else:
            types = None
            if force_types:
                types = force_types[0]  # Can be falsy
                force_types = force_types[1:]
            types = types or node_types or ('operator', 'cond')
            node_data = rng.choice(get_nodes(types, depth))
            node = cls(node_data, tree_type, parent=parent)
            # Generate children
            node.children = []
            kwargs = dict(rng=rng, get_nodes=get_nodes, tree_type=tree_type,
                          depth=depth-1, force_types=force_types, parent=node)
            for i in range(node.arity):
                node_types = None if not node.child_type else node.child_type[i]
                node.children.append(cls.generate(**kwargs, node_types=node_types))
        return node

    # MAY REDESIGN: The original 'fx_..' functions use breadth-first ordering,
    # but the recursive method is depth-first. Changing the order would change
    # the test results, which would defeat the purpose of the tests. This
    # version supports both by adding the 'method' argument ('BFS' by default
    # or 'DFS') to class methods generate, get_child and set_child. The last 2
    # use a new method, i_bfs, which just returns the dfs index given a bfs
    # index from a cached dict, which is cleared when the tree is modified.
    #
    # TODO: Decide which to use by default. DFS is theoretically faster and
    # simpler to implement recursively, but recursive fx aren't cheap in
    # Python.

    def i_bfs(self, n):
        """Convert breadth-first index to depth-first index"""
        if n in [0, 1]:
            return n
        elif self.bfs_ref is None:
            n_children = self.n_children
            if n > n_children:
                raise ValueError(f'Index {n!r} out of range ({n_children})')
            # Generate a 2d list of nodes by depth
            nodes_by_depth = defaultdict(list)
            for i in range(n_children + 1):
                c, _ = self.recursive_get_child(i)
                nodes_by_depth[c.height].append(i)
            # Make a dict of {i_bfs: i_dfs} pairs
            self.bfs_ref = {}
            i_bfs = 0
            depths = list(nodes_by_depth.keys())
            for depth in depths:
                nodes = nodes_by_depth[depth]
                for node in nodes:
                    self.bfs_ref[i_bfs] = node
                    i_bfs += 1
        return self.bfs_ref[n]

    #++++++++++++++++++++++++++++
    #   Display                 |
    #++++++++++++++++++++++++++++

    def __repr__(self):
        return f"<Node: {self.node_data!r}>"

    @property
    def raw_expression(self):
        """Return full list of labels (recursive)"""
        if not self.children:
            return f'({self.label})'
        elif len(self.children) == 1:
            return f'({self.label}{self.children[0].raw_expression})'
        elif len(self.children) == 2:
            return (f'({self.children[0].raw_expression}{self.label}'
                    f'{self.children[1].raw_expression})')
        elif self.label == 'if':
            return (f'({self.children[0].raw_expression}'
                    f'if{self.children[1].raw_expression}'
                    f'else{self.children[2].raw_expression})')

    @property
    def expression(self):
        """Return simplified expression (recursive)"""
        raw_expression = self.raw_expression
        # Certain statements not supported by sympy, so only
        # sympify nodes *below* all of those.
        if 'if' in raw_expression:
            return (f'({self.children[0].expression};'
                    f'if{self.children[1].expression}'
                    f'else{self.children[2].expression})')
        elif any(f in raw_expression for f in ('and', 'or')):
            return (f'({self.children[0].expression}{self.label}'
                    f'{self.children[0].expression})')
        elif 'not' in raw_expression:
            return (f'(not{self.children[0].expression})')
        # Outputs of divide-by-zero, e.g. a/(b-b), are sympified to 'zoo'.
        # Replace any instance, and then re-sympify to let the 0 propogate.
        result = str(sympify(raw_expression))
        while 'zoo' in result:
            result = result.replace('zoo', '0')
            result = str(sympify(result))
        return result

    def display(self, *args, method='viz', **kwargs):
        if method == 'list':
            return self.display_list(*args, **kwargs)
        elif method == 'viz':
            return self.display_viz(*args, **kwargs)

    def display_list(self, prefix=''):
        parent = '' if self.parent is None else self.parent.id
        children = [] if not self.children else [c.id for c in self.children]
        output = (
            f'{prefix}NODE ID: {self.id}\n'
            f'{prefix}  type: {self.node_type}\n'
            f'{prefix}  label: {self.label}\tparent node: {parent}\n'
            f'{prefix}  arity: {self.arity}\tchild node(s): {children}\n\n')
        if self.children:
            output += ''.join(child.display_list(prefix=prefix+'\t')
                              for child in self.children)
        return output

    def display_viz(self, width=60, label_max_len=3):
        """Print a hierarchical tree representation of all nodes

        Cycle through depths starting with the root (centered). At each depth,
        for every node at that depth, portion the horizontal space among its
        children based on the max width of their subtree. Log the child nodes
        and their calculated width for the next loop, and draw the current node
        with lines connecting the first (left-most) child node to the last.
        """
        output = ''
        last_children = [(self, width)]  # Nodes to be added next loop
        for i in range(self.depth + 1):
            depth_output = ''
            depth_children = []
            for (node, subtree_width) in last_children:
                label = ' ' if node is None else str(node.label)[:label_max_len]
                this_output = label.center(subtree_width)
                this_children = []      # Children from this item
                cum_width = 0           # Cumulative character-width of all subtrees
                cum_cols = 0            # Cumulative maximum node-width of all subtrees
                # If no children, propogate the empty spaces below terminal
                if not node or not node.children:
                    cum_cols += 1
                    cum_width += subtree_width
                    this_children.append((None, subtree_width))
                # If children, fill-in this_output with '_' to first/last child label
                else:
                    children_cols = [c.n_cols for c in node.children]
                    total_cols = sum(children_cols)
                    for child, child_cols in zip(node.children, children_cols):
                        # Convert each child's 'cols' into character spacing
                        cum_cols += child_cols
                        cum_ratio = cum_cols / total_cols
                        target_width = math.ceil(cum_ratio * subtree_width) - cum_width
                        remaining_width = subtree_width - cum_width
                        child_width = min(target_width, remaining_width)
                        # Add record and update tracked values
                        this_children.append((child, child_width))
                        cum_width += child_width
                    # Add lines to the output
                    start_padding = this_children[0][1] // 2          # Midpoint of first child
                    end_padding = subtree_width - (this_children[-1][1] // 2)  # ..of last child
                    with_line = ''
                    for i, v in enumerate(this_output):
                        with_line += '_' if (i > start_padding and i < end_padding and v == ' ') else v
                    this_output = with_line
                depth_output += this_output
                depth_children += this_children
            last_children = depth_children
            if last_children:
                depth_output += '\n'
            output += depth_output
        return output

    #++++++++++++++++++++++++++++
    #   Query                   |
    #++++++++++++++++++++++++++++
    node_data_ = None

    @property
    def node_data(self):
        """Return the NodeData object of instance"""
        return self.node_data_

    @node_data.setter
    def node_data(self, node_data):
        """Update the NodeData object and set node-specific attributes"""
        self.node_data_ = node_data
        self.label = node_data.label
        self.node_type = node_data.node_type
        self.arity = node_data.arity
        self.min_depth = node_data.min_depth
        self.child_type = node_data.child_type

    @property
    def depth(self):
        """Return max distance to terminal (bottom) node"""
        ch = self.children
        return 0 if not ch else 1 + max([c.depth for c in ch])

    @property
    def height(self):
        """Return distance from root (top) node"""
        return 0 if not self.parent else 1 + self.parent.height

    @property
    def n_children(self):
        """Return the total childen in subtree excluding root (recursive)"""
        ch = self.children
        return 0 if not ch else len(ch) + sum([c.n_children for c in ch])

    @property
    def n_cols(self):
        """Return max node-width of entire subtree (recursive)"""
        ch = self.children
        return 1 if not ch else sum([c.n_cols for c in ch])

    #++++++++++++++++++++++++++++
    #   Modify                  |
    #++++++++++++++++++++++++++++

    def copy(self):
        """Return an unlinked copy of self (recursive)"""
        copy = Node(self.node_data, self.tree_type, self.parent)
        if self.children:
            copy.children = [c.copy() for c in self.children]
            for c in copy.children:
                c.parent = copy
        return copy

    def get_child(self, n, method='BFS'):
        """Returns the child in the nth position; supports BFS or DFS"""
        n = n if method != 'BFS' else self.i_bfs(n)
        child, _ = self.recursive_get_child(n)
        return child

    def recursive_get_child(self, n):
        """Returns child n (dfs) as the 0th element of a 2-tuple"""
        if n == 0:
            return self, n
        elif not self.children:
            return False, n-1
        else:
            n = n - 1
            for child in self.children:
                target, new_n = child.recursive_get_child(n)
                n = new_n
                if target:
                    return target, n
            return False, n

    def set_child(self, n, node, method='BFS'):
        """Replace the child in the nth position with supplied node"""
        if n == 0:
            raise ValueError('Cannot set child 0; replace from parent node')
        n = n if method != 'BFS' else self.i_bfs(n)
        complete, _ = self.recursive_set_child(n, node)
        self.bfs_ref = None  # Need to re-index as tree has changed
        return complete

    def recursive_set_child(self, n, node):
        """Replace child in nth position with given node (recursive)"""
        for i, child in enumerate(self.children):
            n -= 1
            if n == 0:
                self.children[i] = node
                self.children[i].parent = self
                return True, n
            if child.children:
                target, new_n = child.recursive_set_child(n, node)
                if target:
                    return True, n
                else:
                    n = new_n
        return False, n

    def prune(self, rng, get_nodes):
        """Replace all non-terminal child nodes with terminals"""
        if not self.children:
            return
        for i_c, child in enumerate(self.children):
            if child.children:
                replacement = Node(rng.choice(get_nodes(('terminal', 'constant'))),
                                   self.tree_type)
                self.set_child(i_c + 1, replacement)
