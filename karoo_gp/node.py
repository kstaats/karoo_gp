import ast
import math
from collections import defaultdict
from sympy import sympify
import numpy as np
import tensorflow as tf
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
    """A tree node, with node data, a parent, and children"""

    #++++++++++++++++++++++++++++
    #   Initialize              |
    #++++++++++++++++++++++++++++
    def __init__(self, node_data, tree_type, parent=None):
        self.node_data = node_data  # Sets label, arity.. via setter (below)
        self.tree_type = tree_type
        self.parent = parent
        self.children = None
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
        # TODO: This doesn't need to use ast. Instead:
        # - expr is a string
        # - remove outer-most parenthesis
        # - split on first-level parenthesis into a list of labels & children
        # - find the label, make the appropriate Node
        # - load children recursively
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
                children = [expr.body, expr.test, expr.orelse]
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
    def generate(cls, rng, get_nodes, tree_type, depth, force_types=[],
                 parent=None):
        """Return a randomly generated node and subtree (recursive)"""

        # Determine allowed types
        force_types_ = []           # For children
        if force_types:  # If there's forced type for this depth,
            types = force_types[0]  # use it, and don't pass it on.
            force_types_ = force_types[1:]
        elif parent is not None:    # Otherwise, inherit from parent.
            # Doesn't work with mutate, because
            this_i = len(parent.children)
            types = parent.child_type[this_i]
        else:
            raise ValueError('Types must be specified for each node, either '
                             'by the "force_types" kwarg (required for root, '
                             'optional for depths), or inherited from the '
                             '"child_type" attribute of the parent Node.')

        # Decide whether terminal or function
        if depth == 0:  # Always return a terminal at depth 0
            is_terminal = True
        elif (tree_type == 'g' and         # For grow trees,
              'terminal' in types and      # if it's allowed,
              rng.choice([False, True])):  # flip a coin.
            is_terminal = True
        else:  # Otherwise, return a function
            is_terminal = False
            types = [t for t in types if t != 'terminal']

        # Generate a random node
        if is_terminal:
            node_data = rng.choice(get_nodes(['terminal', 'constant']))
            node = cls(node_data, tree_type, parent=parent)
        else:
            node_data = rng.choice(get_nodes(types, depth))
            node = cls(node_data, tree_type, parent=parent)
            # Generate children
            node.children = []
            for i in range(node.arity):
                node.children.append(cls.generate(
                    rng, get_nodes, tree_type, depth-1, force_types_, node))
        return node

    #++++++++++++++++++++++++++++
    #   Display                 |
    #++++++++++++++++++++++++++++

    def __repr__(self):
        return f"<Node: {self.node_data!r}>"

    def parse(self, simplified=False):
        """Parse nodes either raw (all nodes) or simplified to string"""

        if simplified:
            # Parse the raw subtree by calling again with (simplified=False)
            raw_expr = self.parse(simplified=False)

            # Some functions are not supported by sympy. If none of those funcs
            # appear in the subtree, return the sympified expression.
            if not any(f in raw_expr for f in ('if', 'and', 'or', 'not')):
                result = str(sympify(raw_expr))

                # '0/0' sympifies to 'nan', in which case don't accept sympified
                if result != 'nan':

                    # Some trees, e.g. 'a/(b-b)', result in a 0-division, which
                    # sympy parses as 'zoo'. If this happens, replace 'zoo' with 0
                    # and sympify again to let the zero propagate.
                    while 'zoo' in result:
                        result = result.replace('zoo', '0')
                        result = str(sympify(result))
                    return result

            # If any unsupported funcs DO appear in the subtree, return the
            # raw parsing of this node, but try to simplify each sub-node
            # individually (recursively) by including (simplified=True).

            # TODO: What sympy does, but for those unsupported functions.
            # e.g. ((a)if((a)==(a))else(a)) can clearly be simplified to (a)

        # Position the label and sutrees so that no relational information is
        # lost (i.e. liberal use of parenthesis) and it is ast-interpretable
        ws = ' ' if simplified else '' # whitespace
        if not self.children:  # terminals, constants
            return f'({self.label})'
        elif len(self.children) == 1:  # **, abs, square, sqrt, not
            return f'({self.label}{ws}{self.children[0].parse(simplified)})'
        elif len(self.children) == 2:  # arithmetic, comparison, and/or
            return (f'({self.children[0].parse(simplified)}{ws}{self.label}'
                    f'{ws}{self.children[1].parse(simplified)})')
        elif self.label == 'if':
            return (f'({self.children[0].parse(simplified)}'
                    f'{ws}if{ws}{self.children[1].parse(simplified)}'
                    f'{ws}else{ws}{self.children[2].parse(simplified)})')

    def display(self, *args, method='viz', **kwargs):
        """Return a printable string representation of the tree.

        Supports two visualization methids:
          - viz: display as vertical heirarchy of node labels & edges
          - list: display as indended list of node details
          - TODO min: list, but one line per tree and reduced info
        """
        if method == 'list':
            return self.display_list(*args, **kwargs)
        elif method == 'viz':
            return self.display_viz(*args, **kwargs)

    def display_list(self, prefix=''):
        """Return a printable string of node and subtree as an indented list

        TODO: Single-line only
        """
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
        """Return a printable hierarchical tree representation of all nodes

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
        self.numpy_func = node_data.numpy_func
        self.tensorflow_func = node_data.tensorflow_func

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
        """Return the total children in subtree excluding root (recursive)"""
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

    def get_child(self, n):
        """Return the child in the nth position; supports BFS or DFS"""
        child, _ = self.recursive_get_child(n)
        return child

    def recursive_get_child(self, n):
        """Return child n (dfs) as the 0th element of a 2-tuple"""
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

    def set_child(self, n, node):
        """Replace the child in the nth position with supplied node"""
        if n == 0:
            raise ValueError('Cannot set child 0; replace from parent node')
        complete, _ = self.recursive_set_child(n, node)
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

    def prune(self, rng, get_nodes, max_depth=1):
        """Replace all non-terminal child nodes with terminals"""
        if not self.children:
            return
        for i_c, child in enumerate(self.children):
            if child.min_depth >= max_depth:
                self.children[i_c] = Node(
                    rng.choice(get_nodes(arity=0)),
                    self.tree_type)
                self.children[i_c].parent = self
            elif max_depth > 1:
                child.prune(rng, get_nodes, max_depth - 1)

    #++++++++++++++++++++++++++++
    #   Predict                 |
    #++++++++++++++++++++++++++++

    def predict(self, X, X_index, engine='numpy'):
        """Recursively calculate and return the result of the tree on some data

        Args
        ====
        - X_dict: Data to be predicted on. If X were a pandas DataFrame,
                  X_dict is {term: X[:, term] for term in X.columns}
        - engine: Whether to execute on CPU (numpy) or GPU (tensorflow).
                  Based on experimentation it seems GPU is beneficial for
                  >30,000 samples.
        """
        if self.node_type == 'terminal':
            value = X[:, X_index[self.label]].astype(np.float64)
            if engine == 'tensorflow':
                value = tf.convert_to_tensor(value)
            return value
        elif self.node_type == 'constant':
            length = X.shape[0]
            value = np.repeat(self.label, length).astype(np.float64)
            if engine == 'tensorflow':
                value = tf.convert_to_tensor(value)
            return value
        else:
            if self.label == 'if':
                # Args are stored [value_if_true, condition, value_if_false]
                # for clearer rendering, i.e. 'a if b else c'.
                # Funcs expect [condition, value_if_true, value_if_false]
                # so here there are rearranged.
                reordered = [self.children[i] for i in [1, 0, 2]]
                args = [c.predict(X, X_index, engine) for c in reordered]
            else:
                args = [c.predict(X, X_index, engine) for c in self.children]
            if engine == 'numpy':
                return self.numpy_func(*args)
            if engine == 'tensorflow':
                return self.tensorflow_func(*args)
