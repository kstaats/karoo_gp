import ast
import math
from collections import defaultdict
from . import Function, Terminal

# Used by load, i.e. recreate branch from symbol strings
operators_ref = {
    ast.Add: '+',
    ast.Sub: '-',
    ast.Mult: '*',
    ast.Div: '/',
    ast.Pow: '**',
    ast.USub: '-',
}

# TODO: Rename to 'Node'
class Branch:
    """An recursive tree element with a node, parent and children"""

    #++++++++++++++++++++++++++++
    #   Initialize              |
    #++++++++++++++++++++++++++++

    def __init__(self, node, tree_type, parent=None):
        self.node = node
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
        if type(expr) == ast.Expression:
            expr = expr.body
        if isinstance(expr, ast.Name):
            node = Terminal(expr.id)
            branch = Branch(node, tree_type, parent)
        elif isinstance(expr, ast.Num):
            node = Terminal(expr.value)
            branch = Branch(node, tree_type, parent)
        else:
            symbol = operators_ref[type(expr.op)]
            node = Function(symbol)
            branch = Branch(node, tree_type, parent)
            if isinstance(expr, ast.UnaryOp):
                branch.children = [cls.recursive_load(expr.left, tree_type, branch)]
            elif isinstance(expr, ast.BinOp):
                branch.children = [cls.recursive_load(expr.left, tree_type, branch),
                                   cls.recursive_load(expr.right, tree_type, branch)]
            else:
                raise ValueError("Unrecognized op type in load:", expr.op)
        return branch

    #++++++++++++++++++++++++++++
    #   Generate Random         |
    #++++++++++++++++++++++++++++

    @classmethod
    def generate(cls, rng, functions, terminals, tree_type, depth, parent=None,
                 method='BFS', force_function_root=True):
        if method == 'BFS':
            return cls.breadth_first_generate(rng, functions, terminals,
                                              tree_type, depth, parent,
                                              force_function_root)
        elif method == 'DFS':
            return cls.recursive_generate(rng, functions, terminals,
                                          tree_type, depth, parent,
                                          force_function_root)

    @classmethod
    def breadth_first_generate(cls, rng, functions, terminals, tree_type,
                               tree_depth, parent=None,
                               force_function_root=True):
        """Return a randomly-generated branch and subtree breadth-first"""
        def fn():  # Helper functions to save space
            choice = rng.choice(functions.get())
            return (choice.symbol, choice.arity)
        def tm():
            choice = rng.choice(terminals.get())
            return (choice.symbol, 0)
        # 1. Call the random functions in BFS order and save the output.
        # Generate a dict of lists, one entry for each level of depth, with
        # randomly-chosen node at each level.
        nodes_by_depth = defaultdict(list)  # left-to-right lists of nodes
        if tree_depth == 0:
            root_node = tm()
        elif (tree_type == 'g' and not force_function_root and
              rng.choice([False, True])):
            # In special cases (branch mutate), root can be a terminal
            root_node = tm()
        else:
            root_node = fn()
        nodes_by_depth[0].append(root_node)
        for i in range(1, tree_depth + 1):
            sum_parent_arity = sum(p[1] for p in nodes_by_depth[i-1])
            if not sum_parent_arity:
                break  # Grow trees can be less than tree_depth
            for _ in range(sum_parent_arity):
                if i == tree_depth:
                    node = tm()  # The bottom nodes are always terminals
                elif tree_type == 'f':
                    node = fn()  # For 'full', others are functions
                elif tree_type == 'g':  # For 'grow', others are coin-flips
                    node = rng.choice([fn, tm])()
                else:
                    raise ValueError('Only (f)ull and (g)row trees supported')
                nodes_by_depth[i].append(node)

        # 2. Convert above into a string expr that's compatible with
        # Branch.load().  Parse by recursively filling-in child nodes,
        # beginning with the root (depth 0). For node with arity n at depth d,
        # the children are the n left-most unused nodes from depth d + 1.
        used_index = {k: 0 for k in nodes_by_depth}  # Last node-index used
        def next_from_depth(depth):  # Return left-most unused node
            output = nodes_by_depth[depth][used_index[depth]]
            used_index[depth] += 1
            return output
        def build_expression(item, depth):  # Recursive function to build expr
            if item[1] == 0:  # arity
                return f'({item[0]})'  # symbol
            elif item[1] == 2:
                d = depth + 1
                return (f'({build_expression(next_from_depth(d), d)}'
                        f'{item[0]}'  # symbol
                        f'{build_expression(next_from_depth(d), d)})')
            else:
                raise ValueError(f'Arity {item[1]} not supported')
        expr = build_expression(nodes_by_depth[0][0], 0)
        return cls.load(expr, tree_type, parent=parent)

    @classmethod
    def recursive_generate(cls, rng, functions, terminals, tree_type, depth,
                           parent=None, force_function_root=True):
        """Return a randomly generated branch depth-first (recursive)"""
        # Grow trees flip a coin for function/terminal (except root)
        if depth == 0:
            is_terminal = True
        elif tree_type == 'g' and (parent is not None or not force_function_root):
            is_terminal = rng.choice([False, True])
        else:
            is_terminal = False

        # Create terminal or function
        if is_terminal:
            node = rng.choice(terminals.get())
            branch = cls(node, tree_type, parent=parent)
        else:
            node = rng.choice(functions.get())
            branch = cls(node, tree_type, parent=parent)
            # Generate children
            args = (rng, functions, terminals, tree_type, depth-1)
            branch.children = [cls.generate(*args, parent=branch) for c in range(node.arity)]
        return branch

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
        return f"<Branch: {self.node!r}>"

    def parse(self):
        """Return full list of symbols (recursive)"""
        if not self.children:
            return f'({self.node.symbol})'
        elif len(self.children) == 1:
            return f"{self.node.symbol}{self.children[0].parse()}"
        elif len(self.children) == 2:
            return f"{self.children[0].parse()}{self.node.symbol}{self.children[1].parse()}"

    def save(self):
        """Return a complete representation of state as a string"""
        if not self.children:
            return f"({self.node.symbol})"
        elif len(self.children) == 1:
            return f"({self.node.save}{self.children[0].save()})"
        elif len(self.children) == 2:
            return f"({self.children[0].save()}{self.node.symbol}{self.children[1].save()})"

    def display(self, *args, method='viz', **kwargs):
        if method == 'list':
            return self.display_list(*args, **kwargs)
        elif method == 'viz':
            return self.display_viz(*args, **kwargs)

    def display_list(self, prefix=''):
        _type = 'term' if type(self.node) is Terminal else 'func'
        _symbol = self.node.symbol
        _parent = '' if self.parent is None else self.parent.id
        _arity = 0 if _type == 'term' else self.node.arity
        _children = [] if not self.children else [c.id for c in self.children]
        output = (
            f'{prefix}NODE ID: {self.id}\n'
            f'{prefix}  type: {_type}\n'
            f'{prefix}  label: {_symbol}\tparent node: {_parent}\n'
            f'{prefix}  arity: {_arity}\tchild node(s): {_children}\n\n')
        if self.children:
            output += ''.join(child.display_list(prefix=prefix+'\t')
                              for child in self.children)
        return output

    def display_viz(self, width=60, symbol_max_len=3):
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
            for (branch, branch_width) in last_children:
                symbol = ' ' if branch is None else str(branch.node.symbol)[:symbol_max_len]
                this_output = symbol.center(branch_width)
                this_children = []      # Children from this item
                cum_width = 0           # Cumulative character-width of all subtrees
                cum_cols = 0            # Cumulative maximum node-width of all subtrees
                # If no children, propogate the empty spaces below terminal
                if not branch or not branch.children:
                    cum_cols += 1
                    cum_width += branch_width
                    this_children.append((None, branch_width))
                # If children, fill-in this_output with '_' to first/last child label
                else:
                    children_cols = [c.n_cols for c in branch.children]
                    total_cols = sum(children_cols)
                    for child, child_cols in zip(branch.children, children_cols):
                        # Convert each child's 'cols' into character spacing
                        cum_cols += child_cols
                        cum_ratio = cum_cols / total_cols
                        target_width = math.ceil(cum_ratio * branch_width) - cum_width
                        remaining_width = branch_width - cum_width
                        child_width = min(target_width, remaining_width)
                        # Add record and update tracked values
                        this_children.append((child, child_width))
                        cum_width += child_width
                    # Add lines to the output
                    start_padding = this_children[0][1] // 2          # Midpoint of first child
                    end_padding = branch_width - (this_children[-1][1] // 2)  # ..of last child
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
        copy = Branch(self.node, self.tree_type, self.parent)
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

    def set_child(self, n, branch, method='BFS'):
        """Replace the child in the nth position with supplied branch"""
        if n == 0:
            raise ValueError('Cannot set child 0; replace from parent node')
        n = n if method != 'BFS' else self.i_bfs(n)
        complete, _ = self.recursive_set_child(n, branch)
        self.bfs_ref = None  # Need to re-index as tree has changed
        return complete

    def recursive_set_child(self, n, branch):
        """Replace child in nth position with given branch (recursive)"""
        for i, child in enumerate(self.children):
            n -= 1
            if n == 0:
                self.children[i] = branch
                self.children[i].parent = self
                return True, n
            if child.children:
                target, new_n = child.recursive_set_child(n, branch)
                if target:
                    return True, n
                else:
                    n = new_n
        return False, n

    def prune(self, rng, terminals):
        """Replace all non-terminal child nodes with terminals"""
        if not self.children:
            return
        for i_c, child in enumerate(self.children):
            if type(child.node) is not Terminal:
                replacement = Branch(rng.choice(terminals.get()),
                                     self.tree_type)
                self.set_child(i_c + 1, replacement)
