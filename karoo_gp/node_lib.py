"""
Node Library

Karoo programs (Trees) are composed of different types of nodes, each of must
include certain attributes. Then, in several situations, a subset of those
nodes are selected based on position/parent type.

This module includes:
  - NodeData (dataclass): includes all required node attributes
  - function_lib (list): NodeData objects for all Karoo's supported operators
  - get_function_node (func): return NodeData for an operator symbol (e.g. '*')
  - get_nodes (func): a helper function for subsetting a list of NodeData's
"""


from dataclasses import dataclass
from typing import List

@dataclass
class NodeData:
    """Include all attributes required for a Node"""
    label: str                # The string representation used in expression
    node_type: str            # terminal, constant, operator, bool, or cond
    arity: int = 0            # Number of children
    min_depth: int = 0        # Depth (additional) required to be coherent
    child_type: list = None   # Eligible child node types (optional)

    def __repr__(self):
        return f'<NodeData label={self.label!r} type={self.node_type}>'

numeric = ['terminal', 'constant', 'operator', 'cond']
function_lib = [
    # Operators
    NodeData('+', 'operator', 2, min_depth=1, child_type=[numeric, numeric]),
    NodeData('-', 'operator', 2, min_depth=1, child_type=[numeric, numeric]),
    NodeData('*', 'operator', 2, min_depth=1, child_type=[numeric, numeric]),
    NodeData('/', 'operator', 2, min_depth=1, child_type=[numeric, numeric]),
    NodeData('**', 'operator', 2, min_depth=1, child_type=[numeric, numeric]),
    NodeData('abs', 'operator', 1, min_depth=1, child_type=[numeric]),
    NodeData('square', 'operator', 1, min_depth=1, child_type=[numeric]),
    NodeData('sqrt', 'operator', 1, min_depth=1, child_type=[numeric]),
    # NodeData('log', 'operator', 1, min_depth=1),  # TODO: Sometimes cause nans
    # NodeData('log1p', 'operator', 1, min_depth=1),
    # NodeData('cos', 'operator', 1, min_depth=1),
    # NodeData('sin', 'operator', 1, min_depth=1),
    # NodeData('tan', 'operator', 1, min_depth=1),
    # NodeData('arccos', 'operator', 1, min_depth=1),
    # NodeData('arcsin', 'operator', 1, min_depth=1),
    # NodeData('arctan', 'operator', 1, min_depth=1),
    # Boolean
    NodeData('==', 'bool', 2, min_depth=1, child_type=[numeric, numeric]),
    NodeData('!=', 'bool', 2, min_depth=1, child_type=[numeric, numeric]),
    NodeData('<', 'bool', 2, min_depth=1, child_type=[numeric, numeric]),
    NodeData('<=', 'bool', 2, min_depth=1, child_type=[numeric, numeric]),
    NodeData('>', 'bool', 2, min_depth=1, child_type=[numeric, numeric]),
    NodeData('>=', 'bool', 2, min_depth=1, child_type=[numeric, numeric]),
    NodeData('and', 'bool', 2, min_depth=2, child_type=[['bool'], ['bool']]),
    NodeData('or', 'bool', 2, min_depth=2, child_type=[['bool'], ['bool']]),
    NodeData('not', 'bool', 1, min_depth=2, child_type=[['bool']]),
    # Conditional
    NodeData('if', 'cond', 3, min_depth=2, child_type=[None, ['bool'], None]),
]

def get_function_node(label: str) -> NodeData:
    """Return the NodeData for a function label"""
    for node in function_lib:
        if node.label == label:
            return node
    raise ValueError(f'NodeData not found for label: {label}')

def get_nodes(types: List[str], depth=2,
              lib: List[NodeData]=function_lib) -> List[NodeData]:
    """Return all NodeDatas of given types for min_depth from a given lib

    Used by BaseGP with the lib of user-selected nodes, including terminals
    and constants"""
    return [node for node in lib
            if node.node_type in types and node.min_depth <= depth]
