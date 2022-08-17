from dataclasses import dataclass
from typing import List, Iterable

@dataclass
class NodeData:
    label: str                # The string representation used in expression
    node_type: str            # terminal, constant, operator, bool, or cond
    arity: int = 0            # Number of children
    min_depth: int = 0        # Depth (additional) required to be coherent
    child_type: list = None   # Eligible child node types (optional)

function_lib = [
    # Operators
    NodeData('+', 'operator', 2, min_depth=1),
    NodeData('-', 'operator', 2, min_depth=1),
    NodeData('*', 'operator', 2, min_depth=1),
    NodeData('/', 'operator', 2, min_depth=1),
    NodeData('**', 'operator', 2, min_depth=1),
    NodeData('abs', 'operator', 1, min_depth=1),
    NodeData('square', 'operator', 1, min_depth=1),
    NodeData('sqrt', 'operator', 1, min_depth=1),
    # NodeData('log', 'operator', 1, min_depth=1),  # TODO: Sometimes cause nans
    # NodeData('log1p', 'operator', 1, min_depth=1),
    # NodeData('cos', 'operator', 1, min_depth=1),
    # NodeData('sin', 'operator', 1, min_depth=1),
    # NodeData('tan', 'operator', 1, min_depth=1),
    # NodeData('arccos', 'operator', 1, min_depth=1),
    # NodeData('arcsin', 'operator', 1, min_depth=1),
    # NodeData('arctan', 'operator', 1, min_depth=1),
    # Boolean
    NodeData('==', 'bool', 2, min_depth=1),
    NodeData('!=', 'bool', 2, min_depth=1),
    NodeData('<', 'bool', 2, min_depth=1),
    NodeData('<=', 'bool', 2, min_depth=1),
    NodeData('>', 'bool', 2, min_depth=1),
    NodeData('>=', 'bool', 2, min_depth=1),
    NodeData('and', 'bool', 2, min_depth=2, child_type=[('bool'), ('bool')]),
    NodeData('or', 'bool', 2, min_depth=2, child_type=[('bool'), ('bool')]),
    NodeData('not', 'bool', 1, min_depth=2, child_type=[('bool')]),
    # Conditional
    NodeData('if', 'cond', 3, min_depth=2, child_type=[None, ('bool'), None]),
]

def get_function_node(label: str) -> NodeData:
    """Return the NodeData for a function label"""
    for node in function_lib:
        if node.label == label:
            return node
    raise ValueError(f'NodeData not found for label: {label}')

def get_nodes(types: Iterable[str], depth=2,
              lib: List[NodeData]=function_lib) -> List[NodeData]:
    """Return all NodeDatas of given types for min_depth from a given lib

    Used by BaseGP with the lib of user-selected nodes, including terminals
    and constants"""
    return [node for node in lib
            if node.node_type in types and node.min_depth <= depth]
