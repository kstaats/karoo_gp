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
import numpy as np
import tensorflow as tf

def placeholder(*args, **kwargs):
    raise NotImplementedError()

@dataclass
class NodeData:
    """Include all attributes required for a Node"""
    label: str                # The string representation used in expression
    node_type: str            # terminal, constant, operator, bool, or cond
    arity: int = 0            # Number of children
    min_depth: int = 0        # Depth (additional) required to be coherent
    child_type: list = None   # Eligible child node types (optional)
    numpy_func: callable = placeholder
    tensorflow_func: callable = placeholder

    def __repr__(self):
        return f'<NodeData label={self.label!r} type={self.node_type}>'

# Custom Function definitions to avoid nan/inf values:
def np_safe_divide(a: np.ndarray, b: np.ndarray):
    """If dividing by 0, return 0"""
    nonzero = b != 0
    c = np.zeros(a.shape)
    c[nonzero] = a[nonzero] / b[nonzero]
    return c

def np_safe_sqrt(a: np.ndarray):
    """For sqrt(-a), return -sqrt(abs(a))"""
    negative = np.less(a, 0)
    absolute = np.abs(a)
    square_root = np.sqrt(absolute)
    square_root[negative] *= -1
    return square_root

def tf_safe_sqrt(a):
    """For sqrt(-a), return -sqrt(abs(a))"""
    negative = tf.less(a, 0)
    absolute = tf.abs(a)
    square_root = tf.sqrt(absolute)
    square_root[negative] *= -1
    return square_root

# Function node definitions
numeric = ['terminal', 'constant', 'operator', 'cond']
function_lib = [
    # Operators
    NodeData('+', 'operator', 2, min_depth=1, child_type=[numeric, numeric],
             numpy_func=np.add, tensorflow_func=tf.add),
    NodeData('-', 'operator', 2, min_depth=1, child_type=[numeric, numeric],
             numpy_func=np.subtract, tensorflow_func=tf.subtract),
    NodeData('*', 'operator', 2, min_depth=1, child_type=[numeric, numeric],
             numpy_func=np.multiply, tensorflow_func=tf.multiply),
    NodeData('/', 'operator', 2, min_depth=1, child_type=[numeric, numeric],
             numpy_func=np_safe_divide, tensorflow_func=tf.divide),
    NodeData('**', 'operator', 2, min_depth=1, child_type=[numeric, numeric],
             numpy_func=np.float_power, tensorflow_func=tf.pow),
    NodeData('abs', 'operator', 1, min_depth=1, child_type=[numeric],
             numpy_func=np.abs, tensorflow_func=tf.abs),
    NodeData('square', 'operator', 1, min_depth=1, child_type=[numeric],
             numpy_func=np.square, tensorflow_func=tf.square),
    NodeData('sqrt', 'operator', 1, min_depth=1, child_type=[numeric],
             numpy_func=np_safe_sqrt, tensorflow_func=tf_safe_sqrt),
    # NodeData('log', 'operator', 1, min_depth=1),  # TODO: Sometimes cause nans
    # NodeData('log1p', 'operator', 1, min_depth=1),
    # NodeData('cos', 'operator', 1, min_depth=1),
    # NodeData('sin', 'operator', 1, min_depth=1),
    # NodeData('tan', 'operator', 1, min_depth=1),
    # NodeData('arccos', 'operator', 1, min_depth=1),
    # NodeData('arcsin', 'operator', 1, min_depth=1),
    # NodeData('arctan', 'operator', 1, min_depth=1),
    # Boolean
    NodeData('==', 'bool', 2, min_depth=1, child_type=[numeric, numeric],
             numpy_func=np.equal, tensorflow_func=tf.equal),
    NodeData('!=', 'bool', 2, min_depth=1, child_type=[numeric, numeric],
             numpy_func=np.not_equal, tensorflow_func=tf.not_equal),
    NodeData('<', 'bool', 2, min_depth=1, child_type=[numeric, numeric],
             numpy_func=np.less, tensorflow_func=tf.less),
    NodeData('<=', 'bool', 2, min_depth=1, child_type=[numeric, numeric],
             numpy_func=np.less_equal, tensorflow_func=tf.less_equal),
    NodeData('>', 'bool', 2, min_depth=1, child_type=[numeric, numeric],
             numpy_func=np.greater, tensorflow_func=tf.greater),
    NodeData('>=', 'bool', 2, min_depth=1, child_type=[numeric, numeric],
             numpy_func=np.greater_equal, tensorflow_func=tf.greater_equal),
    NodeData('and', 'bool', 2, min_depth=2, child_type=[['bool'], ['bool']],
             numpy_func=np.logical_and, tensorflow_func=tf.logical_and),
    NodeData('or', 'bool', 2, min_depth=2, child_type=[['bool'], ['bool']],
             numpy_func=np.logical_or, tensorflow_func=tf.logical_or),
    NodeData('not', 'bool', 1, min_depth=2, child_type=[['bool']],
             numpy_func=np.logical_not, tensorflow_func=tf.logical_not),
    # Conditional
    NodeData('if', 'cond', 3, min_depth=2, child_type=[numeric, ['bool'], numeric],
             numpy_func=np.where, tensorflow_func=tf.where),
]

def get_function_node(label: str) -> NodeData:
    """Return the NodeData for a function label"""
    for node in function_lib:
        if node.label == label:
            return node
    raise ValueError(f'NodeData not found for label: {label}')

def get_nodes(types: List[str]=None, depth=2, arity=None,
              lib: List[NodeData]=function_lib) -> List[NodeData]:
    """Return all NodeDatas of given types for min_depth from a given lib

    Used by BaseGP with the lib of user-selected nodes, including terminals
    and constants"""
    return [node for node in lib if all((
        node.min_depth <= depth,
        True if types is None else node.node_type in types,
        True if arity is None else node.arity == arity))]
