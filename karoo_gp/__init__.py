from .pause import pause
from .node_lib import NodeData, function_lib, get_nodes, get_function_node
from .node import Node
from .tree import Tree
from .population import Population
from .base_class import BaseGP, RegressorGP, MultiClassifierGP, \
                        MatchingGP, ClassDecoder

__version__ = "3.0.0"
