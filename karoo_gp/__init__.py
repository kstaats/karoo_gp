from .pause import pause
from .terminals import Terminal, Terminals
from .functions import Function, Functions
from .branch import Branch
from .tree import Tree
from .engine import Engine, NumpyEngine, TensorflowEngine
from .population import Population
from .base_class import Base_GP, Regressor_GP, MultiClassifier_GP, \
                        Matching_GP, LabelEncoder

__version__ = "3.0.0"
