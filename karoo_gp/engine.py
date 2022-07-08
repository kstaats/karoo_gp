import os
import ast
from abc import ABC, abstractmethod

import numpy as np

from . import Tree
from .util import LazyLoader
# Tensorflow takes ~2 seconds to load, so only load when used
tf = LazyLoader('tf', globals(), 'tensorflow.compat.v1')


class Engine(ABC):
    """Calculate the output of a batch of data for a batch of trees"""
    def __init__(self, model, engine_type='default'):
        self.model = model
        self.engine_type = engine_type
        self.dtype = np.float64
        self.num_obj = np.full

    def __repr__(self):
        return f"<Engine: {self.engine_type}>"

    @abstractmethod
    def predict(self, trees, X, X_hash=None):
        """Takes a list of Tree objects and a numpy array of data

        * Operate on a *list* of trees, so X_dict is only compiled once,
          and return a *list* of predictions
        * If X_hash is provided and tree.expression is in cache[X_hash],
          skip the prediction for that tree and return 0; score will be
          copied from the cache later by the .score() method.
        """
        return NotImplementedError(f'Engine.predict() not implemented for type '
                                   f'{self.engine_type}')

    def _type_check(self, trees):
        if not all(isinstance(tree, Tree) for tree in trees):
            raise TypeError(f"trees must be a sequence of Trees")

    def parse_expr(self, expr, X_dict, shape):
        """Parse an expr into a function and insert sample X_dict"""
        tree = ast.parse(expr, mode='eval').body
        return self.parse_node(tree, X_dict, shape)

    def parse_node(self, node, X_dict, shape):
        """Recursively build a function from ast tree and sample X_dict"""
        if isinstance(node, ast.IfExp):
            return self.operators['if'](
                self.parse_node(node.test, X_dict, shape),
                self.parse_node(node.body, X_dict, shape),
                self.parse_node(node.orelse, X_dict, shape))
        if isinstance(node, ast.Name):
            return X_dict[node.id]
        elif isinstance(node, ast.Num):
            return self.num_obj(shape, node.n)
        elif isinstance(node, ast.UnaryOp):
            return self.operators[type(node.op)](
                self.parse_node(node.operand, X_dict, shape))
        elif isinstance(node, ast.BinOp):
            return self.operators[type(node.op)](
                self.parse_node(node.left, X_dict, shape),
                self.parse_node(node.right, X_dict, shape))
        elif isinstance(node, ast.BoolOp):
            return self.operators[type(node.op)](
                *[self.parse_node(v, X_dict, shape) for v in node.values])
        elif isinstance(node, ast.Compare):
            return self.operators[type(node.ops[0])](
                self.parse_node(node.left, X_dict, shape),
                self.parse_node(node.comparators[0], X_dict, shape))
        elif isinstance(node, ast.Call):
            return self.operators[node.func.id](
                self.parse_node(node.args[0], X_dict, shape))


#++++++++++++++++++++++++++++
#   Numpy                   |
#++++++++++++++++++++++++++++

def safe_divide(a, b):
    """If dividing by 0, return 0"""
    return np.where(b==0, 0, a/b)

def safe_sqrt(a):
    """For sqrt(-a), return -sqrt(abs(a))"""
    negative = np.less(a, 0)
    absolute = np.abs(a)
    square_root = np.sqrt(absolute)
    square_root[negative] *= -1
    return square_root

class NumpyEngine(Engine):
    def __init__(self, model):
        super().__init__(model, engine_type='numpy')
        self.dtype = np.float64
        def num_obj(shape, value):  # Function for terminals/constants arrays
            return np.full(shape, value, dtype=self.dtype)
        self.num_obj = num_obj

        self.operators = {
            ast.Add: np.add,
            ast.Sub: np.subtract,
            ast.Mult: np.multiply,
            ast.Div: safe_divide,
            ast.Pow: np.float_power,
            ast.USub: np.negative,
            'if': np.where,
            ast.And: np.logical_and,
            ast.Or: np.logical_or,
            ast.Not: np.logical_not,
            ast.Eq: np.equal,
            ast.NotEq: np.not_equal,
            ast.Lt: np.less,
            ast.LtE: np.less_equal,
            ast.Gt: np.greater,
            ast.GtE: np.greater_equal,
            'abs': np.abs,
            'sign': np.sign,
            'square': np.square,
            'sqrt': safe_sqrt,
            'log': np.log,
            'log1p': np.log1p,
            'cos': np.cos,
            'sin': np.sin,
            'tan': np.tan,
            'arccos': np.arccos,
            'arcsin': np.arcsin,
            'arctan': np.arctan,
        }

    def predict(self, trees, X, X_hash=None):
        """Return predicted the output of each sample for a list of trees"""
        # Check type
        self._type_check(trees)

        # Sort sample columns by terminal
        variables = [t.label for t in self.model.get_nodes(('terminal'))]
        X_dict = {name: X[:, i] for i, name in enumerate(variables)}
        shape = X.shape[0]

        # Return the output of each tree for sample data
        predictions = np.zeros((len(trees), X.shape[0]), dtype=self.dtype)
        for i, tree in enumerate(trees):
            expr = tree.expression

            # Skip tree if cached score for X_hash and expr
            if (X_hash and expr in self.model.cache_[X_hash]):
                continue
            predictions[i] = self.parse_expr(expr, X_dict, shape)
        return predictions

#++++++++++++++++++++++++++++
#   TensorFlow              |
#++++++++++++++++++++++++++++


### TensorFlow Imports and Definitions ###
if os.environ.get("TF_CPP_MIN_LOG_LEVEL") is None:
    # Set the log level, unless it's already set in the env.
    # This allows users to override this value with an env var.
    # 0 = all messages are logged (default behavior)
    # 1 = INFO messages are not printed
    # 2 = INFO and WARNING messages are not printed
    # 3 = INFO, WARNING, and ERROR messages are not printed
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # only print ERRORs

class TensorflowEngine(Engine):

    def __init__(self, model, tf_device="/gpu:0", tf_device_log=False):
        super().__init__(model, engine_type='tensorflow')
        self.dtype = tf.float64
        def num_obj(shape, value):
            # Tensorflow has a different ordering
            return tf.constant(value, dtype=self.dtype, shape=shape)
        self.num_obj = num_obj

        # Configure tensorflow
        seed = (model.random_state if isinstance(model.random_state, int)
                else 1000)
        tf.set_random_seed(seed)
        tf.disable_v2_behavior()
        self.tf_device = tf_device
        self.config = tf.ConfigProto(
            log_device_placement=tf_device_log,
            allow_soft_placement=True)
        self.config.gpu_options.allow_growth = True

        def safe_sqrt(a):
            """For sqrt(-a), return -sqrt(abs(a))"""
            negative = tf.less(a, 0)
            absolute = tf.abs(a)
            square_root = tf.sqrt(absolute)
            square_root[negative] *= -1
            return square_root

        # Reference for parse_node
        self.operators = {
            ast.Add: tf.add,
            ast.Sub: tf.subtract,
            ast.Mult: tf.multiply,
            ast.Div: tf.divide,
            ast.Pow: tf.pow,
            ast.USub: tf.negative,
            'if': tf.where,
            ast.And: tf.logical_and,
            ast.Or: tf.logical_or,
            ast.Not: tf.logical_not,
            ast.Eq: tf.equal,
            ast.NotEq: tf.not_equal,
            ast.Lt: tf.less,
            ast.LtE: tf.less_equal,
            ast.Gt: tf.greater,
            ast.GtE: tf.greater_equal,
            'abs': tf.abs,
            'sign': tf.sign,
            'square': tf.square,
            'sqrt': safe_sqrt,
            'log': tf.log,
            'log1p': tf.log1p,
            'cos': tf.cos,
            'sin': tf.sin,
            'tan': tf.tan,
            'arccos': tf.acos,
            'arcsin': tf.asin,
            'arctan': tf.atan,

        }

    def predict(self, trees, X, X_hash=None):
        """Return the predicted output of each sample for a list of trees"""
        # Check type
        self._type_check(trees)

        shape = (X.shape[0], )
        predictions = np.zeros((len(trees), X.shape[0]), dtype=np.float64)
        for i, tree in enumerate(trees):

            # Skip tree if cached score for X_hash and expr
            expr = tree.expression
            if (X_hash and expr in self.model.cache_[X_hash]):
                continue

            tf.reset_default_graph()
            with tf.Session(config=self.config) as sess:
                with sess.graph.device(self.tf_device):

                    # Sort sample columns by terminal
                    variables = [t.label for t in self.model.get_nodes(('terminal'))]
                    X_dict = {v: tf.constant(X[:, i], dtype=self.dtype)
                            for i, v in enumerate(variables)}

                    # Return the output of each tree for sample data
                    pred = self.parse_expr(tree.expression, X_dict, shape)
                    predictions[i] = sess.run([pred])[0]
        return predictions
