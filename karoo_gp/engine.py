class Engine:
    """Calculate the output of a batch of data for on a batch of trees.

    * Takes a list of Tree objects and a numpy array of data
    * Operates on a *list* of trees, so X_dict is only compiled once
    """
    def __init__(self, model, engine_type='default'):
        self.model = model
        self.engine_type = engine_type

    def __repr__(self):
        return f"<Engine: {self.engine_type}>"

#++++++++++++++++++++++++++++
#   Numpy                   |
#++++++++++++++++++++++++++++
import ast
import operator as op

import numpy as np

from karoo_gp import Tree

def inf_to_zero_divide(a, b):
    return np.where(b==0, 0, a / b)

class NumpyEngine(Engine):
    def __init__(self, model):
        super().__init__(model, engine_type='numpy')
        self.dtype = np.float64
        self.operators = {
            ast.Add: op.add,
            ast.Sub: op.sub,
            ast.Mult: op.mul,
            ast.Div: inf_to_zero_divide,
            ast.Pow: op.pow,
            ast.USub: op.neg,
        }

    def predict(self, trees, X, X_hash=None):
        """Return predicted the output of each sample for a list of trees"""
        # Check type
        if type(trees) == Tree:
            trees = [trees]
        elif type(trees) != list or type(trees[0]) != Tree:
            raise TypeError(f"trees must be a Tree or list of Trees")

        # Sort sample columns by terminal
        variables = self.model.terminals.variables.keys()
        X_dict = {name: X[:, i] for i, name in enumerate(variables)}

        # Return the output of each tree for sample data
        predictions = np.zeros((len(trees), X.shape[0]), dtype=self.dtype)
        for i, tree in enumerate(trees):
            expr = tree.expression

            # Skip tree if cached score for X_hash and expr
            if (X_hash and expr in self.model.cache[X_hash]):
                continue
            predictions[i] = self.parse_expr(expr, X_dict)
        return predictions

    def parse_expr(self, expr, X_dict):
        """Parse an expr into a numpy function and insert sample X_dict"""
        tree = ast.parse(expr, mode='eval').body
        shape = next(iter(X_dict.values())).shape
        return self.parse_node(tree, X_dict, shape)

    def parse_node(self, node, X_dict, shape):
        """Recursively build a numpy graph from ast tree and sample X_dict"""
        if isinstance(node, ast.Name):
            return X_dict[node.id]
        elif isinstance(node, ast.Num):
            return np.full(shape, node.n, dtype=self.dtype)
        elif isinstance(node, ast.UnaryOp):
            return self.operators[type(node.op)](
                self.parse_node(node.operand, X_dict, shape))
        elif isinstance(node, ast.BinOp):
            return self.operators[type(node.op)](
                self.parse_node(node.left, X_dict, shape),
                self.parse_node(node.right, X_dict, shape))

#++++++++++++++++++++++++++++
#   TensorFlow              |
#++++++++++++++++++++++++++++

import os

# Tensorflow takes a few seconds to load, so only load when used
from karoo_gp.util import LazyLoader
tf = LazyLoader('tf', globals(), 'tensorflow.compat.v1')

class TensorflowEngine(Engine):

    def __init__(self, model, tf_device="/gpu:0", tf_device_log=False):
        super().__init__(model, engine_type='tensorflow')
        self.dtype = tf.float64

        # Configure tensorflow
        ### TensorFlow Imports and Definitions ###
        tf.set_random_seed(model.seed)
        tf.disable_v2_behavior()
        self.tf_device = tf_device
        self.config = tf.ConfigProto(
            log_device_placement=tf_device_log,
            allow_soft_placement=True)
        self.config.gpu_options.allow_growth = True
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

        # Reference for parse_node
        self.operators = {
            ast.Add: tf.add,
            ast.Sub: tf.subtract,
            ast.Mult: tf.multiply,
            ast.Div: tf.divide,
            ast.Pow: tf.pow,
            ast.USub: tf.negative,
        }

    def predict(self, trees, X, X_hash=None):
        """Return the predicted output of each sample for a list of trees"""
        # Check type
        if type(trees) == Tree:
            trees = [trees]
        elif type(trees) != list or type(trees[0]) != Tree:
            raise TypeError(f"trees must be a Tree or list of Trees")

        predictions = np.zeros((len(trees), X.shape[0]), dtype=np.float64)
        for i, tree in enumerate(trees):

            # Skip tree if cached score for X_hash and expr
            expr = tree.expression
            if (X_hash and expr in self.model.cache[X_hash]):
                continue

            tf.reset_default_graph()
            with tf.Session(config=self.config) as sess:
                with sess.graph.device(self.tf_device):

                    # Sort sample columns by terminal
                    variables = self.model.terminals.variables.keys()
                    X_dict = {v: tf.constant(X[:, i], dtype=self.dtype)
                            for i, v in enumerate(variables)}

                    # Return the output of each tree for sample data
                    pred = self.parse_expr(tree.expression, X_dict)
                    predictions[i] = sess.run([pred])[0]
        return predictions

    def parse_expr(self, expr, X_dict):
        """Convert a string expression into a tensorflow graph"""
        tree = ast.parse(expr, mode='eval').body
        shape = next(iter(X_dict.values())).shape
        return self.parse_node(tree, X_dict, shape)

    def parse_node(self, node, X_dict, shape):
        """Recursively build a tensorflow graph of an ast node"""
        if isinstance(node, ast.Name):
            return X_dict[node.id]
        elif isinstance(node, ast.Num):
            return tf.constant(node.n, shape=shape, dtype=self.dtype)
        elif isinstance(node, ast.UnaryOp):
            return self.operators[type(node.op)](
                self.parse_node(node.operand, X_dict, shape))
        elif isinstance(node, ast.BinOp):
            return self.operators[type(node.op)](
                self.parse_node(node.left, X_dict, shape),
                self.parse_node(node.right, X_dict, shape))
