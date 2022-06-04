import sys, hashlib

import pytest
from unittest.mock import MagicMock
import numpy as np

from karoo_gp import Tree
from .util import hasher

@pytest.fixture
def default_args(functions, terminals, rng):
    return dict(
        log=MagicMock(),
        pause=MagicMock(),
        error=MagicMock(),
        id=1,
        tree_type='f',
        tree_depth_base=3,
        tree_depth_max=5,
        functions=functions,
        terminals=terminals,
        rng=rng,
    )

@pytest.mark.parametrize('tree_type', ['f', 'g'])
@pytest.mark.parametrize('tree_depth_base', [3, 5])
def test_tree_generate(default_args, tree_type, tree_depth_base):
    kwargs = dict(default_args)
    kwargs['tree_type'] = tree_type
    kwargs['tree_depth_base'] = tree_depth_base
    tree = Tree.generate(**kwargs)
    expected = {
        ('f', 3): '(a)*(b)*(b)/(a)+(a)-(a)/(a)+(a)',
        ('f', 5): ('(a)/(a)+(a)+(b)*(b)*(b)*(b)+(a)*(b)/(a)*(b)/(b)/(a)/(b)-'
                   '(b)*(b)+(a)+(a)+(a)/(b)-(a)+(b)-(a)*(b)/(a)/(a)-(b)+(a)+'
                   '(b)*(a)-(a)+(a)'),
        ('g', 3): '(b)+(b)',
        ('g', 5): '(b)+(b)',
    }
    assert tree.raw_expression == expected[(tree_type, tree_depth_base)]

@pytest.fixture
def tree(default_args):
    return Tree.generate(**default_args)

def test_tree_class(capsys, default_args, tree):
    # Attributes
    assert tree.id == default_args['id']
    assert tree.tree_type == default_args['tree_type']

    # Display Methods
    assert tree.raw_expression == '(a)*(b)*(b)/(a)+(a)-(a)/(a)+(a)'
    assert tree.expression == '2*a + b**2 - 1'

    # Manipulate Methods
    copied = tree.copy(id=tree.id+1)
    assert copied.id == tree.id + 1
    assert copied.pop_tree_type == tree.pop_tree_type
    assert copied.tree_depth_max == tree.tree_depth_max
    assert str(copied.root) == str(tree.root)
