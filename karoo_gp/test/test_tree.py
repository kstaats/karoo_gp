import sys, hashlib

import pytest
import numpy as np

from karoo_gp import Tree
from .util import hasher

@pytest.fixture
def default_args(mock_func, rng):
    return dict(
        log=mock_func,
        pause=mock_func,
        error=mock_func,
        id=1,
        tree_type='f',
        tree_depth_base=3,
        tree_depth_max=5,
        functions=np.array([['+', 2], ['-', 2], ['*', 2], ['/', 2]]),
        terminals=['a', 'b', 'c'],
        rng=rng
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
        ('f', 5): '(a)/(a)+(a)+(b)*(b)*(b)*(b)+(a)*(b)/(a)*(b)/(b)/(a)/(b)-(b)*(b)+(a)+(a)+(a)/(b)-(a)+(b)-(a)*(b)/(a)/(a)-(b)+(a)+(b)*(a)-(a)+(a)',
        ('g', 3): '(b)+(b)',
        ('g', 5): '(b)+(b)',
    }
    assert tree.parse() == expected[(tree_type, tree_depth_base)]

@pytest.fixture
def tree(default_args):
    return Tree.generate(**default_args)

def test_tree_class(capsys, default_args, tree):
    # Attributes
    assert tree.id == default_args['id']
    assert tree.pop_tree_type == default_args['tree_type']
    assert tree.tree_depth_max == default_args['tree_depth_max']
    assert hasher(str(tree.root)) == '258917d2dacbc5aed1f7d7e20b2f63a7'

    # Display Methods
    assert tree.parse() == '(a)*(b)*(b)/(a)+(a)-(a)/(a)+(a)'
    assert str(tree.sym()) == '2*a + b**2 - 1'
    assert tree.fitness() == -1
    tree.display()
    captured = capsys.readouterr()
    output = captured.out
    print(output)
    assert hasher(output) == '16ec6d47fb8437109c6ecf337b9bf69a'

    # Manipulate Methods
    copied = tree.copy(id=tree.id+1)
    assert copied.id == tree.id + 1
    assert copied.pop_tree_type == tree.pop_tree_type
    assert copied.tree_depth_max == tree.tree_depth_max
    assert str(copied.root) == str(tree.root)
