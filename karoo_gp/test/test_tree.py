import sys, hashlib

import pytest
from unittest.mock import MagicMock
import numpy as np
import json

from karoo_gp import Tree, Functions, Terminals, Branch
from .util import hasher, dump_json

@pytest.fixture
def tree_default_kwargs(rng, functions, terminals):
    return dict(
        rng=rng,
        log=MagicMock(),
        pause=MagicMock(),
        error=MagicMock(),
        id=1,
        functions=functions,
        terminals=terminals,
    )

@pytest.mark.parametrize('tree_type', ['f', 'g'])
@pytest.mark.parametrize('tree_depth_base', [3, 5])
@pytest.mark.parametrize('tree_depth_max', [5])
def test_tree(tree_default_kwargs, paths, tree_type, tree_depth_base,
              tree_depth_max):
    """Test all attributes and methods of the tree in different scenarios

    Compile all attributes and method results into a dict (tree_output), and
    compare each item with the reference .json file.
    """
    kwargs = dict(**tree_default_kwargs,
                  tree_type=tree_type,
                  tree_depth_base=tree_depth_base)
    tree = Tree.generate(**kwargs)
    tree.result = dict(fitness=100)  # Add manually to test fitness property

    # Query attributes/methods
    tree_output = dict(
        id=tree.id,
        root_type=str(type(tree.root)),
        tree_type=tree.tree_type,
        repr=str(tree),
        raw_expression=tree.raw_expression,
        expression=tree.expression,
        save=tree.save(),
        display=tree.display(),
        depth=tree.depth,
        fitness = tree.fitness,
        n_children=tree.n_children,
        get_child=tree.get_child(1).save(),
    )

    # Set child (replace a specific subtree with provided branch)
    new_branch = Branch.load('((a)+(b))', tree_type)
    tree.set_child(1, new_branch)
    assert tree.get_child(1).save() == f'((a)+(b))'
    tree_output['set_child'] = tree.save()

    # Point Mutate (randomly change one function or terminal)
    tree.point_mutate(kwargs['rng'], kwargs['functions'], kwargs['terminals'],
                      kwargs['log'])
    point_mutated = tree.save()
    tree_output['point_mutate'] = point_mutated

    # Branch Mutate (randomly modify an entire subtree)
    tree.branch_mutate(kwargs['rng'], kwargs['functions'], kwargs['terminals'],
                       tree_depth_max, kwargs['log'])
    branch_mutated = tree.save()
    tree_output['branch_mutate'] = branch_mutated

    # Prune (remove branches beyond a given depth)
    if tree.depth > 1:
        original_depth = tree.depth  # e.g. 4
        prune_depth = original_depth - 1
        tree.prune(kwargs['rng'], kwargs['terminals'], prune_depth)
        assert tree.depth == prune_depth
        tree_output['prune'] = tree.save()

    # Crossover (set_child + prune)
    # Copy the tree, and insert the copy's node 0 (i.e. root) into the tree
    # at node 1. This should increase the depth of the tree by 1, unless its
    # over the max, in which case prune.
    depth_before_crossover = tree.depth - 1
    crossover_mate = tree.copy()
    branch_to_insert = crossover_mate.get_child(0).save()
    tree.crossover(1, crossover_mate, 0, kwargs['rng'], kwargs['terminals'],
                   tree_depth_max, kwargs['log'], kwargs['pause'])
    assert tree.get_child(1).save() == branch_to_insert
    assert tree.depth - 1 == min(depth_before_crossover + 1, tree_depth_max)
    tree_output['crossover'] = tree.save()

    # Load reference and compare
    fname = paths.test_data / f'tree_ref[{tree_type}-{tree_depth_base}]'
    # dump_json(tree_output, paths.test_data / f'{fname}.json')
    with open(paths.test_data / f'{fname}.json') as f:
        ref = json.load(f)
    for k, v in ref.items():
        assert v == tree_output[k], f'Non-matching value for "{k}"'

# # ---------------------------------------------------------
# # LEGACY TESTS (REPLACE AFTER CONFIRMING ABOVE IS ACCURATE)

@pytest.fixture
def default_args(rng):
    return dict(
        log=MagicMock(),
        pause=MagicMock(),
        error=MagicMock(),
        id=1,
        tree_type='f',
        tree_depth_base=3,
        functions=Functions(['+', '-', '*', '/']),
        terminals=Terminals(['a', 'b']),
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
    copy = tree.copy()
    assert copy.raw_expression == expected[(tree_type, tree_depth_base)]
    saved = tree.save()
    loaded = Tree.load(kwargs['id'], saved)
    assert loaded.raw_expression == expected[(tree_type, tree_depth_base)]

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
    assert copied.tree_type == tree.tree_type
    assert str(copied.root) == str(tree.root)

    # Save and Load
    saved = tree.save()
    loaded = Tree.load(tree.id, saved)
    assert loaded.expression == tree.expression
