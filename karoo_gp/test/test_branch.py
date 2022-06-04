import ast, json
from collections import defaultdict
from unittest.mock import MagicMock

import pytest
import numpy as np

from karoo_gp import Branch
from .util import dump_json


@pytest.fixture
def branch_default_kwargs(rng, functions, terminals):
    return dict(
        rng=rng,
        functions=functions,
        terminals=terminals,
    )

@pytest.mark.parametrize('tree_type', ['f', 'g'])
@pytest.mark.parametrize('tree_depth_base', [1, 5])
@pytest.mark.parametrize('method', ['BFS', 'DFS'])
def test_branch(branch_default_kwargs, paths, tree_type, tree_depth_base,
                      method):
    """Test all attributes and methods of the branch in different scenarios

    Compile all attributes and method results into a dict (branch_output), and
    compare each item with the reference .json file.
    """
    kwargs = dict(**branch_default_kwargs,
                  tree_type=tree_type,
                  depth=tree_depth_base,
                  method=method)
    # random_seed = abs(hash(f'{tree_type}{tree_depth_base}{method}'))
    branch = Branch.generate(**kwargs)

    # Query attributes/methods
    random_int = int(kwargs['rng'].integers(1000))
    branch_output = dict(
        node_type=str(type(branch.node)),
        tree_type=branch.tree_type,
        parent=branch.parent,
        children=str(branch.children),
        repr=str(branch),
        parse=branch.parse(),
        save=branch.save(),
        display=branch.display(),
        depth=branch.depth,
        height=branch.height,
        n_children=branch.n_children,
        n_cols=branch.n_cols,
    )

    if branch.depth > 1:
        second_child = branch.get_child(2, method)
        expected_height = {'BFS': 1, 'DFS': 2}[method]
        assert second_child.height == expected_height
        branch_output['second_child'] = second_child.save()

    new_child_branch = Branch.load('((a)+(b))', tree_type)
    branch.set_child(2, new_child_branch, method)
    assert branch.get_child(2, method).save() == new_child_branch.save()
    branch_output['set_child'] = branch.save()

    if branch.depth > 1:
        branch.prune(kwargs['rng'], kwargs['terminals'])
        assert branch.depth == 1
        branch_output['prune'] = branch.save()

    # Load reference and compare
    fname = paths.test_data / (f'branch_ref[{tree_type}-{tree_depth_base}-'
                               f'{method}]')
    # dump_json(branch_output, paths.test_data / f'{fname}.json')
    with open(paths.test_data / f'{fname}.json') as f:
        ref = json.load(f)
    for k, v in ref.items():
        assert v == branch_output[k], f'Non-matching value for "{k}"'

