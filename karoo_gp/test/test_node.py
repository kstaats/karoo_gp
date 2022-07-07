import json

import pytest
import numpy as np

from karoo_gp import Node, get_nodes
from .util import dump_json

@pytest.fixture
def node_default_kwargs(rng, node_lib):
    def get_nodes_(*args, **kwargs):
        return get_nodes(*args, **kwargs, lib=node_lib)
    return dict(
        rng=rng,
        get_nodes=get_nodes_
    )

@pytest.mark.parametrize('tree_type', ['f', 'g'])
@pytest.mark.parametrize('tree_depth_base', [1, 5])
@pytest.mark.parametrize('method', ['BFS', 'DFS'])
def test_node(node_default_kwargs, paths, tree_type, tree_depth_base,
                      method):
    """Test all attributes and methods of the node in different scenarios

    Compile all attributes and method results into a dict (node_output), and
    compare each item with the reference .json file.
    """
    kwargs = dict(**node_default_kwargs,
                  tree_type=tree_type,
                  depth=tree_depth_base,
                  method=method)
    node = Node.generate(**kwargs)

    # Query attributes/methods
    random_int = int(kwargs['rng'].randint(1000))
    node_output = dict(
        node_type=node.node_type,
        tree_type=node.tree_type,
        parent=node.parent,
        children=str(node.children),
        repr=str(node),
        parse=node.parse(),
        display=node.display(),
        depth=node.depth,
        height=node.height,
        n_children=node.n_children,
        n_cols=node.n_cols,
    )

    if node.depth > 1:
        second_child = node.get_child(2, method)
        expected_height = {'BFS': 1, 'DFS': 2}[method]
        assert second_child.height == expected_height
        node_output['second_child'] = second_child.parse()

    new_child_node = Node.load('((a)+(b))', tree_type)
    node.set_child(2, new_child_node, method)
    assert node.get_child(2, method).parse() == new_child_node.parse()
    node_output['set_child'] = node.parse()

    if node.depth > 1:
        node.prune(kwargs['rng'], kwargs['get_nodes'])
        assert node.depth == 1
        node_output['prune'] = node.parse()

    # Load reference and compare
    fname = paths.test_data / (f'node_ref[{tree_type}-{tree_depth_base}-'
                               f'{method}]')
    # dump_json(node_output, paths.test_data / f'{fname}.json')
    with open(paths.test_data / f'{fname}.json') as f:
        ref = json.load(f)
    for k, v in ref.items():
        assert v == node_output[k], f'Non-matching value for "{k}"'

