import json

import pytest
import numpy as np

from karoo_gp import Node, NodeData, get_nodes, function_lib
from .util import dump_json

@pytest.fixture
def node_default_kwargs(rng, nodes):
    nodes = ([NodeData(t, 'terminal') for t in ['a', 'b', 'c']] +
                [NodeData(c, 'constant') for c in [1, 2, 3]] +
                function_lib)
    def get_nodes_(*args, **kwargs):
        return get_nodes(*args, **kwargs, lib=nodes)
    return dict(
        rng=rng,
        get_nodes=get_nodes_,
        force_types=[['operator', 'cond']]
    )

@pytest.mark.parametrize('tree_type', ['f', 'g'])
@pytest.mark.parametrize('tree_depth_base', [1, 5])
def test_node(node_default_kwargs, paths, tree_type, tree_depth_base):
    """Test all attributes and methods of the node in different scenarios

    Compile all attributes and method results into a dict (node_output), and
    compare each item with the reference .json file.
    """
    kwargs = dict(**node_default_kwargs,
                  tree_type=tree_type,
                  depth=tree_depth_base)
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
        second_child = node.get_child(2)
        assert second_child.height == 2
        node_output['second_child'] = second_child.parse()

    new_child_node = Node.load('((a)+(b))', tree_type)
    node.set_child(2, new_child_node)
    assert node.get_child(2).parse() == new_child_node.parse()
    node_output['set_child'] = node.parse()

    if node.depth > 1:
        node.prune(kwargs['rng'], kwargs['get_nodes'])
        assert node.depth == 1
        node_output['prune'] = node.parse()

    # Load reference and compare
    fname = paths.test_data / (f'node_ref[{tree_type}-{tree_depth_base}]')
    # dump_json(node_output, paths.test_data / f'{fname}.json')
    with open(paths.test_data / f'{fname}.json') as f:
        ref = json.load(f)
    for k, v in ref.items():
        assert v == node_output[k], f'Non-matching value for "{k}"'

@pytest.mark.parametrize('force_types', ['cond', 'bool', 'condbool'])
@pytest.mark.parametrize('tree_type', ['f', 'g'])
def test_node_force_types(rng, force_types, tree_type):
    """Confirm that the force_types kwarg works correctly

    Secondarily, verify that 'cond' and 'bool' funcs work correctly.
    TODO: Add to test_node library"""

    # Generate trees using rules
    ft = dict(
        cond=[['cond']], bool=[['bool']], condbool=[['cond'], ['bool']]
    )[force_types]
    lib = ([NodeData(t, 'terminal') for t in ['a', 'b']] +
           [NodeData(c, 'constant') for c in [2, 3]] +
           function_lib)
    def get_nodes_(*args, **kwargs):
        return get_nodes(*args, **kwargs, lib=lib)
    for i in range(10):
        node = Node.generate(rng, get_nodes_, tree_type, depth=3,
                             force_types=ft)

        # Verify root node is generated correctly
        assert node.node_type in ft[0]
        if len(ft) == 1:
            continue

        # Verify deeper nodes are generated correctly
        i_node = 1
        for height in range(1, len(ft)):
            while True:
                child = node.get_child(i_node)
                i_node += 1
                if child.height > height:
                    break
                assert child.node_type in ft[height]

        # Verify that tree is parsed/simplified correctly
        raw = node.parse()
        expr = node.parse(simplified=True)
        assert len(raw) >= len(expr)
        for i_child in range(node.n_children):
            child = node.get_child(i_child)
            assert str(child.label) in raw
