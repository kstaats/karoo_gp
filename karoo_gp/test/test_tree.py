import pytest
from unittest.mock import MagicMock
import json

from karoo_gp import Tree, Node, get_nodes
from .util import dump_json

@pytest.fixture
def tree_default_kwargs(rng, node_lib):
    def get_nodes_(*args, **kwargs):
        return get_nodes(*args, **kwargs, lib=node_lib)
    return dict(
        rng=rng,
        id=1,
        get_nodes=get_nodes_,
        force_types=[('operator', 'cond')]
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
    log = MagicMock()
    pause = MagicMock()
    tree = Tree.generate(**kwargs)
    tree.score = dict(fitness=100)  # Add manually to test fitness property

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
        get_child=tree.get_child(1).parse(),
    )

    # Set child (replace a specific subtree with provided node)
    new_node = Node.load('((a)+(b))', tree_type)
    tree.set_child(1, new_node)
    assert tree.get_child(1).parse() == f'((a)+(b))'
    tree_output['set_child'] = tree.save()

    # Point Mutate (randomly change one function or terminal)
    tree.point_mutate(kwargs['rng'], kwargs['get_nodes'], log)
    point_mutated = tree.save()
    tree_output['point_mutate'] = point_mutated

    # Node Mutate (randomly modify an entire subtree)
    tree.branch_mutate(kwargs['rng'], kwargs['get_nodes'], kwargs['force_types'],
                       tree_depth_max, log)
    branch_mutated = tree.save()
    tree_output['branch_mutate'] = branch_mutated

    # Prune (remove nodees beyond a given depth)
    if tree.depth > 1:
        original_depth = tree.depth  # e.g. 4
        prune_depth = original_depth - 1
        tree.prune(kwargs['rng'], kwargs['get_nodes'], prune_depth)
        assert tree.depth == prune_depth
        tree_output['prune'] = tree.save()

    # Crossover (set_child + prune)
    # Copy the tree, and insert the copy's node 0 (i.e. root) into the tree
    # at node 1. This should increase the depth of the tree by 1, unless its
    # over the max, in which case prune.
    depth_before_crossover = tree.depth - 1
    crossover_mate = tree.copy()
    node_to_insert = crossover_mate.get_child(0).parse()
    tree.crossover(1, crossover_mate, 0, kwargs['rng'], kwargs['get_nodes'],
                   tree_depth_max, log, pause)
    assert tree.get_child(1).parse() == node_to_insert
    assert tree.depth - 1 == min(depth_before_crossover + 1, tree_depth_max)
    tree_output['crossover'] = tree.save()

    # Load reference and compare
    fname = paths.test_data / f'tree_ref[{tree_type}-{tree_depth_base}]'
    # dump_json(tree_output, paths.test_data / f'{fname}.json')
    with open(paths.test_data / f'{fname}.json') as f:
        ref = json.load(f)
    for k, v in ref.items():
        assert v == tree_output[k], f'Non-matching value for "{k}"'
