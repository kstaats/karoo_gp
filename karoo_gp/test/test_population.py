import pytest
from unittest.mock import MagicMock
import numpy as np

from karoo_gp import Population, Terminals, Functions
from .util import load_data

# Create a dummy model with mock functions to confirm calls
class MockModel:
    def __init__(self):
        self.cache_ = {}
        self.log=MagicMock()
        self.pause=MagicMock()
        self.error=MagicMock()
        self.rng=np.random.RandomState(1000)
        self.build_fittest_dict=MagicMock()

    def batch_predict(self, X, trees, X_hash):
        """Return an array of 1's of expected shape"""
        output = np.ones((len(trees), X.shape[0]))
        return output[0] if len(trees) == 1 else output

    def calculate_score(self, y_pred, y_true):
        """Return expected dict with fitness = 1"""
        return {'fitness': 1}

    def fitness_compare(self, a, b):
        """Return the latter of two trees compared"""
        return b  # So that pop.fittest() returns pop.trees[-1]

@pytest.fixture
def default_pop_kwargs():
    return dict(
        model=MockModel(),
        tree_type='r',
        tree_depth_base=3,
        tree_pop_max=10,
        functions=Functions(['+', '-', '*', '/']),
        terminals=Terminals(['a', 'b']),
    )

@pytest.mark.parametrize('tree_type', ['f', 'g', 'r'])
def test_population_generate(default_pop_kwargs, tree_type):
    """Confirm that population types are generated correctly"""
    kwargs = {**default_pop_kwargs, 'tree_type': tree_type}
    population = Population.generate(**kwargs)
    assert len(population.trees) == kwargs['tree_pop_max']

    if tree_type == 'f':
        for t in population.trees:
            assert t.tree_type == 'f'
            assert t.depth == kwargs['tree_depth_base']  # All same depth
            n_children = sum(
                [2 ** b for b in range(1, kwargs['tree_depth_base'] + 1)])
            assert t.n_children == n_children  # All have max nodes for depth
    elif tree_type == 'g':
        depths = set()
        n_childrens = set()
        for t in population.trees:
            assert t.tree_type == 'g'
            depths.add(t.depth)
            n_childrens.add(t.n_children)
        assert len(depths) > 1  # There are different depths
        assert len(n_childrens) > 1  # There are different numbers of children
    elif tree_type == 'r':
        count = dict(f=0, g=0)
        depths = set()
        for t in population.trees:
            count[t.tree_type] += 1
            depths.add(t.depth)
        pop, depth = kwargs['tree_pop_max'], kwargs['tree_depth_base']
        n_cycles = pop // (2 * depth)
        n_extra = pop - n_cycles * (2 * depth)
        assert count['f'] == n_cycles * depth  # One 'full' tree at each depth
        assert count['g'] == count['f'] + n_extra  # That plus extras 'grow'
        assert len(depths) == kwargs['tree_depth_base']  # Trees at each depth

@pytest.fixture
def default_evolve_params():
    return dict(
        swim='p',
        tree_depth_min=3,
        tree_depth_max=5,
        evolve_repro=0.1,
        evolve_point=0.1,
        evolve_branch=0.2,
        evolve_cross=0.6,
        tourn_size=7,
    )

def test_population_class(tmp_path, paths, default_pop_kwargs,
                          default_evolve_params):
    dataset_params = load_data(tmp_path, paths, 'r')
    terminals = Terminals(dataset_params['terminals'])
    X, y = dataset_params['X'], dataset_params['y']

    pop_kwargs = dict(**default_pop_kwargs)
    pop_kwargs['terminals'] = terminals
    population = Population.generate(**pop_kwargs)
    assert population.gen_id == 1

    # Evaluate
    population.evaluate(X, y)
    assert population.evaluated == True  # Flag set
    highest_fitness = population.fittest().fitness
    for tree in population.trees:
        assert tree.fitness <= highest_fitness

    # Evolve
    new_population = population.evolve(
        **default_evolve_params,
        tree_pop_max=pop_kwargs['tree_pop_max'],
        functions=pop_kwargs['functions'],
        terminals=pop_kwargs['terminals'])
    # Create a new population of same length
    assert len(new_population.trees) == len(population.trees)
    for i, tree in enumerate(new_population.trees, start=1):
        assert tree.id == i  # Ids match index order
        assert tree.fitness is None  # Fitness is not inherited
    assert new_population.gen_id == 2  # Generation is incremented
    assert len(new_population.history) == 1  # History is updated

    # TODO: Evaluate again to check that everything works and that the
    # first generation doesn't affect the second.
