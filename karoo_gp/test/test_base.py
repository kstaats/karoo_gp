import pytest
import numpy as np

from karoo_gp import Base_GP, Regressor_GP, MultiClassifier_GP, Matching_GP, Tree
from .util import load_data

@pytest.fixture
def default_kwargs(tmp_path):
    return dict(
        tree_type='r',
        tree_depth_base=3,
        tree_depth_max=3,
        tree_depth_min=1,
        tree_pop_max=100,
        gen_max=2,
        tourn_size=7,
        filename='',
        output_dir=str(tmp_path),
        evolve_repro=0.1,
        evolve_point=0.1,
        evolve_branch=0.2,
        evolve_cross=0.6,
        display='s',
        precision=6,
        swim='p',
        mode='s',
        seed=1000,
        terminals=['a', 'b'],
        functions=['+', '-', '*', '/'],
    )

@pytest.mark.parametrize('X_shape', [(10, 2), (100, 2)])
def test_model_base(default_kwargs, X_shape):
    # Initialize model
    kwargs = dict(default_kwargs)
    model = Base_GP(**kwargs)
    # Check all kwargs loaded correctly
    for k, v in kwargs.items():
        if k not in ['terminals', 'functions']:  # term/func converted to cls
            assert getattr(model, k) == v
    assert isinstance(model.functions, Functions)
    assert isinstance(model.terminals, Terminals)

    # Initialize data
    X = np.ones(X_shape)
    y = np.sum(X, axis=1)
    noise = np.random.rand(y.shape[0])
    noise  = (noise - 0.5) * 0.1
    y = y + noise

    # Test predict and score functions (independent of population/data)
    trees =[Tree.load(1, 'f((a)+(b))'), Tree.load(2, 'f((a)*(a))')]
    predictions = model.predict(X, trees)
    for pred in predictions:
        assert pred.shape == y.shape
    scores = [model.score(p, y, t) for p, t in zip(predictions, trees)]
    expected = {
        (10, 2): [dict(fitness=0.028375204003104337), dict(fitness=0.9979865501818704)],
        (100, 2): [dict(fitness=0.0250126405341231), dict(fitness=0.9986793786245746)],
    }[X_shape]
    for exp, actual in zip(expected, scores):
        assert exp['fitness'] == actual['fitness']

    # Fit 1 generation to process data, predict and score, but not evolve.
    model.gen_max = 1
    model.fit(X, y)
    assert model.population.gen_id == 1
    assert model.X_hash == hash(X.data.tobytes())  # Fingerprint of X saved
    if X.shape[0] < 11:
        assert model.X_train.shape == model.X_test.shape
        assert model.y_train.shape == model.y_train.shape
    else:  # X was split into train and test sets
        assert model.X_train.shape[0] + model.X_test.shape[0] == X.shape[0]
        assert model.X_train.shape[1] == model.X_test.shape[1] == X.shape[1]
        assert model.y_train.shape[0] + model.y_test.shape[0] == y.shape[0]
    X_train_hash = hash(model.X_train.data.tobytes())
    assert X_train_hash in model.cache  # Tree scores are cached by X_train
    unique_expressions = set([t.expression for t in model.population.trees])
    assert len(model.cache[X_train_hash]) == len(unique_expressions)

    # Fit 2 generations to evolve the population
    old_pop_size = len(model.population.trees)
    model.gen_max = 2
    model.fit(X, y)
    assert model.population.gen_id == 2
    assert len(model.population.trees) == old_pop_size
    best_fitness = model.population.fittest().fitness
    for tree in model.population.trees:
        assert tree.fitness >= best_fitness

def test_base_rng(default_kwargs):
    model = Base_GP(**default_kwargs)
    assert model.rng.integers(1000) == 585
    assert np.random.randint(1000) == 435

@pytest.mark.parametrize('ker', ['c', 'r', 'm'])
def test_model_kernel(tmp_path, paths, default_kwargs, ker):

    # Initialize Model for dataset
    cls = dict(
        m=Matching_GP,
        r=Regressor_GP,
        c=MultiClassifier_GP
    )[ker]
    data = load_data(tmp_path, paths, ker)
    kwargs = dict(default_kwargs)
    kwargs['terminals'] = data['terminals']
    kwargs['functions'] = data['functions']
    model = cls(**kwargs)
    X, y = data['X'], data['y']

    # Helper function to compare results with expected for 3 categories
    def compare_expected(model, expected):
        """Test models fields against expected dict"""
        fitlist = ''.join(map(str, model.population.fittest_dict))
        assert expected['fitlist'] == fitlist

        fittest_id = max(model.population.fittest_dict)
        fittest = model.population.trees[fittest_id - 1]
        assert expected['fit'] == fittest.fitness
        assert expected['sym'] == fittest.expression == model.population.fittest_dict[fittest_id]

    # Evaluate only (don't evolve) by fitting for 1 generation
    model.gen_max = 1
    model.fit(X, y)
    initial_expected = {
        'c': dict(sym='pl + pw - 2*sw', fit=109.0,
                  fitlist='1235689101112131415161718192022232426272829313375'),
        'r': dict(sym='1', fit=0.05, fitlist='17101525314580'),
        'm': dict(sym='3*b', fit=10.0, fitlist='12952'),
    }
    compare_expected(model, initial_expected[ker])

    # Fit
    model.gen_max = 2
    model.fit(X, y)
    fit_expected = {
        'c': dict(sym='-pl/(pw*sw) + pw', fit=110.0, fitlist='121023'),
        'r': dict(sym='1', fit=0.05,
                  fitlist='12101214182432364355608183919294'),
        'm': dict(sym='3*b', fit=10.0, fitlist='109798'),
    }
    compare_expected(model, fit_expected[ker])
