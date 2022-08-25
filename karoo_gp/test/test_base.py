import csv
import pytest
import numpy as np
from pathlib import Path

from karoo_gp import BaseGP, RegressorGP, MultiClassifierGP, MatchingGP, Tree, function_lib
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
        random_state=1000,
        terminals=['a', 'b'],
        functions=['+', '-', '*', '/'],
    )

@pytest.mark.parametrize('X_shape', [(10, 2), (100, 2)])
def test_model_base(default_kwargs, X_shape):
    # Initialize model
    kwargs = dict(default_kwargs)
    model = BaseGP(**kwargs)
    # Check all kwargs loaded correctly
    for k, v in kwargs.items():
        assert getattr(model, k) == v

    # Initialize data
    np.random.seed(kwargs['random_state'])  # Not set by karoo until fit
    X = np.ones(X_shape)
    y = np.sum(X, axis=1)
    noise = np.random.rand(y.shape[0])
    noise  = (noise - 0.5) * 0.1
    y = y + noise

    # Fit 1 generation to process data, predict and score, but not evolve.
    model.gen_max = 1
    model.fit(X, y)
    assert model.population.gen_id == 1
    assert model.X_hash_ == hash(X.data.tobytes())  # Fingerprint of X saved
    if X.shape[0] < 11:
        assert model.X_train.shape == model.X_test.shape
        assert model.y_train.shape == model.y_train.shape
    else:  # X was split into train and test sets
        assert model.X_train.shape[0] + model.X_test.shape[0] == X.shape[0]
        assert model.X_train.shape[1] == model.X_test.shape[1] == X.shape[1]
        assert model.y_train.shape[0] + model.y_test.shape[0] == y.shape[0]
    X_train_hash = hash(model.X_train.data.tobytes())
    assert X_train_hash in model.cache_  # Tree scores are cached by X_train
    unique_expressions = set([t.expression for t in model.population.trees])
    assert len(model.cache_[X_train_hash]) == len(unique_expressions)

    # Test predict and score functions (independent of population/data)
    trees =[Tree.load(1, 'f((a)+(b))'), Tree.load(2, 'f((a)*(a))')]
    predictions = model.batch_predict(X, trees)
    for pred in predictions:
        assert pred.shape == y.shape
    scores = [model.calculate_score(p, y) for p in predictions]
    expected = {
        (10, 2): [dict(fitness=0.028375204003104337), dict(fitness=0.9979865501818704)],
        (100, 2): [dict(fitness=0.0250126405341231), dict(fitness=0.9986793786245746)],
    }[X_shape]
    for exp, actual in zip(expected, scores):
        assert exp['fitness'] == actual['fitness']

    # Fit 2 generations to evolve the population
    old_pop_size = len(model.population.trees)
    model.gen_max = 2
    model.fit(X, y)
    assert model.population.gen_id == 2
    assert len(model.population.trees) == old_pop_size
    best_fitness = model.population.fittest().fitness
    for tree in model.population.trees:
        assert tree.fitness >= best_fitness

def test_model_save_load(tmp_path, paths, default_kwargs):

    # Initialize regression model and fit 2 generations
    data = load_data(tmp_path, paths, 'r')
    X, y = data['X'], data['y']
    kwargs = dict(default_kwargs)
    kwargs['tree_pop_max'] = 10
    kwargs['terminals'] = data['terminals']
    kwargs['functions'] = data['functions']
    kwargs['gen_max'] = 2
    model = RegressorGP(**kwargs)
    model.fit(X, y)

    # Check saved record
    fname = next(tmp_path.iterdir())
    assert 'population_a.csv' in str(fname)
    with open(fname) as f:
        reader = csv.reader(f)
        for i in range(2):
            header = next(reader)
            assert header[0] == f'Karoo GP by Kai Staats - Generation {i+1}'
            for j in range(kwargs['tree_pop_max']):
                saved_tree = next(reader)[0]  # First (and only) item on line
                tree = Tree.load(j+1, saved_tree)
                assert len(tree.save()) == len(saved_tree)
            next(reader)  # Empty line after population

    # Check load function
    # - Note the last tree of pop
    original = model.population.trees[-1].save()
    # - Throw error for unrecognized population
    with pytest.raises(ValueError):
        model.save_population('h')
    # - Save pop to file
    loc = model.save_population('s')
    assert loc == Path(f'{default_kwargs["output_dir"]}/population_s.csv')
    # - Evolve one gen
    model.gen_max = 3
    model.fit(X, y)
    # - Confirm last tree is different
    updated = model.population.trees[-1].save()
    assert original != updated
    # - Load saved pop from file
    model.load_population()
    # - Confirm last tree is same as it was before
    reloaded = model.population.trees[-1].save()
    assert original == reloaded


@pytest.mark.parametrize('ker', ['c', 'r', 'm'])
def test_model_kernel(tmp_path, paths, default_kwargs, ker):

    # Initialize Model for dataset
    cls = dict(
        m=MatchingGP,
        r=RegressorGP,
        c=MultiClassifierGP
    )[ker]
    data = load_data(tmp_path, paths, ker)
    kwargs = dict(default_kwargs)
    kwargs['terminals'] = data['terminals']
    kwargs['functions'] = data['functions']
    if ker == 'm':
        kwargs['random_state'] = 1016
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
        'c': dict(sym='pl*sw - pl + pw**2 - pw - sl/pw', fit=85.0, fitlist='123678101123'),
        'r': dict(sym='1', fit=0.05, fitlist='172637628091'),
        'm': dict(sym='a + b + c', fit=10.0, fitlist='1213374958'),
    }
    compare_expected(model, initial_expected[ker])

    # Fit
    model.gen_max = 2
    model.fit(X, y)
    fit_expected = {
        'c': dict(sym='pl*sw - pl + pw**2 - pw - sl/pw', fit=85.0, fitlist='12352'),
        'r': dict(sym='1', fit=0.05,
                  fitlist='11334416084'),
        'm': dict(sym='a + b + c', fit=10.0, fitlist='1443'),
    }
    compare_expected(model, fit_expected[ker])

@pytest.mark.parametrize('dtype', ['int', 'float'])
@pytest.mark.parametrize('digits', [2, 4])
def test_model_unfit_trees(rng, default_kwargs, dtype, digits):

    # Generate enormous trees with all available operators
    kwargs = dict(default_kwargs)
    kwargs['tree_depth_base'] = 8
    kwargs['tree_depth_max'] = 9
    kwargs['tree_pop_max'] = 20
    kwargs['functions'] = [f.label for f in function_lib]
    model = BaseGP(**kwargs)

    # Compile a synthetic dataset of specified type
    def get_X_y(values, n_samples=100):
        X = np.array([[rng.choice(values) for _ in range(2)]
                      for _ in range(n_samples)])
        y = np.array([rng.choice(values) for _ in range(n_samples)])
        return X, y
    if dtype == 'int':
        X, y = get_X_y(range(10**digits))
    elif dtype == 'float':
        X, y = get_X_y(np.arange(0, 1, 1/10**digits))

    # Evolve the trees for 2 generations
    model.fit(X, y)
    expected = {
        ('int', 2): dict(n=9, first='g((abs(b))**((b)*(a)))'),
        ('int', 4): dict(n=10, first='g((abs(b))**((b)*(a)))'),
        ('float', 2): dict(n=2, first='f(square(((square(square(sqrt(square(sqrt(a))))))*((((((a)+(a))**((a)/(b)))+(((b)/(b))**((a)+(b))))/((((a)**(a))+(sqrt(a)))*(((a)+(a))if((b)!=(a))else(abs(a)))))/(((abs(abs(b)))if((square(b))>((a)/(a)))else(((b)+(a))/((a)-(b))))/(sqrt(abs((b)/(a)))))))if((sqrt(((abs(sqrt(a)))**(sqrt((a)**(a))))**((((b)+(b))**(sqrt(b)))*(((a)**(b))*(sqrt(a))))))>=((((square((b)+(a)))**(((b)*(a))+(sqrt(a))))/(square(sqrt(square(a)))))**((sqrt(sqrt((b)/(a))))*((((b)-(a))-(square(b)))-((sqrt(b))+((a)*(b)))))))else(((abs((((a)**(b))if((a)<=(a))else((b)*(a)))+(((b)*(a))-((b)+(a)))))/(sqrt(sqrt(abs((b)+(a))))))/((((((b)*(a))-((a)+(b)))*((sqrt(b))+((a)-(a))))**(((abs(b))**((b)+(a)))if(((a)/(a))<=(sqrt(b)))else(abs(sqrt(a)))))-(sqrt((((b)*(b))+((a)*(a)))**(square(sqrt(b)))))))))'),
        ('float', 4): dict(n=3, first='f(square(((square(square(sqrt(square(sqrt(a))))))*((((((a)+(a))**((a)/(b)))+(((b)/(b))**((a)+(b))))/((((a)**(a))+(sqrt(a)))*(((a)+(a))if((b)!=(a))else(abs(a)))))/(((abs(abs(b)))if((square(b))>((a)/(a)))else(((b)+(a))/((a)-(b))))/(sqrt(abs((b)/(a)))))))if((sqrt(((abs(sqrt(a)))**(sqrt((a)**(a))))**((((b)+(b))**(sqrt(b)))*(((a)**(b))*(sqrt(a))))))>=((((square((b)+(a)))**(((b)*(a))+(sqrt(a))))/(square(sqrt(square(a)))))**((sqrt(sqrt((b)/(a))))*((((b)-(a))-(square(b)))-((sqrt(b))+((a)*(b)))))))else(((abs((((a)**(b))if((a)<=(a))else((b)*(a)))+(((b)*(a))-((b)+(a)))))/(sqrt(sqrt(abs((b)+(a))))))/((((((b)*(a))-((a)+(b)))*((sqrt(b))+((a)-(a))))**(((abs(b))**((b)+(a)))if(((a)/(a))<=(sqrt(b)))else(abs(sqrt(a)))))-(sqrt((((b)*(b))+((a)*(a)))**(square(sqrt(b)))))))))'),
    }
    # Compare the actual first unfit to expected first unfit
    assert len(model.unfit) == expected[dtype, digits]['n']
    assert model.unfit[0] == expected[dtype, digits]['first']

    # Warnings:
    # - overflow encountered in float_power
    # - overflow encountered in square
    # - divide by zero encountered in float_power
    # - invalid value encountered in float_power
    # - invalid value encountered in multiply
    # - invalid value encountered in true_divide
