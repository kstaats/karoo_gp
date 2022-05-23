import pytest
import numpy as np

from karoo_gp import Population
from .util import hasher, load_data

@pytest.fixture
def default_kwargs(rng, mock_func):
    return dict(
        log=mock_func,
        pause=mock_func,
        error=mock_func,
        gen_id=1,
        tree_type='r',
        tree_depth_base=3,
        tree_depth_max=3,
        tree_pop_max=100,
        functions=np.array([['+', 2], ['-', 2], ['*', 2], ['/', 2]]),
        terminals=['a', 'b', 'c'],
        rng=rng,
        fitness_type='max'
    )

@pytest.mark.parametrize('tree_type', ['f', 'g', 'r'])
@pytest.mark.parametrize('tree_pop_max', [10, 20])
def test_population_generate(default_kwargs, tree_type, tree_pop_max):
    kwargs = dict(default_kwargs)
    kwargs['tree_type'] = tree_type
    kwargs['tree_pop_max'] = tree_pop_max
    population = Population.generate(**kwargs)
    expected = {
        ('f', 10): 'b20c6bba22a114b16a455c4215df3676',
        ('f', 20): 'c80581a0c85b8b8c0d574a8f6f414d46',
        ('g', 10): '98c9b9cf6ecb61716b0b76fc72603094',
        ('g', 20): '253c685daad312ab7c7692df495289ee',
        ('r', 10): '186c8dc7114d5dd1eecfc530128b86fe',
        ('r', 20): 'f261514f6bc74a5897b64e5a44145958',
    }
    trees = [t.root for t in population.trees]
    assert hasher(trees) == expected[(tree_type, tree_pop_max)]

# EVALUATE

@pytest.fixture
def default_evaluate_params(mock_func):
    return dict(
        log=mock_func,
        pause=mock_func,
        error=mock_func,
        data_train=[],
        kernel='m',
        data_train_rows=0,
        tf_device_log=None,
        class_labels=[],
        tf_device="/gpu:0",
        terminals=['a', 'b', 'c', 's'],
        precision=6,
        savefile={},
        fx_data_tree_write=mock_func,
    )

@pytest.fixture
def default_evolve_params():
    return dict(
        swim='p',
        tree_depth_min=1,
        evolve_repro=0.1,
        evolve_point=0.1,
        evolve_branch=0.2,
        evolve_cross=0.6,
        tourn_size=7,
    )

@pytest.mark.parametrize('kernel', ['c', 'r', 'm'])
def test_population_class(default_kwargs, default_evaluate_params,
                          default_evolve_params, rng, kernel):
    # Load the dataset for kernel
    dataset_params = load_data(kernel, save_dir='test')

    # Initialize population using dataset terminals
    kwargs = dict(default_kwargs)
    kwargs['tree_pop_max'] = 10
    kwargs['terminals'] = dataset_params['terminals']
    kwargs['fitness_type'] = dataset_params['fitness_type']
    kwargs['rng'] = rng
    population = Population.generate(**kwargs)

    # Evaluate
    eval_params = dict(default_evaluate_params)
    eval_params['kernel'] = kernel
    eval_params['terminals'] = dataset_params['terminals']
    eval_params['data_train'] = dataset_params['data_train']
    eval_params['data_train_rows'] = dataset_params['data_train_rows']
    eval_params['class_labels'] = dataset_params['class_labels']
    eval_params['savefile'] = dataset_params['savefile']
    population.evaluate(**eval_params)
    expected = {
        'c': dict(exp='pl - pl*sw/pw', fit=43.0),
        'r': dict(exp='2*r', fit=205.509979),
        'm': dict(exp='-2*a - b + 2*c', fit=1.0),
    }
    assert str(population.fittest().sym()) == expected[kernel]['exp']
    assert population.fittest().fitness() == expected[kernel]['fit']

    # Evolve
    evolve_params = {**default_evolve_params, **eval_params}
    evolve_params['functions'] = dataset_params['functions']
    evolve_params['fitness_type'] = dataset_params['fitness_type']
    evolve_params['tree_depth_max'] = kwargs['tree_depth_max']
    evolve_params['tree_pop_max'] = kwargs['tree_pop_max']
    evolve_params['rng'] = kwargs['rng']
    new_population = population.evolve(**evolve_params)
    expected = {
        'c': 'b6aa7e4058ffc8ea23bed3e7f8c5a199',
        'r': '68d5f8da8c7059f587ccfe3115f41401',
        'm': 'd88cfd97628c821525612b221bc3713c',
    }
    trees = [t.root for t in new_population.trees]
    assert hasher(trees) == expected[kernel]

