import pytest
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  # from https://www.tensorflow.org/guide/migrate

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
def test_population_generate(default_kwargs, tree_type):
    kwargs = dict(default_kwargs)
    kwargs['tree_type'] = tree_type
    kwargs['tree_pop_max'] = 10
    population = Population.generate(**kwargs)

    # This will CHANGE with branch-api update
    expected_root = {
        'f': 'b20c6bba22a114b16a455c4215df3676',
        'g': '98c9b9cf6ecb61716b0b76fc72603094',
        'r': '186c8dc7114d5dd1eecfc530128b86fe',
    }
    trees = [t.root for t in population.trees]
    assert hasher(trees) == expected_root[tree_type]

    # This will NOT change with branch-api update
    expected_raw_expression = {
        'f': '(a)*(a)/(a)-(a)-(b)-(a)-(b)*(a)',
        'g': '(b)+(b)',
        'r': '(b)-(a)/(b)*(b)',
    }
    assert population.trees[-1].raw_expression == expected_raw_expression[tree_type]

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
    np.random.seed(1000)
    tf.set_random_seed(1000)
    dataset_params = load_data(kernel, save_dir='test')

    # Initialize population using dataset terminals
    kwargs = {
        **default_kwargs,
        'tree_pop_max': 10,
        'terminals': dataset_params['terminals'],
        'fitness_type': dataset_params['fitness_type']
    }
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
        'c': dict(exp='pl - pl*sw/pw', fit=41.0),
        'r': dict(exp='2*r', fit=205.509979),
        'm': dict(exp='-2*a - b + 2*c', fit=1.0),
    }
    assert population.fittest().expression == expected[kernel]['exp']
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
        'c': dict(exp='pl + pw - sl', fit=98.0),
        'r': dict(exp='2*r', fit=205.509979),
        'm': dict(exp='b**2', fit=1.0),
    }
    assert new_population.fittest().expression == expected[kernel]['exp']
    assert new_population.fittest().fitness() == expected[kernel]['fit']
