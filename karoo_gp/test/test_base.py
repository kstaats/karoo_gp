import hashlib

import pytest

from karoo_gp import Base_GP

@pytest.fixture
def default_kwargs(tmp_path):
    return dict(
        kernel='m',
        tree_type='r',
        tree_depth_base=3,
        tree_depth_max=3,
        tree_depth_min=1,
        tree_pop_max=100,
        gen_max=10,
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
    )

def test_base_init(default_kwargs):
    model = Base_GP(**default_kwargs)
    for k, v in default_kwargs.items():
        assert getattr(model, k) == v

@pytest.mark.parametrize('ker', ['c', 'r', 'm'])
def test_base_fit(default_kwargs, ker):
    # Initialize, check most fit
    kwargs = dict(default_kwargs)
    kwargs['kernel'] = ker
    kwargs['gen_max'] = 2
    model = Base_GP(**kwargs)

    def compare_expected(model, expected):
        """Test models fields against expected dict"""
        fitlist = ''.join([str(f) for f in model.fittest_dict.keys()])
        assert expected['fitlist'] == fitlist
        fittest = max(model.fittest_dict.keys())
        assert expected['sym'] == str(model.fittest_dict[fittest])
        assert expected['fit'] == model.population_a[fittest][12][1]

    initial_expected = {
        'c': dict(sym='pl + pw - 2*sw', fit='109.0',
                  fitlist='1235689101112131415161718192022232426272829313375'),
        'r': dict(sym='1', fit='0.05', fitlist='17101525314580'),
        'm': dict(sym='3*b', fit='10.0', fitlist='52'),
    }
    compare_expected(model, initial_expected[ker])

    model.fit()
    fit_expected = {
        'c': dict(sym='-pl/(pw*sw) + pw', fit='110.0', fitlist='121023'),
        'r': dict(sym='1', fit='0.05',
                  fitlist='12101214182432364355608183919294'),
        'm': dict(sym='3*b', fit='10.0', fitlist='104269100'),
    }
    compare_expected(model, fit_expected[ker])
