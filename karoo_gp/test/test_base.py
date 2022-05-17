from karoo_gp import Base_GP
import pytest

@pytest.fixture(scope="module")
def kwargs():
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
        evolve_repro=0.1,
        evolve_point=0.1,
        evolve_branch=0.2,
        evolve_cross=0.6, 
        display='s',
        precision=6,
        swim='p',
        mode='s'
    )

@pytest.fixture(scope="module")
def model(kwargs):
    model = Base_GP(**kwargs)
    return model

def test_base_init(kwargs, model):
    for k, v in kwargs.items():
        if k == 'filename':
            continue  # Changes during the run
        assert getattr(model, k) == v

def test_base_fit(model):
    model.fit()