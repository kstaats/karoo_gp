import hashlib

import pytest

from karoo_gp import Base_GP

@pytest.fixture
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
        output_dir='test',
        evolve_repro=0.1,
        evolve_point=0.1,
        evolve_branch=0.2,
        evolve_cross=0.6, 
        display='s',
        precision=6,
        swim='p',
        mode='s'
    )

@pytest.fixture
def model(kwargs):
    """Produces a single model used by all tests in module"""
    model = Base_GP(**kwargs)
    return model

def test_base_init(kwargs, model):
    for k, v in kwargs.items():
        assert getattr(model, k) == v

def test_base_fit(model):
    model.fit()
    # Hash of expected output.csv in /runs/
    hashes = {'a': '76afe1c7c10d7498a1f514f7c9b86998',
              'b': 'd41d8cd98f00b204e9800998ecf8427e',
              'f': 'd41d8cd98f00b204e9800998ecf8427e',
              's': 'd41d8cd98f00b204e9800998ecf8427e'}
    for k, v in hashes.items():
        fname = model.savefile[k]
        with open(fname, 'rb') as f:
            bytestring = f.read()
        hasher = hashlib.md5()
        hasher.update(bytestring)
        hash = hasher.hexdigest()
        assert v == hash
            
