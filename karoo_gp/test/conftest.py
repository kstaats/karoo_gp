import pytest
import numpy as np

@pytest.fixture
def rng():
    return np.random.default_rng(1000)

@pytest.fixture
def mock_func():
    def handler(*args, **kwargs):
        pass
    return handler
