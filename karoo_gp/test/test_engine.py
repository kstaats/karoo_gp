import pytest
import numpy as np

from karoo_gp import NumpyEngine, TensorflowEngine, Terminals, Tree

@pytest.fixture
def trees():
    return [
        Tree.load(1, 'g((a)+((b)*(c)))'),
        Tree.load(1, 'f(((a)*(b))/((b)*(c)))')
    ]

@pytest.fixture
def X():
    return np.array([[1, 2, 3], [2, 3, 4]]), np.array([[3, 4, 5], [4, 5, 6]])

class MockModel:
    seed = 1000
    cache = dict()
    terminals = Terminals(['a', 'b', 'c'])

@pytest.mark.parametrize('engine_type', ['numpy', 'tensorflow'])
def test_engine(trees, X, engine_type):
    X_train, X_test = X

    # Initialize
    model = MockModel()
    engine = dict(
        numpy=NumpyEngine, tensorflow=TensorflowEngine
    )[engine_type](model)
    assert engine.engine_type == engine_type
    assert '64' in str(engine.dtype)
    assert len(engine.operators) > 0

    # Test predict
    train_pred = engine.predict(trees, X_train)
    assert type(train_pred) == np.ndarray
    assert train_pred.dtype == engine.dtype
    assert train_pred.shape == (len(trees), len(X_train))
    assert str([list(p) for p in train_pred]) == '[[7.0, 14.0], [3.0, 8.0]]'

    # Test single tree
    pred = engine.predict(trees[0], X_train)
    assert pred.shape == (1, len(X_train))

    # Test skip cached expressions
    X_test_hash = hash(X_test.data.tobytes())
    model.cache[X_test_hash] = {trees[0].expression: 'dummy'}
    test_pred = engine.predict(trees, X_test, X_test_hash)
    assert sum(test_pred[0]) == 0
    assert sum(test_pred[1]) != 0
    assert str([list(p) for p in test_pred]) == '[[0.0, 0.0], [15.0, 24.0]]'
