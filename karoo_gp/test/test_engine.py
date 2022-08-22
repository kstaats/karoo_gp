import pytest
import numpy as np

from karoo_gp import NumpyEngine, TensorflowEngine, Tree, NodeData, get_nodes

@pytest.fixture
def trees():
    return [
        Tree.load(1, 'g((a)+((b)*(c)))'),
        Tree.load(1, 'f(((a)*(b))/((b)*(c)))'),
        Tree.load(1, 'f(((a)<(b))and((a)<(c)))'),
        Tree.load(1, 'f(((a)<(10))if((a)>=(3))else((a)>(0)))'),
    ]

@pytest.fixture
def X():
    return np.array([[1, 2, 3], [2, 3, 4]]), np.array([[3, 4, 5], [4, 5, 6]])

class MockModel:
    random_state = 1000
    cache_ = dict()
    nodes = [NodeData(t, 'terminal') for t in ['a', 'b', 'c']]
    def get_nodes(self, *args, **kwargs):
        return get_nodes(*args, **kwargs, lib=self.nodes)

@pytest.mark.parametrize('engine_type', ['numpy', 'tensorflow'])
def test_engine(X, trees, engine_type):
    X_train, X_test = X

    # Initialize
    model = MockModel()
    engine = dict(
        numpy=NumpyEngine,
        tensorflow=TensorflowEngine
    )[engine_type](model)
    assert engine.engine_type == engine_type
    assert '64' in str(engine.dtype)
    assert len(engine.operators) > 0

    # Test predict
    train_pred = engine.predict(trees, X_train)
    assert isinstance(train_pred, np.ndarray)
    assert train_pred.dtype == engine.dtype
    assert train_pred.shape == (len(trees), len(X_train))
    assert ([list(p) for p in train_pred] ==
        [[7.0, 14.0], [0.3333333333333333, 0.5], [1.0, 1.0], [1.0, 1.0]])

    # Test skip cached expressions
    X_test_hash = hash(X_test.data.tobytes())
    model.cache_[X_test_hash] = {trees[i].expression: 'dummy' for i in (0, 2)}
    test_pred = engine.predict(trees, X_test, X_test_hash)
    assert ([list(p) for p in test_pred] ==
        [[0.0, 0.0], [0.6, 0.6666666666666666], [0.0, 0.0], [1.0, 1.0]])
