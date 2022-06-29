from sklearn.utils.estimator_checks import check_estimator
from karoo_gp import BaseGP

def test_sklearn_estimator():
    check_estimator(BaseGP())
