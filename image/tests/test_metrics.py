import numpy as np
import pandas as pd

from ai4eyes_image.eval.metrics import boot_ci, cls_metrics, reg_metrics
from ai4eyes_image.train.trainer import class_weights, lr_lambda


def test_macro_over_present_classes():
    y = np.array([0, 0, 1, 1])
    P = np.eye(3)[[0, 0, 1, 2]]
    m = cls_metrics(y, P)
    assert m["accuracy"] == 0.75 and np.isclose(m["recall"], 0.75)   # class 2 absent from y → not averaged


def test_regression_metrics():
    y, p = np.array([0.1, 0.3, 0.5]), np.array([0.15, 0.3, 0.4])
    m = reg_metrics(y, p)
    assert np.isclose(m["mae_mm"], 0.05) and m["true_le_0_2"] == 1 and m["pred_le_0_2"] == 1


def test_bootstrap_resamples_whole_patients():
    e = pd.DataFrame({"patient_pid": ["a", "a", "b", "b"], "v": [1.0, 1.0, 0.0, 0.0]})
    lo, hi = boot_ci(e, lambda x: x.v.mean(), n=200)
    assert lo == 0.0 and hi == 1.0                                     # only 0, 0.5, 1 are possible


def test_class_weights_and_schedule():
    w = class_weights(np.array([0] * 90 + [1] * 10), "sqrt_inv")
    assert w[1] > w[0] and class_weights([0, 1], "none") is None
    f = lr_lambda(10, 100)
    assert np.isclose(f(9), 1.0) and np.isclose(f(100), 0.0)
