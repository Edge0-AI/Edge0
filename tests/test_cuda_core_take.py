"""``take`` on the torch backend follows numpy semantics, including negative axes."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from edge0.backends.cuda import core as cc  # noqa: E402


@pytest.mark.parametrize("axis", [0, 1, 2, -1, -2, -3])
def test_take_matches_numpy_for_every_axis(axis):
    rng = np.random.default_rng(0)
    a = rng.standard_normal((4, 5, 6)).astype(np.float32)
    indices = np.array([[3, 0], [2, 1]])
    out = cc.take(torch.as_tensor(a, device=cc.DEVICE),
                  torch.as_tensor(indices, device=cc.DEVICE), axis=axis)
    expected = np.take(a, indices, axis=axis)
    assert tuple(out.shape) == expected.shape
    np.testing.assert_array_equal(out.cpu().numpy(), expected)


def test_take_without_axis_flattens_like_numpy():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    indices = np.array([[11, 0], [5, 6]])
    out = cc.take(torch.as_tensor(a, device=cc.DEVICE),
                  torch.as_tensor(indices, device=cc.DEVICE))
    np.testing.assert_array_equal(out.cpu().numpy(), np.take(a, indices))
