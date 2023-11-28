"""Tests."""

import numpy as np

from stream_mapper.core.typing import ArrayLike


def test_ndarray_is_arraylike():
    """Test that ndarray is ArrayLike."""
    assert isinstance(np.array([1, 2, 3]), ArrayLike)
