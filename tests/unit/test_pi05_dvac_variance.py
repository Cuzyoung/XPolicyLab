from __future__ import annotations

import numpy as np
import pytest
from openpi.policies.policy import compute_dvac_variance


def test_compute_dvac_variance_matches_equation_four_and_ignores_padding() -> None:
    clean_tail = np.zeros((2, 1, 2, 4), dtype=np.float32)
    clean_tail[1, 0, 0, :2] = [2.0, 4.0]
    clean_tail[1, 0, :, 2:] = 1000.0

    variance = compute_dvac_variance(clean_tail, action_dim=2)

    np.testing.assert_allclose(variance, [[5.0, 0.0]])


def test_compute_dvac_variance_rejects_an_invalid_valid_dimension() -> None:
    with pytest.raises(ValueError, match="1 <= D <= 4"):
        compute_dvac_variance(np.zeros((5, 1, 3, 4), dtype=np.float32), 5)
