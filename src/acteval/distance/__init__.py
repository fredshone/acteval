"""Distance metrics used to score synthetic vs. observed feature distributions.

Public API:

- ``emd``  — Earth Mover's Distance (Wasserstein-1) for 1-D and 2-D distributions
- ``mae``  — Mean Absolute Error on weighted feature vectors
- ``mse``  — Mean Squared Error on weighted feature vectors
- ``mape`` — Mean Absolute Percentage Error (clamped to 1.0)

All functions accept ``(a, b)`` where each is a ``(values, weights)`` tuple
returned by the feature extraction functions.

Usage::

    from acteval.distance import emd, mae
"""

from acteval.distance.scalar import (
    abs_av_diff as abs_av_diff,
)
from acteval.distance.scalar import (
    mae as mae,
)
from acteval.distance.scalar import (
    mape as mape,
)
from acteval.distance.scalar import (
    mape_scalar as mape_scalar,
)
from acteval.distance.scalar import (
    mse as mse,
)
from acteval.distance.wasserstein import emd as emd
