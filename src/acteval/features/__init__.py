"""Feature extraction functions for acteval evaluation domains.

This package is internal implementation — not a stable public API. Functions
here take a ``Population`` object and return per-person feature data
(``PidFeatures``) or aggregated feature dicts ``{key: (values, weights)}``.

Sub-modules by domain:

- ``participation`` — activity participation rates and joint rates
- ``times``         — start time, duration, and joint timing distributions
- ``transitions``   — n-gram activity transition features (2-, 3-, 4-gram)
- ``frequency``     — binned activity density/count over the day
- ``structural``    — schedule feasibility flags (home-based, no consecutive duplicates)
- ``creativity``    — diversity, novelty, and conservatism metrics

Internal helpers (underscore-prefixed):

- ``_pid_features`` — ``PidFeatures`` dataclass for per-person subsetting
- ``_utils``        — low-level grouping, compression, and counting helpers
- ``_discretise``   — time-discretisation into a [P, C, H, W] tensor

Use ``list_features()`` to get a catalogue of all available feature functions::

    from acteval import list_features
    print(list_features())
"""

from acteval.features.catalogue import list_features as list_features
