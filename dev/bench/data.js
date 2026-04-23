window.BENCHMARK_DATA = {
  "lastUpdate": 1776940289332,
  "repoUrl": "https://github.com/fredshone/acteval",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "26383933+fredshone@users.noreply.github.com",
            "name": "Fred",
            "username": "fredshone"
          },
          "committer": {
            "email": "26383933+fredshone@users.noreply.github.com",
            "name": "Fred",
            "username": "fredshone"
          },
          "distinct": true,
          "id": "ee1b59bcf33862ef79ff34bbf4f38f0a7c9bb588",
          "message": "plotting for article",
          "timestamp": "2026-04-23T11:29:09+01:00",
          "tree_id": "81b9f1f12e991fabc72e4893b4f1dd515baeaa76",
          "url": "https://github.com/fredshone/acteval/commit/ee1b59bcf33862ef79ff34bbf4f38f0a7c9bb588"
        },
        "date": 1776940289059,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[1k]",
            "value": 5.618370871918018,
            "unit": "iter/sec",
            "range": "stddev: 0.020507323289777463",
            "extra": "mean: 177.98753816666726 msec\nrounds: 6"
          },
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[20k]",
            "value": 0.512578178369944,
            "unit": "iter/sec",
            "range": "stddev: 0.010457730164065463",
            "extra": "mean: 1.9509219124000012 sec\nrounds: 5"
          },
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[100k]",
            "value": 0.10070076316124522,
            "unit": "iter/sec",
            "range": "stddev: 0.02351962984481919",
            "extra": "mean: 9.930411335599995 sec\nrounds: 5"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=256]",
            "value": 85.94228238375423,
            "unit": "iter/sec",
            "range": "stddev: 0.00013198074215185453",
            "extra": "mean: 11.635716113923351 msec\nrounds: 79"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=512]",
            "value": 21.14942933956104,
            "unit": "iter/sec",
            "range": "stddev: 0.00020611273861999784",
            "extra": "mean: 47.28259963636235 msec\nrounds: 22"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=1024]",
            "value": 5.196954614738164,
            "unit": "iter/sec",
            "range": "stddev: 0.0008000382441265737",
            "extra": "mean: 192.42038349999766 msec\nrounds: 6"
          }
        ]
      }
    ]
  }
}