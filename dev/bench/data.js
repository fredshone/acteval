window.BENCHMARK_DATA = {
  "lastUpdate": 1777334029048,
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
      },
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
          "id": "306bf362e49d01cfbf8b77f28c51ee3622da6cc3",
          "message": "linting",
          "timestamp": "2026-04-23T11:37:20+01:00",
          "tree_id": "81dff18db478d017731f315cf383641597c01b8a",
          "url": "https://github.com/fredshone/acteval/commit/306bf362e49d01cfbf8b77f28c51ee3622da6cc3"
        },
        "date": 1776940756149,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[1k]",
            "value": 5.72291750900859,
            "unit": "iter/sec",
            "range": "stddev: 0.019132430216779128",
            "extra": "mean: 174.73604999999992 msec\nrounds: 6"
          },
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[20k]",
            "value": 0.5104732184797213,
            "unit": "iter/sec",
            "range": "stddev: 0.006304536562001392",
            "extra": "mean: 1.9589666289999996 sec\nrounds: 5"
          },
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[100k]",
            "value": 0.10061925538341467,
            "unit": "iter/sec",
            "range": "stddev: 0.02306212134627362",
            "extra": "mean: 9.938455578800006 sec\nrounds: 5"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=256]",
            "value": 86.34406473831504,
            "unit": "iter/sec",
            "range": "stddev: 0.00012206798380940187",
            "extra": "mean: 11.5815719705891 msec\nrounds: 68"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=512]",
            "value": 20.617895248183938,
            "unit": "iter/sec",
            "range": "stddev: 0.0002541415410962038",
            "extra": "mean: 48.50155595237501 msec\nrounds: 21"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=1024]",
            "value": 5.16404590189593,
            "unit": "iter/sec",
            "range": "stddev: 0.0004050352297688087",
            "extra": "mean: 193.64661333332833 msec\nrounds: 6"
          }
        ]
      },
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
          "id": "60caafa4ebf58254fbd85c13864991f5d241c0b9",
          "message": "remove old windows runners",
          "timestamp": "2026-04-27T02:01:09+01:00",
          "tree_id": "e370b24422043d61e6348ea0c399a4243097af06",
          "url": "https://github.com/fredshone/acteval/commit/60caafa4ebf58254fbd85c13864991f5d241c0b9"
        },
        "date": 1777251786570,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[1k]",
            "value": 5.74939902442338,
            "unit": "iter/sec",
            "range": "stddev: 0.019602006640237528",
            "extra": "mean: 173.93122233332767 msec\nrounds: 6"
          },
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[20k]",
            "value": 0.5102453552232405,
            "unit": "iter/sec",
            "range": "stddev: 0.007540271813128136",
            "extra": "mean: 1.959841456199996 sec\nrounds: 5"
          },
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[100k]",
            "value": 0.10044217020186379,
            "unit": "iter/sec",
            "range": "stddev: 0.03102441062666619",
            "extra": "mean: 9.955977633599996 sec\nrounds: 5"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=256]",
            "value": 84.75603574132244,
            "unit": "iter/sec",
            "range": "stddev: 0.0001214148643354016",
            "extra": "mean: 11.79856975675485 msec\nrounds: 74"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=512]",
            "value": 20.45402124056152,
            "unit": "iter/sec",
            "range": "stddev: 0.00021775273668852167",
            "extra": "mean: 48.89014185713963 msec\nrounds: 21"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=1024]",
            "value": 5.173703869543688,
            "unit": "iter/sec",
            "range": "stddev: 0.000389033290343147",
            "extra": "mean: 193.28512516666288 msec\nrounds: 6"
          }
        ]
      },
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
          "id": "3fbea497e1c5f018979e75b7eeaf65861d6b5b09",
          "message": "fix readme demo",
          "timestamp": "2026-04-27T06:39:44+01:00",
          "tree_id": "ec9a9a28f2738494f489317d6910c9973c9bb903",
          "url": "https://github.com/fredshone/acteval/commit/3fbea497e1c5f018979e75b7eeaf65861d6b5b09"
        },
        "date": 1777268500569,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[1k]",
            "value": 3.4280375017402385,
            "unit": "iter/sec",
            "range": "stddev: 0.020464403417070412",
            "extra": "mean: 291.71209460000114 msec\nrounds: 5"
          },
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[20k]",
            "value": 0.5117830640387955,
            "unit": "iter/sec",
            "range": "stddev: 0.011197763208591147",
            "extra": "mean: 1.9539528957999976 sec\nrounds: 5"
          },
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[100k]",
            "value": 0.09975973162729253,
            "unit": "iter/sec",
            "range": "stddev: 0.16586894057577756",
            "extra": "mean: 10.024084705200002 sec\nrounds: 5"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=256]",
            "value": 88.00275379490085,
            "unit": "iter/sec",
            "range": "stddev: 0.00012053156240839313",
            "extra": "mean: 11.363280771083586 msec\nrounds: 83"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=512]",
            "value": 21.095230748646237,
            "unit": "iter/sec",
            "range": "stddev: 0.0001289019437048384",
            "extra": "mean: 47.40407971428205 msec\nrounds: 21"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=1024]",
            "value": 5.235822117558115,
            "unit": "iter/sec",
            "range": "stddev: 0.0003499324598244578",
            "extra": "mean: 190.99197366666468 msec\nrounds: 6"
          }
        ]
      },
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
          "id": "17a26f9a9061b2137ee996f674768cc639d7d55f",
          "message": "adds mincount trigger for allowing nice examples",
          "timestamp": "2026-04-27T06:56:59+01:00",
          "tree_id": "c783b181d6288a32cfdbba14210e1b55b886ffca",
          "url": "https://github.com/fredshone/acteval/commit/17a26f9a9061b2137ee996f674768cc639d7d55f"
        },
        "date": 1777269534124,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[1k]",
            "value": 6.432385583803619,
            "unit": "iter/sec",
            "range": "stddev: 0.025677918622306013",
            "extra": "mean: 155.46331714285648 msec\nrounds: 7"
          },
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[20k]",
            "value": 0.5199843419527831,
            "unit": "iter/sec",
            "range": "stddev: 0.010998693425660346",
            "extra": "mean: 1.9231348318000017 sec\nrounds: 5"
          },
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[100k]",
            "value": 0.0991261781051652,
            "unit": "iter/sec",
            "range": "stddev: 0.023460954836478757",
            "extra": "mean: 10.088152485199998 sec\nrounds: 5"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=256]",
            "value": 78.36402597272185,
            "unit": "iter/sec",
            "range": "stddev: 0.00020289354328561436",
            "extra": "mean: 12.76095743661888 msec\nrounds: 71"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=512]",
            "value": 20.61597949105744,
            "unit": "iter/sec",
            "range": "stddev: 0.00019824889357572244",
            "extra": "mean: 48.5060630000029 msec\nrounds: 21"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=1024]",
            "value": 5.200695177964658,
            "unit": "iter/sec",
            "range": "stddev: 0.00027348227823055456",
            "extra": "mean: 192.28198649999703 msec\nrounds: 6"
          }
        ]
      },
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
          "id": "40b4357fdd0adc042bab5fc917c1c1404131152f",
          "message": "adds agent txt",
          "timestamp": "2026-04-28T00:51:56+01:00",
          "tree_id": "224c1337b44a953306747098785da6665a376535",
          "url": "https://github.com/fredshone/acteval/commit/40b4357fdd0adc042bab5fc917c1c1404131152f"
        },
        "date": 1777334028225,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[1k]",
            "value": 6.192892074326366,
            "unit": "iter/sec",
            "range": "stddev: 0.018209749605706548",
            "extra": "mean: 161.47544442856696 msec\nrounds: 7"
          },
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[20k]",
            "value": 0.5516931517389322,
            "unit": "iter/sec",
            "range": "stddev: 0.01073943610365039",
            "extra": "mean: 1.8126018001999995 sec\nrounds: 5"
          },
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[100k]",
            "value": 0.10953394891423876,
            "unit": "iter/sec",
            "range": "stddev: 0.08716140560122695",
            "extra": "mean: 9.129589592199995 sec\nrounds: 5"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=256]",
            "value": 81.07943308862593,
            "unit": "iter/sec",
            "range": "stddev: 0.0002605720532192978",
            "extra": "mean: 12.333584016391983 msec\nrounds: 61"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=512]",
            "value": 16.260637356245148,
            "unit": "iter/sec",
            "range": "stddev: 0.001407053440640389",
            "extra": "mean: 61.498204411768306 msec\nrounds: 17"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=1024]",
            "value": 3.6118924868621893,
            "unit": "iter/sec",
            "range": "stddev: 0.007759708427813166",
            "extra": "mean: 276.863169000012 msec\nrounds: 5"
          }
        ]
      }
    ]
  }
}