window.BENCHMARK_DATA = {
  "lastUpdate": 1788894744039,
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
      },
      {
        "commit": {
          "author": {
            "email": "26383933+fredshone@users.noreply.github.com",
            "name": "fredshone",
            "username": "fredshone"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "81f2858562f8d984bcb3ac157bb66d191f1b6278",
          "message": "better cli (#2)\n\n* better cli\n\n* formatting",
          "timestamp": "2026-05-04T10:44:37+01:00",
          "tree_id": "54d1b5209eac04e5e344731a7494e37c219f8ca6",
          "url": "https://github.com/fredshone/acteval/commit/81f2858562f8d984bcb3ac157bb66d191f1b6278"
        },
        "date": 1777887993805,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[1k]",
            "value": 5.779047186587668,
            "unit": "iter/sec",
            "range": "stddev: 0.0162407631172735",
            "extra": "mean: 173.0389055000027 msec\nrounds: 6"
          },
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[20k]",
            "value": 0.5095913272850958,
            "unit": "iter/sec",
            "range": "stddev: 0.01885851875814847",
            "extra": "mean: 1.962356787599998 sec\nrounds: 5"
          },
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[100k]",
            "value": 0.10083309570538812,
            "unit": "iter/sec",
            "range": "stddev: 0.02688873531293709",
            "extra": "mean: 9.917378743599993 sec\nrounds: 5"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=256]",
            "value": 86.7288550032469,
            "unit": "iter/sec",
            "range": "stddev: 0.0001707993133760153",
            "extra": "mean: 11.530187962962989 msec\nrounds: 81"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=512]",
            "value": 20.754686085925186,
            "unit": "iter/sec",
            "range": "stddev: 0.0030689174690415218",
            "extra": "mean: 48.18188990476474 msec\nrounds: 21"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=1024]",
            "value": 5.173126087969511,
            "unit": "iter/sec",
            "range": "stddev: 0.00041068375686056967",
            "extra": "mean: 193.30671300001256 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "26383933+fredshone@users.noreply.github.com",
            "name": "Fred Shone",
            "username": "fredshone"
          },
          "committer": {
            "email": "26383933+fredshone@users.noreply.github.com",
            "name": "Fred Shone",
            "username": "fredshone"
          },
          "distinct": true,
          "id": "d9f148e9fbcbd1fb57e9509506c5d3c537392055",
          "message": "fix multiple --split-on args",
          "timestamp": "2026-05-06T14:25:26+01:00",
          "tree_id": "4dbab4cc4703dcee9a75174273d2ff10b9bee76a",
          "url": "https://github.com/fredshone/acteval/commit/d9f148e9fbcbd1fb57e9509506c5d3c537392055"
        },
        "date": 1778074051196,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[1k]",
            "value": 5.375588680940257,
            "unit": "iter/sec",
            "range": "stddev: 0.024356240883401714",
            "extra": "mean: 186.02613766667275 msec\nrounds: 6"
          },
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[20k]",
            "value": 0.5041307085919079,
            "unit": "iter/sec",
            "range": "stddev: 0.03783517958226817",
            "extra": "mean: 1.9836125492000065 sec\nrounds: 5"
          },
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[100k]",
            "value": 0.10103629970424408,
            "unit": "iter/sec",
            "range": "stddev: 0.027072621194272957",
            "extra": "mean: 9.897432931800001 sec\nrounds: 5"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=256]",
            "value": 82.40000073243907,
            "unit": "iter/sec",
            "range": "stddev: 0.0005772485336825253",
            "extra": "mean: 12.135922222223014 msec\nrounds: 72"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=512]",
            "value": 19.79514906846864,
            "unit": "iter/sec",
            "range": "stddev: 0.0013079509728811775",
            "extra": "mean: 50.517427100000134 msec\nrounds: 20"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=1024]",
            "value": 5.024252877544808,
            "unit": "iter/sec",
            "range": "stddev: 0.003210592753521744",
            "extra": "mean: 199.0345677999926 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "26383933+fredshone@users.noreply.github.com",
            "name": "Fred Shone",
            "username": "fredshone"
          },
          "committer": {
            "email": "26383933+fredshone@users.noreply.github.com",
            "name": "Fred Shone",
            "username": "fredshone"
          },
          "distinct": true,
          "id": "c3d1dde392195ba18bd8ffe0c56a33ef2db5a0bb",
          "message": "adds beta metavars",
          "timestamp": "2026-05-06T14:38:15+01:00",
          "tree_id": "293d93a875bef98fd62bd1b3ad8eefdde47f5bd6",
          "url": "https://github.com/fredshone/acteval/commit/c3d1dde392195ba18bd8ffe0c56a33ef2db5a0bb"
        },
        "date": 1778074823730,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[1k]",
            "value": 5.234030839096361,
            "unit": "iter/sec",
            "range": "stddev: 0.022540049346745288",
            "extra": "mean: 191.0573381666675 msec\nrounds: 6"
          },
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[20k]",
            "value": 0.4745288484398099,
            "unit": "iter/sec",
            "range": "stddev: 0.016712386167351573",
            "extra": "mean: 2.1073534375999943 sec\nrounds: 5"
          },
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[100k]",
            "value": 0.09322245568359734,
            "unit": "iter/sec",
            "range": "stddev: 0.08980550683801866",
            "extra": "mean: 10.727029154800004 sec\nrounds: 5"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=256]",
            "value": 85.22173447242135,
            "unit": "iter/sec",
            "range": "stddev: 0.00010382255647289208",
            "extra": "mean: 11.734095840581729 msec\nrounds: 69"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=512]",
            "value": 20.884636089668017,
            "unit": "iter/sec",
            "range": "stddev: 0.00021854240172900454",
            "extra": "mean: 47.88208880952046 msec\nrounds: 21"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=1024]",
            "value": 5.187599834223939,
            "unit": "iter/sec",
            "range": "stddev: 0.0023507974014033532",
            "extra": "mean: 192.76737450000306 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "26383933+fredshone@users.noreply.github.com",
            "name": "Fred Shone",
            "username": "fredshone"
          },
          "committer": {
            "email": "26383933+fredshone@users.noreply.github.com",
            "name": "Fred Shone",
            "username": "fredshone"
          },
          "distinct": true,
          "id": "57cf851eefede340055f84afc06d2dfe0611030b",
          "message": "fixes",
          "timestamp": "2026-05-07T16:15:13+01:00",
          "tree_id": "f0542496c5e62d8bd1994649c9cc4f430403cef9",
          "url": "https://github.com/fredshone/acteval/commit/57cf851eefede340055f84afc06d2dfe0611030b"
        },
        "date": 1778167038319,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[1k]",
            "value": 5.687572637294195,
            "unit": "iter/sec",
            "range": "stddev: 0.020175676715431263",
            "extra": "mean: 175.8219303333135 msec\nrounds: 6"
          },
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[20k]",
            "value": 0.5176620372296282,
            "unit": "iter/sec",
            "range": "stddev: 0.0036886687762446996",
            "extra": "mean: 1.9317622852000114 sec\nrounds: 5"
          },
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[100k]",
            "value": 0.1015666065272285,
            "unit": "iter/sec",
            "range": "stddev: 0.010209225530721533",
            "extra": "mean: 9.84575574779999 sec\nrounds: 5"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=256]",
            "value": 85.49228907258205,
            "unit": "iter/sec",
            "range": "stddev: 0.00011612704230304699",
            "extra": "mean: 11.696961338244323 msec\nrounds: 68"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=512]",
            "value": 20.677666444749956,
            "unit": "iter/sec",
            "range": "stddev: 0.0004985303751398973",
            "extra": "mean: 48.36135657144712 msec\nrounds: 21"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=1024]",
            "value": 5.1764480355673825,
            "unit": "iter/sec",
            "range": "stddev: 0.0011148092717659555",
            "extra": "mean: 193.1826598333449 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "26383933+fredshone@users.noreply.github.com",
            "name": "Fred Shone",
            "username": "fredshone"
          },
          "committer": {
            "email": "26383933+fredshone@users.noreply.github.com",
            "name": "Fred Shone",
            "username": "fredshone"
          },
          "distinct": true,
          "id": "5e5ec1da613374ff941f94aff2fad1ebf944a1ec",
          "message": "makes novel feasibility optional, default off",
          "timestamp": "2026-05-11T14:03:06+01:00",
          "tree_id": "401605eb973b336db2e904207a59ec904bb77617",
          "url": "https://github.com/fredshone/acteval/commit/5e5ec1da613374ff941f94aff2fad1ebf944a1ec"
        },
        "date": 1778504703061,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[1k]",
            "value": 5.488712138737863,
            "unit": "iter/sec",
            "range": "stddev: 0.022932130291888547",
            "extra": "mean: 182.19210166666736 msec\nrounds: 6"
          },
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[20k]",
            "value": 0.5139169131778802,
            "unit": "iter/sec",
            "range": "stddev: 0.005977565318111906",
            "extra": "mean: 1.94583983199999 sec\nrounds: 5"
          },
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[100k]",
            "value": 0.10145759034795579,
            "unit": "iter/sec",
            "range": "stddev: 0.048438149814392824",
            "extra": "mean: 9.856335012199986 sec\nrounds: 5"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=256]",
            "value": 81.31353246963059,
            "unit": "iter/sec",
            "range": "stddev: 0.00012688911763426958",
            "extra": "mean: 12.298075973682307 msec\nrounds: 76"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=512]",
            "value": 20.60659852094643,
            "unit": "iter/sec",
            "range": "stddev: 0.0002064567323641016",
            "extra": "mean: 48.52814495238059 msec\nrounds: 21"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=1024]",
            "value": 5.140839400112869,
            "unit": "iter/sec",
            "range": "stddev: 0.0007927382030342727",
            "extra": "mean: 194.52076250000042 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "26383933+fredshone@users.noreply.github.com",
            "name": "Fred Shone",
            "username": "fredshone"
          },
          "committer": {
            "email": "26383933+fredshone@users.noreply.github.com",
            "name": "Fred Shone",
            "username": "fredshone"
          },
          "distinct": true,
          "id": "cd736219f543d1d2f2a22969cf3bad29fc5af2ba",
          "message": "refactor creativity and feasibility configuration",
          "timestamp": "2026-05-11T14:38:08+01:00",
          "tree_id": "d0ca21f954c2d825af0536d07ab32659dcc59025",
          "url": "https://github.com/fredshone/acteval/commit/cd736219f543d1d2f2a22969cf3bad29fc5af2ba"
        },
        "date": 1778506811942,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[1k]",
            "value": 6.342609454327608,
            "unit": "iter/sec",
            "range": "stddev: 0.018835829099594626",
            "extra": "mean: 157.66381442856974 msec\nrounds: 7"
          },
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[20k]",
            "value": 0.5306755579273315,
            "unit": "iter/sec",
            "range": "stddev: 0.011126613659161599",
            "extra": "mean: 1.8843905378000017 sec\nrounds: 5"
          },
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[100k]",
            "value": 0.09991525456080862,
            "unit": "iter/sec",
            "range": "stddev: 0.06098189868708349",
            "extra": "mean: 10.008481731800003 sec\nrounds: 5"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=256]",
            "value": 78.41996404425085,
            "unit": "iter/sec",
            "range": "stddev: 0.00020757025545626415",
            "extra": "mean: 12.751854864862214 msec\nrounds: 74"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=512]",
            "value": 20.091341053350057,
            "unit": "iter/sec",
            "range": "stddev: 0.00017285549986926702",
            "extra": "mean: 49.772685523809706 msec\nrounds: 21"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=1024]",
            "value": 5.140662768703895,
            "unit": "iter/sec",
            "range": "stddev: 0.0009980583886627405",
            "extra": "mean: 194.52744616665996 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "26383933+fredshone@users.noreply.github.com",
            "name": "Fred Shone",
            "username": "fredshone"
          },
          "committer": {
            "email": "26383933+fredshone@users.noreply.github.com",
            "name": "Fred Shone",
            "username": "fredshone"
          },
          "distinct": true,
          "id": "ec3484ce5502a46744124ea5d428c23ea947cba3",
          "message": "adds progress bars and better outputs",
          "timestamp": "2026-05-15T13:46:20+01:00",
          "tree_id": "4d86aa3a977d8bae2a626bbf0c7e77979f8204e1",
          "url": "https://github.com/fredshone/acteval/commit/ec3484ce5502a46744124ea5d428c23ea947cba3"
        },
        "date": 1778849293253,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[1k]",
            "value": 5.490419702149355,
            "unit": "iter/sec",
            "range": "stddev: 0.02102496274763659",
            "extra": "mean: 182.1354384999978 msec\nrounds: 6"
          },
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[20k]",
            "value": 0.5137579617039214,
            "unit": "iter/sec",
            "range": "stddev: 0.004619656536897073",
            "extra": "mean: 1.9464418550000004 sec\nrounds: 5"
          },
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[100k]",
            "value": 0.10177545280999768,
            "unit": "iter/sec",
            "range": "stddev: 0.03436930534911242",
            "extra": "mean: 9.82555196159999 sec\nrounds: 5"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=256]",
            "value": 84.17585426874618,
            "unit": "iter/sec",
            "range": "stddev: 0.00010431489165110307",
            "extra": "mean: 11.879891314287404 msec\nrounds: 70"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=512]",
            "value": 20.65552248114273,
            "unit": "iter/sec",
            "range": "stddev: 0.0004160214577226249",
            "extra": "mean: 48.413202857150715 msec\nrounds: 21"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=1024]",
            "value": 5.137912898447973,
            "unit": "iter/sec",
            "range": "stddev: 0.0006711505309328888",
            "extra": "mean: 194.63155949998168 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "26383933+fredshone@users.noreply.github.com",
            "name": "Fred Shone",
            "username": "fredshone"
          },
          "committer": {
            "email": "26383933+fredshone@users.noreply.github.com",
            "name": "Fred Shone",
            "username": "fredshone"
          },
          "distinct": true,
          "id": "44dc7136f7073d512859298086066aecd92b0d2d",
          "message": "fixes discover bug on empty dataframes",
          "timestamp": "2026-05-27T11:30:45+01:00",
          "tree_id": "154bec4a34ce6da3b4c62d7bf751912c76477a66",
          "url": "https://github.com/fredshone/acteval/commit/44dc7136f7073d512859298086066aecd92b0d2d"
        },
        "date": 1779877961644,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[1k]",
            "value": 6.4584402603081426,
            "unit": "iter/sec",
            "range": "stddev: 0.020477624102072108",
            "extra": "mean: 154.83614614285963 msec\nrounds: 7"
          },
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[20k]",
            "value": 0.5319524853409867,
            "unit": "iter/sec",
            "range": "stddev: 0.007575378435326393",
            "extra": "mean: 1.8798671451999893 sec\nrounds: 5"
          },
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[100k]",
            "value": 0.1006128687332183,
            "unit": "iter/sec",
            "range": "stddev: 0.05246035134482107",
            "extra": "mean: 9.939086446800024 sec\nrounds: 5"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=256]",
            "value": 80.57960471651728,
            "unit": "iter/sec",
            "range": "stddev: 0.00011660071253213113",
            "extra": "mean: 12.410088179484692 msec\nrounds: 78"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=512]",
            "value": 20.426640620790202,
            "unit": "iter/sec",
            "range": "stddev: 0.0006361881891324911",
            "extra": "mean: 48.95567600000764 msec\nrounds: 21"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=1024]",
            "value": 5.175806386947387,
            "unit": "iter/sec",
            "range": "stddev: 0.0004005678783436815",
            "extra": "mean: 193.206608833331 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "26383933+fredshone@users.noreply.github.com",
            "name": "Fred Shone",
            "username": "fredshone"
          },
          "committer": {
            "email": "26383933+fredshone@users.noreply.github.com",
            "name": "Fred Shone",
            "username": "fredshone"
          },
          "distinct": true,
          "id": "66cf3443f58ee3d69739946cb3a2f33e6737d10e",
          "message": "cmap bug",
          "timestamp": "2026-06-02T17:50:52+01:00",
          "tree_id": "78792419a0517853b4246387ade7a2c692eb7967",
          "url": "https://github.com/fredshone/acteval/commit/66cf3443f58ee3d69739946cb3a2f33e6737d10e"
        },
        "date": 1780419169862,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[1k]",
            "value": 5.421959238001455,
            "unit": "iter/sec",
            "range": "stddev: 0.022459114556733942",
            "extra": "mean: 184.43517483333238 msec\nrounds: 6"
          },
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[20k]",
            "value": 0.5057237993068814,
            "unit": "iter/sec",
            "range": "stddev: 0.02873754486550485",
            "extra": "mean: 1.9773639314000007 sec\nrounds: 5"
          },
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[100k]",
            "value": 0.10192713341885828,
            "unit": "iter/sec",
            "range": "stddev: 0.02335840007361886",
            "extra": "mean: 9.810930283800001 sec\nrounds: 5"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=256]",
            "value": 83.79299706006523,
            "unit": "iter/sec",
            "range": "stddev: 0.00014747528335997298",
            "extra": "mean: 11.934171530863983 msec\nrounds: 81"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=512]",
            "value": 20.176069409644708,
            "unit": "iter/sec",
            "range": "stddev: 0.0002709160674745915",
            "extra": "mean: 49.56366771428596 msec\nrounds: 21"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=1024]",
            "value": 5.071308029877915,
            "unit": "iter/sec",
            "range": "stddev: 0.00129568982669554",
            "extra": "mean: 197.18778550000118 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "26383933+fredshone@users.noreply.github.com",
            "name": "Fred Shone",
            "username": "fredshone"
          },
          "committer": {
            "email": "26383933+fredshone@users.noreply.github.com",
            "name": "Fred Shone",
            "username": "fredshone"
          },
          "distinct": true,
          "id": "f95d59cd2dc85fb9a02616967a8ec30f763557c1",
          "message": "removes vebose printing",
          "timestamp": "2026-07-22T10:43:09+01:00",
          "tree_id": "1474e97306ebafe6baf67f71ce8f414d5ed12a7f",
          "url": "https://github.com/fredshone/acteval/commit/f95d59cd2dc85fb9a02616967a8ec30f763557c1"
        },
        "date": 1784713502673,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[1k]",
            "value": 6.590913736005317,
            "unit": "iter/sec",
            "range": "stddev: 0.023140941863332078",
            "extra": "mean: 151.72403100000054 msec\nrounds: 7"
          },
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[20k]",
            "value": 0.5189335152474565,
            "unit": "iter/sec",
            "range": "stddev: 0.046394223834054234",
            "extra": "mean: 1.9270291291999981 sec\nrounds: 5"
          },
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[100k]",
            "value": 0.09963069132857888,
            "unit": "iter/sec",
            "range": "stddev: 0.018798096423946437",
            "extra": "mean: 10.037067761600003 sec\nrounds: 5"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=256]",
            "value": 78.20949791927389,
            "unit": "iter/sec",
            "range": "stddev: 0.00011372211992110627",
            "extra": "mean: 12.786170818180905 msec\nrounds: 66"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=512]",
            "value": 19.96918184123268,
            "unit": "iter/sec",
            "range": "stddev: 0.00013303170745750505",
            "extra": "mean: 50.07716430000073 msec\nrounds: 20"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=1024]",
            "value": 5.141878411276376,
            "unit": "iter/sec",
            "range": "stddev: 0.0004947962718383367",
            "extra": "mean: 194.4814560000007 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "26383933+fredshone@users.noreply.github.com",
            "name": "Fred Shone",
            "username": "fredshone"
          },
          "committer": {
            "email": "26383933+fredshone@users.noreply.github.com",
            "name": "Fred Shone",
            "username": "fredshone"
          },
          "distinct": true,
          "id": "64325ce5671cb2f0e1e25f0747e573e6d56c364d",
          "message": "api tweaks",
          "timestamp": "2026-08-20T17:38:18+01:00",
          "tree_id": "7add0b4677ca0507fb8e4f10850cac438d5de910",
          "url": "https://github.com/fredshone/acteval/commit/64325ce5671cb2f0e1e25f0747e573e6d56c364d"
        },
        "date": 1787244009693,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[1k]",
            "value": 5.64482927269373,
            "unit": "iter/sec",
            "range": "stddev: 0.023792578669709533",
            "extra": "mean: 177.15327633333308 msec\nrounds: 6"
          },
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[20k]",
            "value": 0.5344994300726114,
            "unit": "iter/sec",
            "range": "stddev: 0.009503138026611175",
            "extra": "mean: 1.8709093850000003 sec\nrounds: 5"
          },
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[100k]",
            "value": 0.10501473533529061,
            "unit": "iter/sec",
            "range": "stddev: 0.026399349322663967",
            "extra": "mean: 9.522473173 sec\nrounds: 5"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=256]",
            "value": 86.61623062903864,
            "unit": "iter/sec",
            "range": "stddev: 0.00008814889427632923",
            "extra": "mean: 11.545180305557464 msec\nrounds: 72"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=512]",
            "value": 20.650618687374802,
            "unit": "iter/sec",
            "range": "stddev: 0.00012105055788817502",
            "extra": "mean: 48.424699285710574 msec\nrounds: 21"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=1024]",
            "value": 5.172072596824636,
            "unit": "iter/sec",
            "range": "stddev: 0.00038104517994431145",
            "extra": "mean: 193.34608733333405 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "26383933+fredshone@users.noreply.github.com",
            "name": "fredshone",
            "username": "fredshone"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e4dda7e80a8c9144f6cb146f60c5e285f1a8570d",
          "message": "fixes unfinished prog bars (#7)\n\n* fixes unfinished prog bars\n\n* fixes unfinished prog bars\n\n* lints\n\n* format",
          "timestamp": "2026-09-08T20:10:37+01:00",
          "tree_id": "09e9c892954a89fe3952fb7bf8def3476a3c63bc",
          "url": "https://github.com/fredshone/acteval/commit/e4dda7e80a8c9144f6cb146f60c5e285f1a8570d"
        },
        "date": 1788894743077,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[1k]",
            "value": 5.814576076217472,
            "unit": "iter/sec",
            "range": "stddev: 0.018155628858509704",
            "extra": "mean: 171.98158333333305 msec\nrounds: 6"
          },
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[20k]",
            "value": 0.538805230514105,
            "unit": "iter/sec",
            "range": "stddev: 0.009755322870144016",
            "extra": "mean: 1.8559582263999972 sec\nrounds: 5"
          },
          {
            "name": "tests/test_bench_evaluate.py::test_bench_compare[100k]",
            "value": 0.10505107600985192,
            "unit": "iter/sec",
            "range": "stddev: 0.007350532959892953",
            "extra": "mean: 9.519179031599998 sec\nrounds: 5"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=256]",
            "value": 87.83914599306374,
            "unit": "iter/sec",
            "range": "stddev: 0.00009984695895486341",
            "extra": "mean: 11.38444583783824 msec\nrounds: 74"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=512]",
            "value": 21.40402854094097,
            "unit": "iter/sec",
            "range": "stddev: 0.0001410044356044548",
            "extra": "mean: 46.72017690909123 msec\nrounds: 22"
          },
          {
            "name": "tests/test_bench_pairwise.py::test_pairwise[N=1024]",
            "value": 5.314131133802756,
            "unit": "iter/sec",
            "range": "stddev: 0.0006252063682162098",
            "extra": "mean: 188.17751666666283 msec\nrounds: 6"
          }
        ]
      }
    ]
  }
}