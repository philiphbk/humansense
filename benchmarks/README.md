# Benchmarks

Reproduces every number in [paper/REPORT.md](../paper/REPORT.md). Nothing here is downloaded
automatically: fetch the datasets yourself (check their licenses, see `docs/LICENSING.md`) and
point `HS_DATA` at them. `bench_latency.py` is the one exception -- it only needs `assets/demo.mp4`,
already in the repo, so it needs no `HS_DATA` setup.

```text
$HS_DATA/
  urfd/      fall-01-cam0-rgb.zip … fall-30-…, adl-01-cam0-rgb.zip … adl-40-…,
             urfall-cam0-falls.csv, urfall-cam0-adls.csv      (https://fenix.ur.edu.pl/mkepski/ds/uf.html)
  coco/      val2017.zip, annotations_trainval2017.zip         (https://cocodataset.org/#download)
  fer2013/   privateTest.parquet   (HF dataset Aaryan333/fer2013_train_publicTest_privateTest @ da8d164)
```

```bash
pip install -e ".[bench]"
export HS_DATA=/path/to/data
python benchmarks/bench_falls.py     --backend mediapipe   # also: yolo
python benchmarks/bench_detection.py --backend mediapipe
python benchmarks/bench_pose.py
python benchmarks/bench_emotion.py
python benchmarks/bench_latency.py                          # run on an otherwise idle machine
python benchmarks/make_tables.py                            # results/*.json -> paper/tables.md
```

`bench_falls.py --parity odd|even` restricts the run to sequences numbered odd/even (`fall-01`,
`fall-03`, ... for "odd"). Tune any detection threshold against the odd-numbered sequences only,
then report its numbers from a single run on the even-numbered ones -- never tune and report on the
same half; see [paper/REPORT.md](../paper/REPORT.md) for why and what came out of it.

Each script writes `results/<name>.json` including the software/hardware environment. Timing numbers
are only meaningful on an idle machine; accuracy numbers are deterministic.
