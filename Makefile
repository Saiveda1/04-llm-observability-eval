# LLM Observability & Evaluation Platform
# ----------------------------------------
# make setup | data | run | test | bench | screenshots | all | clean

PY ?= python3
export MPLBACKEND=Agg
export PYTHONPATH=src

TRACES ?= 5000000
DATA ?= data

.PHONY: all setup data run test bench screenshots clean

all: data run screenshots

setup:
	$(PY) -m pip install -r requirements.txt

## Generate the synthetic trace dataset -> partitioned Parquet
data:
	$(PY) scripts/generate_data.py --traces $(TRACES) --out $(DATA)

## Run analytics + drift + detector + eval harness; write artifacts
run:
	$(PY) scripts/run_pipeline.py --data $(DATA)

## Run the pytest suite (asserts PSI spike, percentile math, cost agg, detector>random)
test:
	$(PY) -m pytest tests/ -q

## Scaling benchmark across dataset sizes -> benchmarks/scaling.csv
bench:
	$(PY) scripts/benchmark_scaling.py

## Render the 4 product-grade PNG screenshots into assets/
screenshots:
	$(PY) scripts/make_screenshots.py

clean:
	rm -rf $(DATA)/traces benchmarks/*.csv benchmarks/*.json assets/*.png
