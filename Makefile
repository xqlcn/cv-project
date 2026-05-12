PYTHON ?= python3
HF_HOME ?= data/hf_cache
EXP1_CONFIG ?= configs/exp1_smoke.yaml
EXP1_MVP_CONFIG ?= configs/exp1_mvp.yaml
EXP1_BOUNDED_CONFIG ?= configs/exp1_bounded.yaml

.PHONY: exp1-smoke exp1-smoke-post exp1-download-objaverse exp1-bounded-estimate exp1-bounded-plan exp1-mvp-plan exp1-full-plan exp1-results exp1-figures

exp1-smoke:
	HF_HOME=$(HF_HOME) PYTHONPATH=. $(PYTHON) scripts/run_exp1_pipeline.py --config $(EXP1_CONFIG) --stages smoke_prepare

exp1-smoke-post:
	HF_HOME=$(HF_HOME) PYTHONPATH=. $(PYTHON) scripts/run_exp1_pipeline.py --config $(EXP1_CONFIG) --stages post_render ml

exp1-download-objaverse:
	HF_HOME=$(HF_HOME) PYTHONPATH=. $(PYTHON) scripts/download_objaverse_assets.py --config $(EXP1_BOUNDED_CONFIG)

exp1-bounded-estimate:
	HF_HOME=$(HF_HOME) PYTHONPATH=. $(PYTHON) scripts/estimate_exp1_run.py --config $(EXP1_BOUNDED_CONFIG)

exp1-bounded-plan:
	HF_HOME=$(HF_HOME) PYTHONPATH=. $(PYTHON) scripts/run_exp1_pipeline.py --config $(EXP1_BOUNDED_CONFIG) --stages prepare

exp1-mvp-plan:
	HF_HOME=$(HF_HOME) PYTHONPATH=. $(PYTHON) scripts/run_exp1_pipeline.py --config $(EXP1_MVP_CONFIG) --stages prepare

exp1-full-plan:
	HF_HOME=$(HF_HOME) PYTHONPATH=. $(PYTHON) scripts/run_exp1_pipeline.py --config configs/exp1_full.yaml --stages prepare

exp1-results:
	HF_HOME=$(HF_HOME) PYTHONPATH=. $(PYTHON) scripts/aggregate_exp1_results.py --config $(EXP1_CONFIG)

exp1-figures:
	HF_HOME=$(HF_HOME) PYTHONPATH=. $(PYTHON) scripts/make_exp1_figures.py --config $(EXP1_CONFIG)
