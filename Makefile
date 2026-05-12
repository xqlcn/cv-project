PYTHON ?= python3
EXP1_CONFIG ?= configs/exp1_smoke.yaml
EXP1_MVP_CONFIG ?= configs/exp1_mvp.yaml

.PHONY: exp1-smoke exp1-smoke-post exp1-mvp-plan exp1-results exp1-figures

exp1-smoke:
	PYTHONPATH=. $(PYTHON) scripts/run_exp1_pipeline.py --config $(EXP1_CONFIG) --stages smoke_prepare

exp1-smoke-post:
	PYTHONPATH=. $(PYTHON) scripts/run_exp1_pipeline.py --config $(EXP1_CONFIG) --stages post_render ml

exp1-mvp-plan:
	PYTHONPATH=. $(PYTHON) scripts/run_exp1_pipeline.py --config $(EXP1_MVP_CONFIG) --stages render_plan render_chunks

exp1-results:
	PYTHONPATH=. $(PYTHON) scripts/aggregate_exp1_results.py --config $(EXP1_CONFIG)

exp1-figures:
	PYTHONPATH=. $(PYTHON) scripts/make_exp1_figures.py --config $(EXP1_CONFIG)
