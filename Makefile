PYTHON ?= python3
HF_HOME ?= data/hf_cache
EXP1_SMOKE_CONFIG ?= configs/exp1_smoke.yaml
EXP1_MVP_CONFIG ?= configs/exp1_mvp.yaml
EXP1_BOUNDED_CONFIG ?= configs/exp1_bounded.yaml
EXP1_MAIN_CONFIG ?= configs/exp1_main.yaml
EXP1_DENSE_CONFIG ?= configs/exp1_dense.yaml

# Backward-compat alias (older docs reference EXP1_CONFIG).
EXP1_CONFIG ?= $(EXP1_SMOKE_CONFIG)

.PHONY: exp1-smoke exp1-smoke-post exp1-download-objaverse \
	exp1-bounded-estimate exp1-bounded-plan exp1-mvp-plan exp1-full-plan \
	exp1-main-prepare exp1-main-post exp1-dense-prepare exp1-dense-post \
	exp1-results exp1-figures exp1-rerender-analysis \
	exp1-bounded-dense-patches exp1-bounded-dense-probes exp1-bounded-dense

exp1-smoke:
	HF_HOME=$(HF_HOME) PYTHONPATH=. $(PYTHON) scripts/run_exp1_pipeline.py --config $(EXP1_SMOKE_CONFIG) --stages smoke_prepare

exp1-smoke-post:
	HF_HOME=$(HF_HOME) PYTHONPATH=. $(PYTHON) scripts/run_exp1_pipeline.py --config $(EXP1_SMOKE_CONFIG) --stages post_render ml

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

# Canonical pipeline stages for the rerender analysis
exp1-main-prepare:
	HF_HOME=$(HF_HOME) PYTHONPATH=. $(PYTHON) scripts/run_exp1_pipeline.py --config $(EXP1_MAIN_CONFIG) --stages prepare

exp1-main-post:
	HF_HOME=$(HF_HOME) PYTHONPATH=. $(PYTHON) scripts/run_exp1_pipeline.py --config $(EXP1_MAIN_CONFIG) --stages post_render

exp1-dense-prepare:
	HF_HOME=$(HF_HOME) PYTHONPATH=. $(PYTHON) scripts/run_exp1_pipeline.py --config $(EXP1_DENSE_CONFIG) --stages prepare

exp1-dense-post:
	HF_HOME=$(HF_HOME) PYTHONPATH=. $(PYTHON) scripts/run_exp1_pipeline.py --config $(EXP1_DENSE_CONFIG) --stages post_render

exp1-results:
	HF_HOME=$(HF_HOME) PYTHONPATH=. $(PYTHON) scripts/aggregate_exp1_results.py --config $(EXP1_MAIN_CONFIG)
	HF_HOME=$(HF_HOME) PYTHONPATH=. $(PYTHON) scripts/aggregate_exp1_results.py --config $(EXP1_DENSE_CONFIG)

exp1-figures:
	HF_HOME=$(HF_HOME) PYTHONPATH=. $(PYTHON) scripts/make_exp1_figures.py --config $(EXP1_MAIN_CONFIG)
	HF_HOME=$(HF_HOME) PYTHONPATH=. $(PYTHON) scripts/make_exp1_figures.py --config $(EXP1_DENSE_CONFIG)

exp1-rerender-analysis:
	HF_HOME=$(HF_HOME) PYTHONPATH=. $(PYTHON) scripts/analysis/plot_exp1_rerender_summary.py --main-dir outputs/exp1_main --dense-dir outputs/exp1_dense
	HF_HOME=$(HF_HOME) PYTHONPATH=. $(PYTHON) scripts/analysis/plot_exp1_dense_reliability.py --dense-dir outputs/exp1_dense
	HF_HOME=$(HF_HOME) PYTHONPATH=. $(PYTHON) scripts/analysis/plot_exp1_state_space_breakdown.py --main-dir outputs/exp1_main --dense-dir outputs/exp1_dense
	HF_HOME=$(HF_HOME) PYTHONPATH=. $(PYTHON) scripts/analysis/plot_exp1_qualitative_examples.py --dense-dir outputs/exp1_dense
	HF_HOME=$(HF_HOME) PYTHONPATH=. $(PYTHON) scripts/analyze_exp1_dense_probes.py

exp1-bounded-dense-patches:
	HF_HOME=$(HF_HOME) PYTHONPATH=. $(PYTHON) scripts/extract_exp1_patch_features.py --config $(EXP1_BOUNDED_CONFIG)

exp1-bounded-dense-probes:
	HF_HOME=$(HF_HOME) PYTHONPATH=. $(PYTHON) scripts/train_all_dense_depth_probes.py --config $(EXP1_BOUNDED_CONFIG)
	HF_HOME=$(HF_HOME) PYTHONPATH=. $(PYTHON) scripts/train_all_dense_surface_normal_probes.py --config $(EXP1_BOUNDED_CONFIG)

exp1-bounded-dense:
	HF_HOME=$(HF_HOME) PYTHONPATH=. $(PYTHON) scripts/run_exp1_pipeline.py --config $(EXP1_BOUNDED_CONFIG) --stages dense
