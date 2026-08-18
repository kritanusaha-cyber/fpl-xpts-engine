PY := .venv/bin/python
export PYTHONPATH := .

.PHONY: all ingest facts config validate snapshot clean

all: ingest config facts validate

ingest:            ## pull 10 seasons of vaastav history -> data/raw
	$(PY) fpl/ingest/vaastav.py data/raw

config:            ## generate scoring YAML from the live API
	$(PY) fpl/ingest/scoring_config.py

facts:             ## build player_gw fact table in DuckDB
	$(PY) fpl/ingest/build_facts.py

validate:          ## reconstruct historical points; must be 100% exact
	$(PY) fpl/backtest/validate_scoring.py

snapshot:          ## capture mutable API state (run daily 22:45 UTC)
	$(PY) fpl/ingest/snapshot.py

clean:
	rm -rf data/interim/* data/features/*

team-match:        ## build team_match fact table
	$(PY) fpl/ingest/build_team_match.py

odds:              ## historical closing odds -> implied goal expectations
	$(PY) fpl/ingest/odds.py

tune-team:         ## tune Dixon-Coles decay on out-of-sample log-loss
	$(PY) fpl/backtest/tune_team_model.py

blend:             ## model vs market vs blend, walk-forward
	$(PY) fpl/backtest/eval_blend.py
