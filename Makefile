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

attacking:         ## evaluate share-based attacking returns
	$(PY) fpl/backtest/eval_attacking.py

bonus:             ## evaluate BPS -> bonus recovery
	$(PY) fpl/backtest/eval_bonus.py

defcon:            ## walk-forward DefCon threshold evaluation
	$(PY) fpl/backtest/eval_defcon.py

gw1:               ## end-to-end GW1 projection + optimal squad
	$(PY) fpl/predict_gw1.py

coldstart:         ## build 2026/27 cold-start priors
	$(PY) fpl/models/coldstart.py

install-snapshot:  ## install the daily snapshotter as a LaunchAgent
	cp scripts/com.fpl.snapshot.plist ~/Library/LaunchAgents/
	launchctl unload ~/Library/LaunchAgents/com.fpl.snapshot.plist 2>/dev/null || true
	launchctl load  ~/Library/LaunchAgents/com.fpl.snapshot.plist
	@echo "installed. verify with: launchctl list | grep fpl"

uninstall-snapshot:
	launchctl unload ~/Library/LaunchAgents/com.fpl.snapshot.plist 2>/dev/null || true
	rm -f ~/Library/LaunchAgents/com.fpl.snapshot.plist

fbref:             ## pull penalty attempts from FBref (via soccerdata)
	$(PY) fpl/ingest/fbref.py
