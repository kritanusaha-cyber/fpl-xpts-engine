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

dashboard:         ## export combined JSON and build the dashboard
	$(PY) fpl/export_combined.py
	$(PY) scripts/build_combined.py

foreign:           ## pull Big 5 output for incoming transfers
	$(PY) fpl/ingest/fbref_foreign.py

transfers:         ## backtest cold-start priors for new signings
	$(PY) fpl/backtest/eval_transfers.py

horizon:           ## multi-gameweek simulation (H=6)
	$(PY) fpl/predict_horizon.py

# The standalone simulation page is superseded by `dashboard`, which merges the
# charts and the value table into one payload. Kept for regenerating it alone.
simulation:        ## (superseded) standalone simulation page
	$(PY) fpl/export_simulation.py
	$(PY) scripts/build_simulation.py

PORT ?= 8733
serve:             ## serve the dashboards at http://localhost:$(PORT)
	@echo "serving $(CURDIR) at http://localhost:$(PORT)   (ctrl-C to stop)"
	@python3 -m http.server $(PORT) --bind 127.0.0.1 --directory $(CURDIR)

dist:              ## collect the static site into dist/ for any static host
	@rm -rf dist && mkdir -p dist
	@cp index.html dashboard.html simulation.html dist/
	@echo "dist/ ready ($$(du -sh dist | cut -f1)) -- self-contained, no server needed"

refresh:           ## full weekly refresh: data -> models -> dashboard
	$(MAKE) snapshot facts team-match horizon dashboard dist
