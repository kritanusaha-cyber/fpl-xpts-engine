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


install-snapshot:  ## install the daily snapshotter as a LaunchAgent
	@sed 's|__FPL_ROOT__|$(CURDIR)|g' scripts/com.fpl.snapshot.plist \
	  > ~/Library/LaunchAgents/com.fpl.snapshot.plist
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

live:              ## pull finished gameweeks of the running season into player_gw
	$(PY) fpl/ingest/live.py

coldstart:         ## rebuild 2026/27 priors, blending in played gameweeks
	$(PY) -c "from fpl.models.coldstart import build; \
	  build().to_parquet('data/features/coldstart_2026_27.parquet', index=False)"

roles:             ## k-means role clusters over the horizon projection
	$(PY) -c "import pandas as pd; from fpl.models.roles import assign_all; \
	  assign_all(pd.read_parquet('data/features/horizon_projection.parquet')) \
	    .to_parquet('data/features/horizon_roles.parquet', index=False)"

# Order matters and is not obvious. `live` must precede `coldstart`, which
# blends played gameweeks into the priors; `coldstart` must precede `horizon`,
# which simulates from them; and `roles` must sit between `horizon` and
# `dashboard`, because the clusters are fitted on the projection the dashboard
# then displays. Running these out of order produces a dashboard that is stale
# in a way nothing errors on.
refresh:           ## full weekly refresh: data -> models -> dashboard
	$(MAKE) snapshot live facts team-match coldstart horizon roles log dashboard dist

watch:             ## poll the FPL feed; rebuild only if something changed
	$(PY) fpl/ingest/watch.py

watch-deploy:      ## same, and publish when it rebuilds
	$(PY) fpl/ingest/watch.py --deploy

watch-check:       ## report whether anything changed, without rebuilding
	$(PY) fpl/ingest/watch.py --check-only

install-watch:     ## run the watcher every 20 minutes via launchd
	@sed 's|__FPL_ROOT__|$(CURDIR)|g' scripts/com.fpl.watch.plist \
	  > $(HOME)/Library/LaunchAgents/com.fpl.watch.plist
	@launchctl unload $(HOME)/Library/LaunchAgents/com.fpl.watch.plist 2>/dev/null || true
	@launchctl load $(HOME)/Library/LaunchAgents/com.fpl.watch.plist
	@echo "watcher installed -- polls every 20 min, deploys on change"
	@echo "  log:      tail -f $(CURDIR)/data/raw/watch.log"
	@echo "  stop:     make uninstall-watch"

uninstall-watch:
	@launchctl unload $(HOME)/Library/LaunchAgents/com.fpl.watch.plist 2>/dev/null || true
	@rm -f $(HOME)/Library/LaunchAgents/com.fpl.watch.plist
	@echo "watcher removed"

log:               ## record projection vs outcome, so calibration can refit
	$(PY) scripts/log_projection.py

deploy:            ## rebuild the dashboard and publish to GitHub Pages
	$(MAKE) dashboard
	$(PY) scripts/build_findings.py
	@git add -A && git commit -q -m "Refresh projections" || echo "  (nothing changed)"
	@git push -q origin main && echo "pushed -- Pages rebuilds in ~30s"
	@echo "https://kritanusaha-cyber.github.io/fpl-xpts-engine/"

fotmob:            ## ingest FotMob shotmaps (xGOT, penalty + set-piece tags)
	$(PY) fpl/ingest/fotmob.py

xgot:              ## test whether xGOT predicts anything before using it
	$(PY) -c "import pandas as pd; from fpl.backtest.eval_xgot import evaluate, evaluate_keepers; \
	s=pd.read_parquet('data/raw/fotmob/shots_2025_2026.parquet'); evaluate(s); print(); evaluate_keepers(s)"

zonal:             ## ingest FotMob per-match player stats + shot coordinates
	$(PY) fpl/ingest/fotmob_zonal.py

eval-zonal:        ## test whether territorial features predict
	$(PY) -c "import pandas as pd; from fpl.backtest.eval_zonal import evaluate; \
	evaluate(pd.read_parquet('data/raw/fotmob/player_match_stats.parquet'), \
	         pd.read_parquet('data/raw/fotmob/shots_zoned.parquet'))"
