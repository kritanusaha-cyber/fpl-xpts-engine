// Gentle, resumable SofaScore heatmap collector.
//
// Paste into the console on https://api.sofascore.com and call start().
//
// Two things this fixes from the first attempt:
//
//   PERSISTENCE. The earlier version accumulated into a window variable and lost
//   154 players when the tab closed. This writes to localStorage after every
//   match, so a closed tab, a reload, or a rate-limit pause costs at most one
//   match. resume() picks up exactly where it stopped.
//
//   REQUEST PACING. The earlier version fired ~150 heatmap requests in a burst
//   and triggered SofaScore's bot detection ("reason":"challenge"). This runs
//   strictly serially with a delay, and STOPS ON ITS OWN the moment it sees a
//   403 rather than retrying into the block. Slow is the point: a season takes
//   hours, spread over sessions, which is the pattern that does not get flagged.
//
// Nothing here attempts to bypass a challenge. If one appears, it halts and says
// so; the correct response is to wait, not to push.

(() => {
  const KEY = 'sofa_heatmaps_v1';
  const NX = 6, NY = 5;
  const DELAY_MS = 1500;          // between player requests
  const MATCH_PAUSE_MS = 4000;    // between matches
  const ROUNDS = [2, 6, 10, 14, 18, 22, 26, 30, 34, 38];
  const SEASON = 76986, TOURNAMENT = 17;

  const load = () => { try { return JSON.parse(localStorage.getItem(KEY)) ||
      {agg:{}, done:[], stoppedReason:null}; } catch(e){ return {agg:{}, done:[], stoppedReason:null}; } };
  const save = s => localStorage.setItem(KEY, JSON.stringify(s));
  const sleep = ms => new Promise(r => setTimeout(r, ms));

  let halted = false;
  async function get(url) {
    const r = await fetch(url);
    if (r.status === 403) { halted = true; return {__blocked:true}; }
    return r.ok ? r.json() : null;
  }

  async function start() {
    const S = load();
    const done = new Set(S.done);
    halted = false;
    console.log(`resuming: ${done.size} matches already collected, ` +
                `${Object.keys(S.agg).length} players`);

    for (const rd of ROUNDS) {
      if (halted) break;
      const ev = await get(`/api/v1/unique-tournament/${TOURNAMENT}/season/${SEASON}/events/round/${rd}`);
      if (halted || !ev) break;
      const ms = (ev.events || []).filter(e => e.status && e.status.type === 'finished');
      for (const m of ms) {
        if (halted) break;
        if (done.has(m.id)) continue;
        const lu = await get(`/api/v1/event/${m.id}/lineups`);
        if (halted) break;
        if (!lu) { done.add(m.id); continue; }

        for (const side of ['home','away']) {
          for (const p of ((lu[side] && lu[side].players) || [])) {
            if (halted) break;
            const st = p.statistics || {};
            if (!st.minutesPlayed) continue;
            const pid = p.player && p.player.id;
            if (!pid) continue;
            const hm = await get(`/api/v1/event/${m.id}/player/${pid}/heatmap`);
            if (halted) break;
            await sleep(DELAY_MS);
            const rec = S.agg[pid] || (S.agg[pid] = {name: p.player.name, mins: 0,
              matches: 0, grid: new Array(NX*NY).fill(0), pts: 0, box: 0,
              oppHalfPass: 0, keyPass: 0, bigCh: 0});
            rec.mins += st.minutesPlayed || 0;
            rec.matches++;
            rec.oppHalfPass += st.accurateOppositionHalfPasses || 0;
            rec.keyPass += st.keyPass || 0;
            rec.bigCh += st.bigChanceCreated || 0;
            for (const q of ((hm && hm.heatmap) || [])) {
              const gx = Math.min(NX-1, Math.floor(q.x/100*NX));
              const gy = Math.min(NY-1, Math.floor(q.y/100*NY));
              rec.grid[gy*NX+gx]++; rec.pts++;
              if (q.x >= 83 && q.y >= 21 && q.y <= 79) rec.box++;
            }
          }
        }
        if (!halted) {
          done.add(m.id);
          S.done = [...done];
          save(S);                       // checkpoint after every match
          console.log(`round ${rd}: ${done.size} matches, ` +
                      `${Object.keys(S.agg).length} players`);
          await sleep(MATCH_PAUSE_MS);
        }
      }
    }
    S.stoppedReason = halted ? 'blocked (403 challenge) - wait, do not retry immediately' : 'complete';
    S.done = [...done];
    save(S);
    console.log(S.stoppedReason);
    return S.stoppedReason;
  }

  function status(){ const S=load();
    return {matches:S.done.length, players:Object.keys(S.agg).length,
            points:Object.values(S.agg).reduce((a,b)=>a+b.pts,0), reason:S.stoppedReason}; }
  function exportJson(){ const S=load();
    const blob=new Blob([JSON.stringify(S)],{type:'application/json'});
    const a=document.createElement('a'); a.href=URL.createObjectURL(blob);
    a.download='sofa_heatmaps.json'; a.click(); }

  window.sofa = {start, status, exportJson, reset:()=>localStorage.removeItem(KEY)};
  console.log('ready: sofa.start() / sofa.status() / sofa.exportJson()');
})();
