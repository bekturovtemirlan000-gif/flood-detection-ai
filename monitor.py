"""Read-only localhost dashboard for the current training run."""
import ctypes
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/five-hour-20261027'


def read_json(name):
    path = RUN / name
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def snapshot():
    state = read_json('status.json')
    alive = None
    if state.get('pid'):
        kernel = ctypes.windll.kernel32
        kernel.OpenProcess.restype = ctypes.c_void_p
        handle = kernel.OpenProcess(0x1000, False, state['pid'])
        if handle:
            code = ctypes.c_ulong()
            kernel.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))
            alive = code.value == 259
            kernel.CloseHandle(ctypes.c_void_p(handle))
        else:
            alive = False
    trials = []
    path = RUN / 'trials.jsonl'
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                trials.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # A record may be mid-write.
    log = ROOT / 'outputs/five-hour-training.log'
    return {'state': state, 'alive': alive, 'trials': trials,
            'baseline': read_json('baseline_validation.json'),
            'final': read_json('final_metrics.json'),
            'log': log.read_text(encoding='utf-8')[-7000:] if log.exists() else ''}


PAGE = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Flood Detection AI — Live Training</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#0b121b;color:#e7eff8;font:15px system-ui,sans-serif}main{max-width:1160px;margin:auto;padding:40px 28px}header{display:flex;justify-content:space-between;align-items:center;gap:20px}.eyebrow{color:#54d8c6;letter-spacing:.16em;font-size:12px;font-weight:700}h1{font-size:36px;margin:8px 0}p{color:#91a6ba;line-height:1.6}.badge{background:#133930;color:#73e5c8;padding:12px 18px;border-radius:30px;white-space:nowrap}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:currentColor;margin-right:8px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:28px 0}.card,.panel{background:#121e2c;border:1px solid #253546;border-radius:16px;padding:22px}.label{color:#91a6ba;font-size:13px}.number{font-size:32px;font-weight:650;margin:12px 0 4px}.small{color:#91a6ba;font-size:12px}.grid{display:grid;grid-template-columns:1.7fr 1fr;gap:18px}.panel h2{font-size:17px;margin:0 0 18px}.bar{height:8px;background:#253546;border-radius:8px;overflow:hidden;margin:14px 0}.bar>div{height:100%;background:#57d8c5;transition:width 1s}.row{display:flex;justify-content:space-between;gap:12px;margin:14px 0}.chart{width:100%;height:190px;margin-top:10px}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:12px 8px;border-bottom:1px solid #253546}th{color:#91a6ba;font-weight:500}pre{white-space:pre-wrap;word-break:break-word;color:#a8c3d7;font:12px/1.7 monospace;max-height:230px;overflow:auto}.wide{margin-top:18px}a{color:#65dbce}.error{color:#ffb878}.foot{font-size:12px;margin-top:24px}@media(max-width:800px){.cards{grid-template-columns:1fr 1fr}.grid{grid-template-columns:1fr}header{align-items:flex-start;flex-direction:column}h1{font-size:28px}}
</style><main>
<header><div><div class="eyebrow">SATELLITE INTELLIGENCE / LOCAL TRAINING</div><h1>Flood Detection AI</h1><p>Your five-hour training run, live from this computer.</p></div><div class="badge" id="badge">Connecting…</div></header>
<div class="cards"><div class="card"><div class="label">Current trial</div><div class="number" id="trial">—</div><div class="small" id="completed">Waiting for status</div></div><div class="card"><div class="label">Best validation IoU</div><div class="number" id="score">—</div><div class="small" id="gain">Compared with the same validation sample</div></div><div class="card"><div class="label">Time remaining</div><div class="number" id="remaining">—</div><div class="small">Five-hour wall-clock budget</div></div><div class="card"><div class="label">Training pixel pool</div><div class="number" id="pixels">—</div><div class="small" id="chips">Sentinel-1 radar images</div></div></div>
<div class="grid"><section class="panel"><h2>Model search progress</h2><div class="row"><span class="label" id="elapsed">Elapsed —</span><span class="label" id="percent">—</span></div><div class="bar"><div id="progress" style="width:0%"></div></div><svg class="chart" id="chart" viewBox="0 0 620 190" role="img" aria-label="Validation IoU by completed trial"></svg><div class="small">Each point is a completed trial. Higher water IoU is better. Scores use a fixed validation sample; final test results appear after training.</div></section><section class="panel"><h2>Current configuration</h2><div id="config"></div><div class="row"><span class="label">Scheduled finish</span><span id="finish">—</span></div><div class="small">Bishkek time · automatic checkpoint saving</div></section></div>
<section class="panel wide"><h2>Recent trials</h2><table><thead><tr><th>Trial</th><th>Water IoU</th><th>F1</th><th>Duration</th><th>Result</th></tr></thead><tbody id="trials"></tbody></table></section>
<section class="panel wide" id="finalPanel" hidden><h2>Final evaluation</h2><pre id="final"></pre></section>
<details class="panel wide"><summary>Live training log</summary><pre id="log"></pre></details>
<p class="foot" id="updated">Fetching local training status…</p><p class="foot">This page displays the actual background Python job. Refreshes every 3 seconds. Closing this tab does not stop training. <a href="https://github.com/bekturovtemirlan000-gif/flood-detection-ai" target="_blank" rel="noopener">GitHub repository ↗</a></p>
</main><script>
const $=id=>document.getElementById(id), pct=x=>Number.isFinite(x)?(x*100).toFixed(2)+'%':'—';
const duration=n=>{n=Math.max(0,Math.floor(n));return `${Math.floor(n/3600)}h ${Math.floor(n%3600/60)}m`};
function row(k,v){const r=document.createElement('div');r.className='row';const a=document.createElement('span');a.className='label';a.textContent=k;const b=document.createElement('span');b.textContent=v;r.append(a,b);return r}
async function refresh(){try{const r=await fetch('/status',{cache:'no-store'});if(!r.ok)throw Error('Status unavailable');const d=await r.json(),s=d.state,now=Date.now(),start=Date.parse(s.started_at),end=Date.parse(s.deadline_utc);const done=(s.status||'').startsWith('completed'),terminal=done||s.status==='failed';
$('badge').textContent=terminal?s.status.replaceAll('_',' '):d.alive?'● '+s.status.replaceAll('_',' '):'Process stopped — check log';$('badge').style.color=d.alive||done?'#73e5c8':'#ffb878';
$('trial').textContent=s.current_trial||'—';$('completed').textContent=(s.completed_trials||0)+' trials completed';$('score').textContent=pct(s.best_validation_iou);const base=d.baseline.metrics?.water_iou;$('gain').textContent=Number.isFinite(base)&&Number.isFinite(s.best_validation_iou)?((s.best_validation_iou-base)*100).toFixed(2)+' percentage points vs baseline':'Baseline comparison pending';
$('remaining').textContent=terminal?'Finished':duration((end-now)/1000);$('pixels').textContent=s.train_pixels?(s.train_pixels/1e6).toFixed(2)+'M':'Preparing';$('chips').textContent=s.chip_counts?`${s.chip_counts.train} training / ${s.chip_counts.valid} validation images`:'Downloading satellite data';
const elapsed=((terminal&&s.finished_at?Date.parse(s.finished_at):now)-start)/1000,progress=Math.min(100,Math.max(0,elapsed/((end-start)/1000)*100));$('elapsed').textContent='Elapsed '+duration(elapsed);$('percent').textContent=progress.toFixed(1)+'% of budget';$('progress').style.width=progress+'%';$('finish').textContent=new Date(end).toLocaleString('en-GB',{timeZone:'Asia/Bishkek',month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'});
const c=s.configuration||{};$('config').replaceChildren(...[['Boosting iterations',c.max_iter],['Maximum leaves',c.max_leaf_nodes],['Learning rate',c.learning_rate],['Training pixels / trial',c.sample_pixels?.toLocaleString()],['CPU threads',s.threads]].map(([k,v])=>row(k,v??'—')));
$('trials').replaceChildren(...d.trials.slice(-8).reverse().map(t=>{const tr=document.createElement('tr');[t.trial,pct(t.metrics?.water_iou),pct(t.metrics?.water_f1),t.seconds?Math.round(t.seconds)+'s':'—',t.success?'Completed':'Interrupted'].forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.append(td)});return tr}));
const points=d.trials.filter(t=>t.success&&t.metrics),values=points.map(t=>t.metrics.water_iou);if(values.length){const low=Math.min(...values,base??1)-.005,high=Math.max(...values,base??0)+.005;const y=v=>165-(v-low)/(high-low)*140;const line=values.map((v,i)=>`${45+i/Math.max(values.length-1,1)*555},${y(v)}`).join(' ');$('chart').innerHTML=`<text x="0" y="18" fill="#91a6ba" font-size="11">${pct(high)}</text><text x="0" y="169" fill="#91a6ba" font-size="11">${pct(low)}</text><line x1="45" x2="600" y1="165" y2="165" stroke="#304354"/><polyline points="${line}" fill="none" stroke="#57d8c5" stroke-width="2.5"/>`}
$('log').textContent=d.log;$('finalPanel').hidden=!Object.keys(d.final).length;$('final').textContent=JSON.stringify(d.final,null,2);$('updated').textContent='Live connection · checked '+new Date().toLocaleTimeString()+' · training process '+s.pid;
}catch(e){$('updated').textContent='Connection interrupted. Retrying automatically…';$('badge').textContent='Disconnected';}}refresh();setInterval(refresh,3000);
</script></html>'''


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ('/', '/status'):
            self.send_error(404)
            return
        try:
            body = (json.dumps(snapshot()) if self.path == '/status' else PAGE).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json' if self.path == '/status' else 'text/html; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (OSError, ValueError):
            self.send_error(503, 'Status temporarily unavailable')

    def log_message(self, *_):
        pass


if __name__ == '__main__':
    print('Training dashboard: http://127.0.0.1:8765', flush=True)
    ThreadingHTTPServer(('127.0.0.1', 8765), Handler).serve_forever()
