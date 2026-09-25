// --- Props mode ----------------------------------------------------------------------
// Split a prop-sheet GLB into separate props and bake each one's LODs. Drives
// POST /api/props (viewer/props_api.py), a sibling of the Finish job, so the
// SSE-with-polling-fallback shape here is the one modes/finish.js uses.
//
// The results come from the run directory, not from this page's memory: the server
// describes a run from disk, and the run list below re-opens any of them.
import { JobProgressPanel, formatDuration } from '../components/job-progress.js';

const f = (id) => document.getElementById(id);

const progress = new JobProgressPanel({
  stages: f('props-stages'),
  bar: f('props-overall-bar'),
  label: f('props-overall-label'),
  eta: f('props-overall-eta'),
});
const SPLIT_ONLY = { stages: ['split'], stage_labels: { split: 'Split the sheet' } };

const state = { asset: null, running: false, source: null, poll: null, tools: {}, shown: null };

const kb = (bytes) => (bytes == null ? '' : bytes >= 1048576
  ? `${(bytes / 1048576).toFixed(1)} MB` : `${Math.round(bytes / 1024)} KB`);

function updateSubmit() {
  const source = state.asset || f('props-generated').value;
  f('props-submit').disabled = !source || state.running || !state.tools.blender;
}

function setRunning(running) {
  state.running = running;
  f('props-cancel').hidden = !running;
  f('props-runs-refresh').disabled = running;
  for (const button of document.querySelectorAll('.prop-turn')) button.disabled = running;
  updateSubmit();
}

function stopStreams() {
  if (state.source) { state.source.close(); state.source = null; }
  if (state.poll) { clearInterval(state.poll); state.poll = null; }
}

function applyEvent(event) {
  if (!event) return;
  // The prop rows only exist once the split has said which props there are.
  if (event.stages) progress.configure({ stages: event.stages, stage_labels: event.stage_labels });
  progress.apply(event);
  if (event.message) f('props-status').textContent = event.message;

  // Only the server's own terminal event carries the run, so only it ends the watch.
  if (event.phase === 'done' && event.run) {
    stopStreams();
    setRunning(false);
    f('props-status').textContent =
      `${event.message} in ${formatDuration(event.elapsed_seconds)}`;
    showRun(event.run);
    loadRuns();
  } else if (event.phase === 'error' || event.phase === 'cancelled') {
    stopStreams();
    setRunning(false);
    f('props-status').textContent = event.message || 'Failed';
    if (event.log_tail) console.warn('[props]', event.log_tail);
    loadRuns();
  }
}

function startPolling(jobId) {
  if (state.poll) return;
  state.poll = setInterval(async () => {
    try {
      const response = await fetch(`/api/props/${jobId}/status`);
      if (!response.ok) return;
      applyEvent((await response.json()).last_event);
    } catch (_) { /* the next poll or an SSE reconnect recovers */ }
  }, 2000);
}

function watch(payload) {
  state.source = new EventSource(payload.events_url);
  state.source.onmessage = (message) => applyEvent(JSON.parse(message.data));
  state.source.onerror = () => startPolling(payload.job_id);
  f('props-cancel').onclick = async () => {
    await fetch(`/api/props/${payload.job_id}/cancel`, { method: 'POST' });
  };
}

function begin(message) {
  setRunning(true);
  progress.configure(SPLIT_ONLY);
  progress.reset();
  f('props-progress-box').hidden = false;
  f('props-status').textContent = message;
}

// --- Results ---------------------------------------------------------------------------

function preview(prop, row) {
  if (!prop.preview_url) return;
  f('props-viewer').hidden = false;
  // The Compare page, one model and no add-pane, as the Generate tab embeds it.
  const src = `${prop.preview_url}?t=${Date.now()}`;
  f('props-frame').src =
    `/viewer/index.html?a=${encodeURIComponent(src)}&la=${encodeURIComponent(prop.name)}&restricted=1`;
  for (const other of document.querySelectorAll('.prop-row')) other.classList.remove('showing');
  row.classList.add('showing');
  state.shown = prop.name;
}

function propRow(run, prop) {
  const row = document.createElement('div');
  row.className = 'prop-row';
  const size = prop.size ? prop.size.map((s) => s.toFixed(2)).join(' × ') : '';
  const turn = prop.extra_turn_degrees ? ` · turned ${prop.extra_turn_degrees}°` : '';
  row.innerHTML =
    `<div><strong></strong><div class="prop-meta">${(prop.faces || 0).toLocaleString()} faces` +
    `${size ? ` · ${size}` : ''}${turn}</div></div>` +
    '<div class="prop-actions"></div><div class="prop-lods"></div>';
  row.querySelector('strong').textContent = prop.name;

  const actions = row.querySelector('.prop-actions');
  if (prop.preview_url) {
    const view = document.createElement('button');
    view.className = 'ghost';
    view.textContent = 'View';
    view.onclick = () => preview(prop, row);
    actions.appendChild(view);
  }
  const turnButton = document.createElement('button');
  turnButton.className = 'ghost prop-turn';
  turnButton.textContent = 'Turn 90°';
  turnButton.title = 'Turn this prop a quarter turn about the vertical and re-bake only it';
  turnButton.disabled = state.running;
  turnButton.onclick = () => turnProp(run.directory, prop.name);
  actions.appendChild(turnButton);

  const lods = row.querySelector('.prop-lods');
  if (!prop.lods.length) lods.textContent = 'no LODs yet';
  for (const lod of prop.lods) {
    const span = document.createElement('span');
    const plain = document.createElement('a');
    plain.href = lod.url;
    plain.download = '';
    plain.textContent = `LOD${lod.index} ${kb(lod.bytes)}`;
    span.appendChild(plain);
    if (lod.web_url) {
      const web = document.createElement('a');
      web.href = lod.web_url;
      web.download = '';
      web.textContent = `web ${kb(lod.web_bytes)}`;
      span.append(' · ', web);
    }
    lods.appendChild(span);
  }

  // A turn this close to 45° is a coin toss between the front and the side: the test
  // sheet's chest came back with its lock facing sideways.
  if (prop.yaw_tie && !prop.extra_turn_degrees) {
    const tie = document.createElement('div');
    tie.className = 'prop-tie';
    tie.textContent =
      `Turned ${prop.yaw_degrees}° to square it up, close to 45°: it may face sideways. ` +
      'View it, and Turn 90° if so.';
    row.appendChild(tie);
  }
  return row;
}

function showRun(run) {
  f('props-result').hidden = false;
  f('props-result-title').textContent =
    `${run.props.length} props · output/props/${run.directory}/`;
  f('props-blend').hidden = !run.blend_url;
  if (run.blend_url) f('props-blend').href = run.blend_url;
  const list = f('props-list');
  list.innerHTML = '';
  for (const prop of run.props) {
    const row = propRow(run, prop);
    list.appendChild(row);
    if (prop.name === state.shown) preview(prop, row);
  }
  if (!state.shown && run.props.length) {
    preview(run.props[0], list.firstElementChild);
  }
}

async function turnProp(directory, name) {
  if (state.running) return;
  begin(`Turning ${name} and re-baking it…`);
  state.shown = name;
  try {
    const response = await fetch(`/api/props/runs/${encodeURIComponent(directory)}/turn`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prop: name, degrees: 90 }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    watch(payload);
  } catch (error) {
    setRunning(false);
    f('props-status').textContent = `Could not turn ${name}: ${error.message}`;
  }
}

// --- Runs on disk, generated models and tools --------------------------------------------

function showTools(tools) {
  const host = f('props-tools');
  host.innerHTML = '';
  const line = (cls, html) => {
    const div = document.createElement('div');
    div.className = cls;
    div.innerHTML = html;
    host.appendChild(div);
  };
  if (!tools.blender) {
    line('bad', 'Blender not found. Install it, or point <code>I2L_BLENDER</code> at it.');
  }
  if (!tools.gltfpack) {
    line('warn', 'gltfpack not found, so LODs stay uncompressed. Put a native release ' +
      '(github.com/zeux/meshoptimizer/releases) on PATH or at ' +
      '<code>vendor/gltfpack/gltfpack</code>; the npm build cannot write WebP.');
  }
  // Refreshed on every visit, so only force it off; a choice to skip it stays made.
  f('props-compress').disabled = !tools.gltfpack;
  if (!tools.gltfpack) f('props-compress').checked = false;
}

function showGenerated(models) {
  const select = f('props-generated');
  const chosen = select.value;
  select.length = 1;
  for (const model of models) {
    const option = document.createElement('option');
    option.value = model.path;
    option.textContent = `${model.name} (${kb(model.bytes)})`;
    select.appendChild(option);
  }
  select.value = [...select.options].some((o) => o.value === chosen) ? chosen : '';
}

function runRow(run) {
  const row = document.createElement('div');
  row.className = 'stage-row';
  if (run.finished) row.classList.add('done');
  row.innerHTML =
    `<span class="stage-dot">${run.finished ? '✓' : '·'}</span>` +
    '<span><code style="font-size:11px"></code>' +
    `<br><small style="opacity:.7">${run.props.length} props</small></span>` +
    '<span class="stage-detail"></span>';
  row.querySelector('code').textContent = run.directory;
  if (run.props.length) {
    const open = document.createElement('button');
    open.className = 'ghost';
    open.textContent = 'Open';
    open.onclick = () => { state.shown = null; showRun(run); };
    row.querySelector('.stage-detail').appendChild(open);
  }
  return row;
}

async function loadRuns() {
  const host = f('props-runs');
  try {
    const response = await fetch('/api/props/runs');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const { runs, generated, tools } = await response.json();
    state.tools = tools;
    showTools(tools);
    showGenerated(generated);
    host.innerHTML = '';
    if (!runs.length) host.innerHTML = '<small style="opacity:.7">No prop-sheet runs yet.</small>';
    for (const run of runs) host.appendChild(runRow(run));
  } catch (error) {
    host.innerHTML = `<small style="opacity:.7">Could not list runs: ${error.message}</small>`;
  }
  updateSubmit();
}

// --- Inputs ------------------------------------------------------------------------------

f('props-asset').onchange = (event) => {
  state.asset = event.target.files[0] || null;
  f('props-asset-name').textContent = state.asset ? state.asset.name : 'no file chosen';
  updateSubmit();
};
f('props-generated').onchange = updateSubmit;
f('props-runs-refresh').onclick = loadRuns;

f('props-submit').onclick = async () => {
  if (state.running) return;
  begin('Uploading…');
  f('props-result').hidden = true;
  f('props-viewer').hidden = true;
  state.shown = null;

  const settings = {
    names: f('props-names').value,
    lods: f('props-lods').value,
    atlas: Number(f('props-atlas').value),
    metallic: Number(f('props-metallic').value),
    roughness: Number(f('props-roughness').value),
    ior: Number(f('props-ior').value),
    compress: f('props-compress').checked,
  };
  const form = new FormData();
  if (state.asset) form.append('asset', state.asset, state.asset.name);
  else form.append('generated', f('props-generated').value);
  form.append('settings', JSON.stringify(settings));

  try {
    const response = await fetch('/api/props', { method: 'POST', body: form });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    watch(payload);
  } catch (error) {
    setRunning(false);
    f('props-status').textContent = `Could not start: ${error.message}`;
  }
};

// The generated-model list goes stale while the page is open; refresh it on arrival.
document.addEventListener('viewer:modechange', (event) => {
  if (event.detail?.mode === 'props' && !state.running) loadRuns();
});

progress.configure(SPLIT_ONLY);
updateSubmit();
loadRuns();
