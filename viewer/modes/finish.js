// --- Finish mode ---------------------------------------------------------------------
// Retopologise, repaint and compress an asset that already exists. Drives POST /api/finish
// (viewer/finish_api.py), which is a sibling of the rig rebind job, so the SSE-with-polling
// -fallback shape here matches modes/rig-review.js rather than inventing a second one.
//
// Repaint is optional on purpose. A Pixal3D asset arrives with usable colour already, so
// finishing it is retopologise + compress and takes seconds; a bleached TRELLIS.2 asset
// needs the paint stage and takes ~6 minutes.
//
// Every run lives in its own directory under output/finish/ and every stage leaves its
// artifact there, so the run list below is the recovery path: the job registry is in the
// server's memory and dies with it, but the directory does not.
import { JobProgressPanel, formatDuration } from '../components/job-progress.js';

const f = (id) => document.getElementById(id);
const STAGE_META = {
  stages: ['retopologise', 'repaint', 'compress'],
  stage_labels: {
    retopologise: 'Retopologise',
    repaint: 'Repaint',
    compress: 'Compress textures',
  },
};

const progress = new JobProgressPanel({
  stages: f('finish-stages'),
  bar: f('finish-overall-bar'),
  label: f('finish-overall-label'),
  eta: f('finish-overall-eta'),
});

const state = { asset: null, image: null, jobId: null, running: false, source: null, poll: null };

function updateSubmit() {
  f('finish-submit').disabled = !state.asset || !state.image || state.running;
}

function configureStages(stages) {
  progress.configure({
    stages: stages || STAGE_META.stages.filter(
      (stage) => stage !== 'repaint' || !f('finish-skip-paint').checked,
    ),
    stage_labels: STAGE_META.stage_labels,
  });
}

function setRunning(running) {
  state.running = running;
  f('finish-cancel').hidden = !running;
  f('finish-runs-refresh').disabled = running;
  updateSubmit();
}

function stopStreams() {
  if (state.source) { state.source.close(); state.source = null; }
  if (state.poll) { clearInterval(state.poll); state.poll = null; }
}

function applyEvent(event) {
  if (!event) return;
  progress.apply(event);
  // The stage list is the server's to decide once the job exists: --skip-paint can be set
  // in the request, but only the worker's own stage_plan knows what that resolves to.
  if (event.stages) configureStages(event.stages);
  if (event.message) f('finish-status').textContent = event.message;

  // Only a terminal event that actually carries its artifacts ends the run. The worker
  // also announces its own completion, and treating that as the job's (which this did
  // until 2026-09-21) closed the stream before the URLs arrived and left a download link
  // pointing at `undefined` while the finished GLB sat on disk.
  if (event.phase === 'done' && event.result_url) {
    stopStreams();
    setRunning(false);
    f('finish-result').hidden = false;
    f('finish-download').href = event.result_url;
    f('finish-record').href = event.record_url;
    f('finish-where').textContent = `output/finish/${event.directory}/`;
    f('finish-status').textContent =
      `Finished in ${formatDuration(event.elapsed_seconds)} — ` +
      `${(event.size_bytes / 1048576).toFixed(1)} MB`;
    loadRuns();
  } else if (event.phase === 'error' || event.phase === 'cancelled') {
    stopStreams();
    setRunning(false);
    f('finish-status').textContent = event.message || 'Finishing failed';
    loadRuns();
  }
}

function startPolling(jobId) {
  if (state.poll) return;
  // The rig mode learned this the hard way: a dropped EventSource that never reconnects
  // leaves a finished job looking like a hung one.
  state.poll = setInterval(async () => {
    try {
      const response = await fetch(`/api/finish/${jobId}/status`);
      if (!response.ok) return;
      const payload = await response.json();
      applyEvent(payload.last_event);
    } catch (_) { /* the next poll or an SSE reconnect recovers */ }
  }, 2000);
}

function watch(payload) {
  state.jobId = payload.job_id;
  state.source = new EventSource(payload.events_url);
  state.source.onmessage = (message) => applyEvent(JSON.parse(message.data));
  state.source.onerror = () => startPolling(payload.job_id);
  f('finish-cancel').onclick = async () => {
    await fetch(`/api/finish/${payload.job_id}/cancel`, { method: 'POST' });
  };
}

// --- Runs on disk ---------------------------------------------------------------------

function runRow(run) {
  const row = document.createElement('div');
  row.className = 'stage-row';
  const done = run.stages_complete.join(' → ') || 'nothing yet';
  const size = run.size_bytes == null ? '' : ` · ${(run.size_bytes / 1048576).toFixed(1)} MB`;
  row.innerHTML =
    `<span class="stage-dot">${run.finished ? '✓' : '·'}</span>` +
    `<span><code style="font-size:11px">${run.directory}</code>` +
    `<br><small style="opacity:.7">${done}${size}</small></span>` +
    '<span class="stage-detail"></span>';
  if (run.finished) row.classList.add('done');

  const actions = row.querySelector('.stage-detail');
  if (run.result_url) {
    const link = document.createElement('a');
    link.href = run.result_url;
    link.download = '';
    link.textContent = 'Download';
    actions.appendChild(link);
  }
  if (run.resumable) {
    const button = document.createElement('button');
    button.className = 'ghost';
    button.textContent = 'Resume';
    button.onclick = () => resume(run.directory, button);
    actions.appendChild(button);
  }
  return row;
}

async function loadRuns() {
  const host = f('finish-runs');
  try {
    const response = await fetch('/api/finish/runs');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const { runs } = await response.json();
    host.innerHTML = '';
    if (!runs.length) {
      host.innerHTML = '<small style="opacity:.7">No finishing runs yet.</small>';
      return;
    }
    for (const run of runs) host.appendChild(runRow(run));
  } catch (error) {
    host.innerHTML = `<small style="opacity:.7">Could not list runs: ${error.message}</small>`;
  }
}

async function resume(directory, button) {
  button.disabled = true;
  setRunning(true);
  progress.reset();
  configureStages();
  f('finish-result').hidden = true;
  f('finish-progress-box').hidden = false;
  f('finish-status').textContent = `Resuming ${directory}…`;
  try {
    const response = await fetch(`/api/finish/runs/${encodeURIComponent(directory)}/resume`, {
      method: 'POST',
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    watch(payload);
  } catch (error) {
    setRunning(false);
    button.disabled = false;
    f('finish-status').textContent = `Could not resume: ${error.message}`;
  }
}

// --- Inputs ----------------------------------------------------------------------------

f('finish-asset').onchange = (event) => {
  state.asset = event.target.files[0] || null;
  f('finish-asset-name').textContent = state.asset ? state.asset.name : 'no asset chosen';
  updateSubmit();
};

f('finish-image').onchange = (event) => {
  state.image = event.target.files[0] || null;
  f('finish-image-name').textContent = state.image ? state.image.name : 'no image chosen';
  updateSubmit();
};

f('finish-skip-paint').onchange = () => {
  // Without a repaint the source image is still needed: the worker records it, and the
  // stage list changes, so the panel has to be rebuilt.
  configureStages();
  f('finish-paint-fields').hidden = f('finish-skip-paint').checked;
};

f('finish-runs-refresh').onclick = loadRuns;

f('finish-submit').onclick = async () => {
  if (!state.asset || !state.image || state.running) return;
  setRunning(true);
  progress.reset();
  configureStages();
  f('finish-result').hidden = true;
  f('finish-progress-box').hidden = false;
  f('finish-status').textContent = 'Uploading…';

  const settings = {
    faces: Number(f('finish-faces').value),
    metallic: Number(f('finish-metallic').value),
    roughness: Number(f('finish-roughness').value),
    ior: Number(f('finish-ior').value),
    texture_size: Number(f('finish-texture').value),
    skip_paint: f('finish-skip-paint').checked,
    paint_res: Number(f('finish-paint-res').value),
    paint_steps: Number(f('finish-paint-steps').value),
  };
  const form = new FormData();
  form.append('asset', state.asset, state.asset.name);
  form.append('image', state.image, state.image.name);
  form.append('settings', JSON.stringify(settings));

  try {
    const response = await fetch('/api/finish', { method: 'POST', body: form });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    watch(payload);
  } catch (error) {
    setRunning(false);
    f('finish-status').textContent = `Could not start: ${error.message}`;
  }
};

configureStages();
updateSubmit();
loadRuns();
