// --- Finish mode ---------------------------------------------------------------------
// Retopologise, repaint and compress an asset that already exists. Drives POST /api/finish
// (viewer/finish_api.py), which is a sibling of the rig rebind job, so the SSE-with-polling
// -fallback shape here matches modes/rig-review.js rather than inventing a second one.
//
// Repaint is optional on purpose. A Pixal3D asset arrives with usable colour already, so
// finishing it is retopologise + compress and takes seconds; a bleached TRELLIS.2 asset
// needs the paint stage and takes ~6 minutes.
import { JobProgressPanel } from '../components/job-progress.js';

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

function configureStages() {
  const stages = ['retopologise'];
  if (!f('finish-skip-paint').checked) stages.push('repaint');
  stages.push('compress');
  progress.configure({
    stages,
    stage_labels: STAGE_META.stage_labels,
  });
}

function setRunning(running) {
  state.running = running;
  f('finish-cancel').hidden = !running;
  updateSubmit();
}

function stopStreams() {
  if (state.source) { state.source.close(); state.source = null; }
  if (state.poll) { clearInterval(state.poll); state.poll = null; }
}

function applyEvent(event) {
  if (!event) return;
  progress.apply(event);
  if (typeof event.overall_pct === 'number') {
    f('finish-overall-bar').style.width = `${event.overall_pct}%`;
  }
  if (event.message) f('finish-status').textContent = event.message;

  if (event.phase === 'done') {
    stopStreams();
    setRunning(false);
    f('finish-result').hidden = false;
    f('finish-download').href = event.result_url;
    f('finish-record').href = event.record_url;
    const megabytes = (event.size_bytes / 1048576).toFixed(1);
    f('finish-status').textContent = `Finished — ${megabytes} MB`;
  } else if (event.phase === 'error') {
    stopStreams();
    setRunning(false);
    f('finish-status').textContent = event.message || 'Finishing failed';
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
    state.jobId = payload.job_id;
    state.source = new EventSource(payload.events_url);
    state.source.onmessage = (message) => applyEvent(JSON.parse(message.data));
    state.source.onerror = () => startPolling(payload.job_id);
    f('finish-cancel').onclick = async () => {
      await fetch(`/api/finish/${payload.job_id}/cancel`, { method: 'POST' });
    };
  } catch (error) {
    setRunning(false);
    f('finish-status').textContent = `Could not start: ${error.message}`;
  }
};

configureStages();
updateSubmit();
