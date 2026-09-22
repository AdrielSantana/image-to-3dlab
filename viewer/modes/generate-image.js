// Text to image: the step before everything else, for people with no source picture yet.
//
// Deliberately plain. One prompt box, three settings that matter, and the result beside
// it. Anything that needs explaining has the explanation next to it rather than in a doc,
// because the person who needs it is looking at the field, not the repository.

const $ = (id) => document.getElementById(id);

const state = { jobId: null, source: null, startedAt: 0, tick: null };

function setStatus(text, kind = '') {
  const node = $('image-status');
  node.textContent = text;
  node.className = `image-status ${kind}`;
}

function setBar(percent) {
  const bar = $('image-bar');
  if (percent === null) { bar.hidden = true; return; }
  bar.hidden = false;
  $('image-bar-fill').style.width = `${Math.max(0, Math.min(100, percent))}%`;
}

function setRunning(running) {
  $('image-run').disabled = running;
  $('image-cancel').hidden = !running;
  $('image-prompt').disabled = running;
}

function readSettings() {
  const size = Number($('image-size').value);
  return {
    width: size,
    height: size,
    steps: Number($('image-steps').value),
    seed: Number($('image-seed').value),
    cfg_scale: Number($('image-cfg').value),
    sampler: $('image-sampler').value,
    negative_prompt: $('image-negative').value.trim(),
  };
}

function humanSeconds(seconds) {
  if (!Number.isFinite(seconds)) return '';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m${String(Math.round(seconds % 60)).padStart(2, '0')}s`;
}

// The elapsed counter runs off a local clock rather than the event stream, so the page
// still looks alive during the decode, which emits nothing for minutes at a time.
function startTicker() {
  stopTicker();
  state.startedAt = Date.now();
  state.tick = setInterval(() => {
    if (!state.jobId) return;
    const elapsed = (Date.now() - state.startedAt) / 1000;
    const node = $('image-elapsed');
    node.textContent = `running ${humanSeconds(elapsed)}`;
  }, 1000);
}

function stopTicker() {
  if (state.tick) { clearInterval(state.tick); state.tick = null; }
}

function closeStream() {
  if (state.source) { state.source.close(); state.source = null; }
}

function applyEvent(event) {
  if (event.phase === 'sampling') {
    const eta = event.eta_seconds ? `, about ${humanSeconds(event.eta_seconds)} left` : '';
    setStatus(`Step ${event.step} of ${event.total_steps} (${event.seconds_per_step.toFixed(1)}s each${eta})`);
    // Sampling is most of the run but not all of it; the decode is the rest, so the bar
    // stops at 90 rather than sitting at 100 through several silent minutes.
    setBar(event.percent * 0.9);
    return;
  }
  if (event.phase === 'decoding') {
    setStatus('Decoding the image. This part is quiet and takes a couple of minutes.');
    setBar(95);
    return;
  }
  if (event.status === 'done') {
    finish(event);
    return;
  }
  if (event.status === 'error') {
    closeStream(); stopTicker(); setRunning(false); setBar(null);
    state.jobId = null;
    setStatus(event.error || 'Generation failed.', 'bad');
    if (event.needs_setup) $('image-health-goto').hidden = false;
    return;
  }
  if (event.status === 'cancelled') {
    closeStream(); stopTicker(); setRunning(false); setBar(null);
    state.jobId = null;
    setStatus('Cancelled.', '');
  }
}

function finish(event) {
  const jobId = state.jobId;
  closeStream(); stopTicker(); setRunning(false); setBar(null);
  state.jobId = null;
  const url = `/api/image/${jobId}/result.png?t=${Date.now()}`;
  const image = $('image-output');
  image.src = url;
  image.hidden = false;
  $('image-empty').hidden = true;
  $('image-done').hidden = false;
  $('image-download').href = url;
  $('image-download').download = 'generated.png';
  const seconds = event.elapsed_seconds;
  $('image-elapsed').textContent = seconds ? `done in ${humanSeconds(seconds)}` : 'done';
  setStatus('Done. Save it, then switch to Generate 3D to turn it into a mesh.', 'good');
}

async function run() {
  const prompt = $('image-prompt').value.trim();
  if (!prompt) { setStatus('Write a prompt first.', 'bad'); return; }
  setRunning(true);
  setStatus('Loading the model. The first run of a session takes longer.');
  setBar(2);
  try {
    const response = await fetch('/api/image', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, settings: readSettings() }),
    });
    const payload = await response.json();
    if (!response.ok) {
      setRunning(false); setBar(null);
      setStatus(payload.error || 'Could not start.', 'bad');
      if (payload.needs_setup) $('image-health-goto').hidden = false;
      return;
    }
    state.jobId = payload.job_id;
    startTicker();
    const source = new EventSource(payload.events_url);
    state.source = source;
    source.onmessage = (message) => {
      try { applyEvent(JSON.parse(message.data)); } catch { /* keep-alive */ }
    };
    source.onerror = () => { /* the status poll below is the safety net */ };
  } catch (error) {
    setRunning(false); setBar(null);
    setStatus(`Could not reach the server: ${error}`, 'bad');
  }
}

async function cancel() {
  if (!state.jobId) return;
  setStatus('Cancelling…');
  try { await fetch(`/api/image/${state.jobId}/cancel`, { method: 'POST' }); }
  catch { /* the stream will report the outcome */ }
}

async function refreshHealth() {
  try {
    const response = await fetch('/api/image/defaults');
    const payload = await response.json();
    const dot = $('image-health-dot');
    if (payload.installed) {
      dot.className = 'health-dot ok';
      $('image-health-text').textContent = 'Qwen-Image 2.1 is ready.';
      $('image-health-goto').hidden = true;
    } else {
      dot.className = 'health-dot bad';
      $('image-health-text').textContent =
        'Qwen-Image 2.1 is not installed yet (about 13 GB).';
      $('image-health-goto').hidden = false;
    }
  } catch {
    $('image-health-text').textContent = 'Could not reach the server.';
  }
}

$('image-run').onclick = run;
$('image-cancel').onclick = cancel;
$('image-health-goto').onclick = () => document.getElementById('mode-setup').click();
// Ctrl/Cmd+Enter from the prompt box, because that is what everyone tries.
$('image-prompt').addEventListener('keydown', (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') run();
});

document.addEventListener('viewer:modechange', (event) => {
  if (event.detail.mode === 'generate-image') refreshHealth();
});

export { humanSeconds, readSettings };
