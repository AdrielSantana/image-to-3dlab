// --- Setup & Status -------------------------------------------------------------------
// What is installed, what it would cost to install the rest, and the only place a download
// can be started. Reads GET /api/catalog (viewer/backend_catalog.py).
//
// `AGENTS.md` forbids fetching weights before the user has confirmed which pipeline and
// which route, so the download button here states the size, the source and the licence
// and waits for an answer. Nobody should find 50 GB on their disk from a button that said
// "Run setup".
//
// This page is also where the lab's other tools will report themselves as they land, which
// is why it is "Setup & Status" and not "Install": it is the machine-state page, and the
// natural home later for account and preferences.

const s = (id) => document.getElementById(id);
const SKIP_KEY = 'i2l.setup.skip';

// `panelFor` is the backend whose download the progress panel shows. It outlives the run
// so the finished result stays under the card that started it.
const state = { catalog: null, source: null, running: null, panelFor: null };

/** Put the progress panel right under the card it belongs to. It used to sit below every
 * card, so on a long list the bar was off-screen and a click on Set up looked like it had
 * done nothing (issue #36). */
function placeRunPanel() {
  const card = state.panelFor
    && s('setup-backends').querySelector(`[data-backend="${CSS.escape(state.panelFor)}"]`);
  if (card) card.after(s('setup-run'));
}

/** Whether the user asked not to land here. Browser storage can throw; never block on it. */
export function skipRequested() {
  try {
    return localStorage.getItem(SKIP_KEY) === '1';
  } catch (_) {
    return false;
  }
}

function setSkip(value) {
  try {
    localStorage.setItem(SKIP_KEY, value ? '1' : '0');
  } catch (_) { /* private window: the preference simply does not persist */ }
}

const STATE_META = {
  ready: { dot: '●', cls: 'ok', text: 'installed' },
  partial: { dot: '◐', cls: 'warn', text: 'partly downloaded' },
  missing: { dot: '○', cls: 'off', text: 'not installed' },
  unsupported: { dot: '–', cls: 'off', text: 'not available on this machine' },
};

/** A setup time a person would say out loud: "about 20 min", "about an hour". */
function roughly(minutes) {
  if (minutes < 60) return `about ${minutes} min`;
  const hours = minutes / 60;
  return hours === 1 ? 'about an hour' : `about ${Number(hours.toFixed(1))} hours`;
}

function backendCard(backend) {
  const meta = STATE_META[backend.state] || STATE_META.missing;
  const card = document.createElement('div');
  card.className = `setup-card ${backend.state}`;
  card.dataset.backend = backend.id;

  const size = backend.state === 'partial'
    ? `${backend.human_present} of ${backend.human_expected}`
    : backend.human_expected;
  // Say where the weights are when they are not here yet but the backend still works.
  const later = backend.action === 'none' && backend.bytes_present === 0
    ? ' · weights download on your first run' : '';
  const minutes = backend.setup_minutes ? ` · ${roughly(backend.setup_minutes)} to set up` : '';

  card.innerHTML = `
    <div class="setup-card-head">
      <span class="setup-dot ${meta.cls}">${meta.dot}</span>
      <div class="setup-card-title">
        <strong>${backend.label}</strong>
        ${backend.recommended ? '<span class="setup-pill">start here</span>' : ''}
        <div class="setup-card-state">${meta.text} · ${size}${minutes}${later}</div>
      </div>
      <div class="setup-card-action"></div>
    </div>
    <p class="setup-card-best">${backend.best_for}</p>
    <p class="setup-card-trade">${backend.tradeoff}</p>
    ${backend.platform_note
      ? `<p class="setup-card-caveat">${backend.platform_note}</p>` : ''}
    ${backend.caveat ? `<p class="setup-card-caveat">${backend.caveat}</p>` : ''}
    <details class="setup-card-detail">
      <summary>What gets downloaded</summary>
      <ul>${backend.weights.map((w) => `
        <li>
          <a href="${w.source_url}" target="_blank" rel="noopener">${w.source}</a>
          — ${w.human_expected}${w.present ? ` (${w.human_present} on disk)` : ''}
          ${w.note ? `<br><small>${w.note}</small>` : ''}
        </li>`).join('')}
      </ul>
      ${backend.extra_steps.length
        ? `<p class="setup-card-extra">${backend.extra_steps.join('<br>')}</p>` : ''}
      <p class="setup-card-licence">
        Licence: ${backend.license.name}.
        <a href="${backend.license.url}" target="_blank" rel="noopener">Read it here</a>.
      </p>
    </details>`;

  const action = card.querySelector('.setup-card-action');
  // Driven by the server's `action`, not inferred from state here. Build and weights
  // are independent: a built TRELLIS with no weights is usable and must not be offered a
  // Set up button, because re-running a finished bootstrap fails on its own patches.
  const LABELS = { build: 'Set up', download: 'Download', resume: 'Resume download' };
  // A backend that cannot run here gets no button at all. Offering one would spend
  // gigabytes of someone's bandwidth on a build that fails partway through.
  if (backend.supported_here === false) {
    action.innerHTML = `<span class="setup-unavailable">needs ${backend.requires}</span>`;
  } else if (backend.action === 'none') {
    action.innerHTML = '<span class="setup-ready">✓ ready</span>';
  } else if (backend.action === 'manual') {
    // Some routes the viewer cannot install for you: SF3D wants a shell bootstrap, and
    // the dgrauet Hunyuan shape stage is a deliberate manual clone because it is
    // Tencent-licensed code rather than just weights. Showing the command is honest; a
    // button that throws is not.
    const span = document.createElement('span');
    span.className = 'setup-manual';
    span.textContent = 'run ';
    const code = document.createElement('code');
    code.textContent = backend.install;
    span.appendChild(code);
    action.replaceChildren(span);
  } else {
    const button = document.createElement('button');
    button.textContent = LABELS[backend.action];
    button.onclick = () => confirmDownload(backend, button);
    action.appendChild(button);
  }
  // Reclaiming is about bytes on disk, not about whether the backend works.
  if (backend.bytes_present > 0) action.appendChild(removeButton(backend));
  return card;
}

/** Reclaiming the disk, kept separate from cancelling a download.
 *
 * Cancelling leaves partial files that Hugging Face resumes from, so clearing them
 * automatically would turn a pause into a restart. This is the deliberate version, for
 * when the space is wanted back or a download came down corrupt.
 */
function removeButton(backend) {
  const button = document.createElement('button');
  button.className = 'ghost setup-remove';
  button.textContent = 'Remove';
  button.title = `Delete ${backend.human_present} of weights from disk`;
  button.onclick = async () => {
    const shared = backend.weights.some((w) => w.path.includes('huggingface'));
    // Typed confirmation, not a yes/no. Three Remove buttons sit in one column and each
    // one deletes several gigabytes that take minutes to hours to replace; a misplaced
    // click plus a reflexive OK is a real way to lose an afternoon. Typing the backend's
    // id proves both that it was meant and *which* one was meant, which a bare "delete"
    // would not.
    // eslint-disable-next-line no-alert -- deliberate, and the strongest gate available.
    const typed = window.prompt([
      `Delete ${backend.human_present} of ${backend.label} weights?`,
      '',
      ...backend.weights.filter((w) => w.present).map((w) => `  ${w.path}`),
      '',
      shared
        ? 'These live in the shared Hugging Face cache, so other tools on this machine'
          + ' may be using them.'
        : 'This cannot be undone from here.',
      `They can be downloaded again, which takes ${roughly(backend.setup_minutes)}.`,
      '',
      `Type  ${backend.id}  to confirm:`,
    ].join('\n'), '');
    if (typed === null) return;
    if (typed.trim().toLowerCase() !== backend.id.toLowerCase()) {
      window.alert(`Not deleted. You typed "${typed}", expected "${backend.id}".`);
      return;
    }
    button.disabled = true;
    try {
      const response = await fetch(`/api/setup/${encodeURIComponent(backend.id)}/remove`,
        { method: 'POST' });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
      s('setup-summary').textContent = `Removed ${payload.freed}.`;
    } catch (error) {
      button.disabled = false;
      window.alert(`Could not remove: ${error.message}`);
    }
    load();
  };
  return button;
}

/** State the cost, then ask. The confirmation is the point of the whole page. */
function confirmDownload(backend, button) {
  const remaining = backend.bytes_expected - backend.bytes_present;
  // Say what this particular button actually does. TRELLIS's setup builds the Metal port
  // and fetches nothing; its weights arrive on the first generation run, and promising a
  // download here would be a lie the progress bar then has to keep.
  const body = backend.setup_fetches_weights
    ? [
      `About ${formatBytes(remaining)} will be downloaded from Hugging Face`,
      `into ${backend.weights[0].path.replace(/\/[^/]*$/, '/')}`,
      '',
      'Some sources need you to be signed in to Hugging Face and to have accepted their',
      'terms; the log will say so if the download is refused.',
    ]
    : [
      `This builds the Metal port first, which takes ${roughly(backend.setup_minutes)}`,
      'and downloads no weights.',
      '',
      `The ${backend.human_expected} of weights are fetched on your first generation run,`,
      'not now.',
    ];
  const lines = [
    `${backend.label}`,
    '',
    ...body,
    '',
    `Licence: ${backend.license.name}`,
    backend.caveat ? `\n${backend.caveat}\n` : '',
    '',
    backend.setup_fetches_weights ? 'Start the download?' : 'Start the build?',
  ].filter(Boolean);

  // eslint-disable-next-line no-alert -- a deliberate, blocking confirmation: this is the
  // one action on the page that spends the user's disk and bandwidth.
  if (!window.confirm(lines.join('\n'))) return;
  startDownload(backend, button);
}

function formatBytes(value) {
  if (value <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let size = value;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) { size /= 1024; unit += 1; }
  return `${size.toFixed(unit >= 3 ? 1 : 0)} ${units[unit]}`;
}

async function startDownload(backend, button) {
  button.disabled = true;
  s('setup-run').hidden = false;
  s('setup-run-cancel').hidden = false;
  s('setup-run-title').textContent =
    `${backend.setup_fetches_weights ? 'Downloading' : 'Building'} ${backend.label}`;
  s('setup-run-detail').textContent = 'starting…';
  s('setup-log').textContent = '';
  s('setup-run-bar').style.width = '0%';
  state.panelFor = backend.id;
  placeRunPanel();
  s('setup-run').scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  try {
    const response = await fetch(`/api/setup/${encodeURIComponent(backend.id)}/download`, {
      method: 'POST',
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    state.running = backend.id;
    watch(payload);
  } catch (error) {
    button.disabled = false;
    s('setup-run-detail').textContent = `Could not start: ${error.message}`;
  }
}

function watch(payload) {
  state.source?.close();
  state.source = new EventSource(payload.events_url);
  state.source.onmessage = (message) => applyEvent(JSON.parse(message.data));
  s('setup-run-cancel').onclick = async () => {
    await fetch(`/api/setup/${encodeURIComponent(state.running)}/cancel`, { method: 'POST' });
  };
}

function applyEvent(event) {
  if (!event) return;
  if (typeof event.overall_pct === 'number') {
    s('setup-run-bar').style.width = `${event.overall_pct}%`;
  }
  if (event.detail) s('setup-run-detail').textContent = event.detail;
  if (event.log) {
    const log = s('setup-log');
    log.textContent += `${event.log}\n`;
    log.scrollTop = log.scrollHeight;
  }
  if (event.phase === 'done' || event.phase === 'error' || event.phase === 'cancelled') {
    state.source?.close();
    state.source = null;
    state.running = null;
    s('setup-run-cancel').hidden = true;
    load();
  }
}

function summarise(catalog) {
  const ready = catalog.backends.filter((b) => b.state === 'ready');
  const onDisk = catalog.backends.reduce((total, b) => total + b.bytes_present, 0);
  // Said before anything else, because "nothing is installed" and "nothing can be
  // installed here" look the same in a list of cards, and only one of them is fixable
  // by clicking a button. Someone on the wrong machine deserves to know on arrival.
  if (catalog.host && !catalog.host.any_backend_runs_here) {
    return `<strong>This machine is ${catalog.host.label}.</strong> `
      + `Every backend here needs ${catalog.host.supported.join(' or ')}, so there is `
      + `nothing to install yet. NVIDIA support is on the way — the backends below `
      + `are listed so you can see what is coming.`;
  }
  if (!ready.length) {
    return `<strong>No backend installed yet.</strong> Pick one below. `
      + `The recommended one is about ${catalog.backends[0].human_expected}.`;
  }
  return `<strong>${ready.length} of ${catalog.backends.length} backends installed</strong>`
    + ` · ${formatBytes(onDisk)} of model weights on disk`;
}

export async function load() {
  const host = s('setup-backends');
  try {
    const response = await fetch('/api/catalog');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.catalog = await response.json();
    s('setup-summary').innerHTML = summarise(state.catalog);
    host.after(s('setup-run')); // park it outside the cards before they are rebuilt
    host.innerHTML = '';
    for (const backend of state.catalog.backends) host.appendChild(backendCard(backend));
    placeRunPanel();
    document.dispatchEvent(new CustomEvent('viewer:catalog', { detail: state.catalog }));
  } catch (error) {
    s('setup-summary').textContent = `Could not read machine status: ${error.message}`;
  }
}

s('setup-skip').checked = skipRequested();
s('setup-skip').onchange = (event) => setSkip(event.target.checked);
document.addEventListener('viewer:modechange', (event) => {
  if (event.detail.mode === 'setup') load();
});

load();
