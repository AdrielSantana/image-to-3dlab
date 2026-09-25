import './modes/compare.js';
import { skipRequested } from './modes/setup.js';
import { welcomeOnArrival } from './modes/welcome.js';
import { checkForUpdates } from './modes/update.js';

import './modes/generate-image.js';
import './modes/generate.js';
import './modes/finish.js';
import './modes/props.js';
import './modes/rig-review.js';
import './modes/animate.js';
import { subscribeRigEditState } from './core/rig-edit-state.js';

const byId = (id) => document.getElementById(id);
const modes = {
  setup: byId('setup-view'),
  compare: byId('compare-view'),
  'generate-image': byId('generate-image-view'),
  generate: byId('generate-view'),
  finish: byId('finish-view'),
  props: byId('props-view'),
  rig: byId('rig-view'),
  animate: byId('animate-view'),
  about: byId('about-view'),
};

function setMode(activeMode) {
  modes.setup.hidden = activeMode !== 'setup';
  modes.compare.classList.toggle('hidden', activeMode !== 'compare');
  modes['generate-image'].hidden = activeMode !== 'generate-image';
  modes.generate.hidden = activeMode !== 'generate';
  modes.finish.hidden = activeMode !== 'finish';
  modes.props.hidden = activeMode !== 'props';
  modes.rig.hidden = activeMode !== 'rig';
  modes.animate.hidden = activeMode !== 'animate';
  modes.about.hidden = activeMode !== 'about';
  for (const mode of Object.keys(modes)) {
    byId(`mode-${mode}`).classList.toggle('on', mode === activeMode);
  }
  document.dispatchEvent(new CustomEvent('viewer:modechange', { detail: { mode: activeMode } }));
}

for (const mode of Object.keys(modes)) {
  byId(`mode-${mode}`).onclick = () => setMode(mode);
}

// The About page's "Get started" button, and its first-visit landing, ask for a screen
// by name.
document.addEventListener('viewer:navigate', (event) => {
  if (modes[event.detail?.mode]) setMode(event.detail.mode);
});

subscribeRigEditState(({ pendingCount }) => {
  const button = byId('mode-rig');
  button.classList.toggle('has-pending', pendingCount > 0);
  button.title = pendingCount
    ? `${pendingCount} rig edit${pendingCount === 1 ? '' : 's'} awaiting Blender rebind`
    : 'Correct the rest skeleton';
});

// First run lands on Setup & Status, because a fresh clone can generate nothing until
// weights exist and the page is where that is explained. Once the user ticks "skip this
// next time" it is never the landing page again -- it stays one click away in the menu.
//
// A link that names a model (?a=, as `serve.py --open` and the Generate tab's embedded
// preview build it) asks for Compare, so it lands there, and the About page's
// first-visit announcement waits for a visit that was not a link. Without this the
// embedded preview landed on Setup or Generate, not on the model it was showing.
const linked = new URLSearchParams(location.search).has('a');
if (linked) setMode('compare');
else setMode(skipRequested() ? 'generate' : 'setup');
if (!linked) welcomeOnArrival();
checkForUpdates();
