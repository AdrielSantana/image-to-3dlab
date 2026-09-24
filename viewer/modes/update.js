// --- Update banner -------------------------------------------------------------------------
// Says when a newer release is out and gives the one line that updates this install. The
// server asks GitHub at most once a day (viewer/update_api.py); the About page has the
// switch that stops this page asking at all.

import { bannerFor } from '../components/update-banner.js';

const OFF_KEY = 'i2l.updates.off';
const DISMISSED_KEY = 'i2l.updates.dismissed';
const s = (id) => document.getElementById(id);

/** Browser storage can throw; a missing preference means "on" and "not dismissed". */
function read(key) {
  try {
    return localStorage.getItem(key);
  } catch (_) {
    return null;
  }
}

function write(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch (_) { /* private window: the preference does not persist */ }
}

function hide() {
  s('update-banner').hidden = true;
}

export async function checkForUpdates() {
  if (read(OFF_KEY) === '1') return hide();
  try {
    const response = await fetch('/api/update-check');
    if (!response.ok) return hide();
    const banner = bannerFor(await response.json(), read(DISMISSED_KEY));
    if (!banner) return hide();
    s('update-text').textContent = banner.text;
    s('update-command').textContent = banner.command;
    s('update-notes').href = banner.url || '#';
    s('update-notes').hidden = !banner.url;
    s('update-banner').dataset.version = banner.version;
    s('update-banner').hidden = false;
  } catch (error) {
    // Offline, or a viewer older than this page: no banner, nothing broken.
    console.warn(error);
    hide();
  }
}

s('update-dismiss').addEventListener('click', () => {
  write(DISMISSED_KEY, s('update-banner').dataset.version || '');
  hide();
});

s('update-copy').addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(s('update-command').textContent);
    s('update-copy').textContent = 'Copied';
  } catch (_) {
    // No clipboard permission: select the text so a manual copy is one keystroke.
    window.getSelection().selectAllChildren(s('update-command'));
  }
});

const toggle = s('update-check-toggle');
toggle.checked = read(OFF_KEY) !== '1';
toggle.addEventListener('change', () => {
  write(OFF_KEY, toggle.checked ? '0' : '1');
  if (toggle.checked) checkForUpdates();
  else hide();
});
