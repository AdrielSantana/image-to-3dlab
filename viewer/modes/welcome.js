// --- Welcome ------------------------------------------------------------------------------
// The card shown on a first visit, once after each update, and whenever the brand in the
// top bar is clicked. It is also how updates are announced: this browser remembers the
// last version it showed, and the server sends the changelog headlines since then.
// Nothing leaves the machine; the news is the CHANGELOG.md that shipped with the code.

import {
  greeting, inlineCode, newsItems, primaryAction, shouldShow,
} from '../components/welcome-content.js';

const SEEN_KEY = 'i2l.welcome.seen';
const CHANGELOG_URL = 'https://github.com/Bingeljell/image-to-3dlab/blob/main/CHANGELOG.md';
const s = (id) => document.getElementById(id);

/** Browser storage can throw or be empty; the card must work either way. */
function lastSeen() {
  try {
    return localStorage.getItem(SEEN_KEY);
  } catch (_) {
    return null;
  }
}

function markSeen(version) {
  try {
    localStorage.setItem(SEEN_KEY, version);
  } catch (_) { /* private window: the card may show again next time, which is fine */ }
}

async function fetchWelcome(since) {
  const query = since ? `?since=${encodeURIComponent(since)}` : '';
  const response = await fetch(`/api/welcome${query}`);
  if (!response.ok) throw new Error(`welcome: HTTP ${response.status}`);
  return response.json();
}

function renderMachine(payload) {
  const routes = payload.routes || [];
  if (!routes.length) {
    return `<h2>This machine</h2><p>${inlineCode(payload.host.label)}. Nothing in the lab
      runs here yet: it needs an Apple Silicon Mac, or Linux with an NVIDIA card.</p>`;
  }
  const rows = routes.map((route) => {
    const ready = route.state === 'ready';
    return `<li><span class="${ready ? 'ok' : 'off'}">${ready ? '●' : '○'}</span>
      ${inlineCode(route.label)} <span class="off">${ready ? 'installed' : 'not installed'}</span></li>`;
  }).join('');
  return `<h2>This machine: ${inlineCode(payload.host.label)}</h2><ul>${rows}</ul>`;
}

function renderNews(payload) {
  const releases = payload.news || [];
  if (!releases.length) return '';
  const blocks = releases.map((release) => {
    const { items, more } = newsItems(release);
    const list = items.map((item) => `<li>${inlineCode(item)}</li>`).join('');
    const rest = more
      ? `<p class="more">…and ${more} more in the <a href="${CHANGELOG_URL}" target="_blank"
          rel="noopener">changelog</a>.</p>`
      : '';
    const date = release.date ? ` <span class="date">· ${inlineCode(release.date)}</span>` : '';
    return `<div class="release"><h2>New in ${inlineCode(release.version)}${date}</h2>
      <ul>${list}</ul>${rest}</div>`;
  }).join('');
  return blocks;
}

function render(payload, seen, reopened) {
  const { kicker, title } = greeting(seen, payload, reopened);
  s('welcome-kicker').textContent = kicker;
  s('welcome-title').textContent = title;
  s('welcome-tagline').textContent = payload.brand.tagline || '';
  s('welcome-machine').innerHTML = renderMachine(payload);
  const news = renderNews(payload);
  s('welcome-news').innerHTML = news;
  s('welcome-news').hidden = !news;
  const action = primaryAction(payload);
  const go = s('welcome-go');
  go.hidden = !action;
  if (action) {
    go.textContent = action.label;
    go.dataset.mode = action.mode;
  }
  document.title = payload.brand.name;
  s('brand').textContent = payload.brand.short || payload.brand.name;
}

function open(payload, seen, reopened) {
  render(payload, seen, reopened);
  const dialog = s('welcome');
  if (!dialog.open) dialog.showModal();
  markSeen(payload.version);
}

// Straight from the click rather than the dialog's `close` event, which a background tab
// can hold back; the form's method="dialog" still closes the card.
s('welcome-go').addEventListener('click', () => {
  const mode = s('welcome-go').dataset.mode;
  if (mode) document.dispatchEvent(new CustomEvent('viewer:navigate', { detail: { mode } }));
});

s('brand').addEventListener('click', async () => {
  try {
    // Reopened on purpose: show the current release's news, whatever was seen before.
    open(await fetchWelcome(null), lastSeen(), true);
  } catch (error) {
    console.warn(error);
  }
});

/** On arrival: show the card on a first visit or after an update, else stay quiet. */
export async function welcomeOnArrival() {
  const seen = lastSeen();
  try {
    const payload = await fetchWelcome(seen);
    if (shouldShow(seen, payload)) open(payload, seen, false);
    else {
      document.title = payload.brand.name;
      s('brand').textContent = payload.brand.short || payload.brand.name;
    }
  } catch (error) {
    // A static-only viewer has no API; the page works without the card.
    console.warn(error);
  }
}
