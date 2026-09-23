// --- About: welcome and what's new -------------------------------------------------------
// The top of the About page. It says hello, names this machine and what runs on it, and
// lists what changed. It is also how updates are announced: this browser remembers the
// last version it showed, and on a first visit or after an update the viewer lands here
// once. Nothing leaves the machine; the news is the CHANGELOG.md that shipped with the code.

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

function navigate(mode) {
  document.dispatchEvent(new CustomEvent('viewer:navigate', { detail: { mode } }));
}

s('welcome-go').addEventListener('click', () => {
  const mode = s('welcome-go').dataset.mode;
  if (mode) navigate(mode);
});

s('brand').addEventListener('click', () => navigate('about'));

// Refresh on each visit: what is installed changes while the viewer is open. The page
// keeps its "Welcome back" wording only for the arrival that announced an update.
let announcing = false;
document.addEventListener('viewer:modechange', async (event) => {
  if (event.detail?.mode !== 'about') return;
  if (announcing) {
    announcing = false;
    return;
  }
  try {
    render(await fetchWelcome(null), lastSeen(), true);
  } catch (error) {
    console.warn(error);
  }
});

/** On arrival: land on About once for a first visit or an update, else stay quiet. */
export async function welcomeOnArrival() {
  const seen = lastSeen();
  try {
    const payload = await fetchWelcome(seen);
    if (shouldShow(seen, payload)) {
      render(payload, seen, false);
      markSeen(payload.version);
      announcing = true;
      navigate('about');
    } else {
      render(await fetchWelcome(null), seen, true);
    }
  } catch (error) {
    // A static-only viewer has no API; the page works without this section.
    console.warn(error);
  }
}
