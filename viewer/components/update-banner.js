// --- Update banner content -----------------------------------------------------------------
// Whether to show "a new version is out", decided without touching the page so Node can
// test it. The check itself is GET /api/update-check (viewer/update_api.py).

/** The banner to show, or null. Dismissing hides it until the next release. */
export function bannerFor(check, dismissedVersion) {
  if (!check || !check.enabled || !check.newer || !check.latest) return null;
  if (dismissedVersion === check.latest) return null;
  return {
    text: `Version ${check.latest} is out (you have ${check.current}). To update, run this again:`,
    command: check.command,
    url: check.url,
    version: check.latest,
  };
}
