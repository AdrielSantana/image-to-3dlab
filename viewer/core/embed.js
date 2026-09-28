// The Generate tab previews its result by loading this same page in an iframe with
// ?restricted=1. That copy must be a bare 3D view: no tab bar, no About, no Setup landing,
// no update banner -- otherwise the whole app appears nested inside its own panel.
export function isEmbedded(search) {
  return new URLSearchParams(search).get('restricted') === '1';
}

export function landingMode({ embedded, skipSetup }) {
  if (embedded) return 'compare';
  return skipSetup ? 'generate' : 'setup';
}
