// Wording for the "does this image have a cut-out?" badge. Only backends that require a
// cut-out (TRELLIS) offer the rembg checkbox; the rest remove the background themselves,
// so telling their users to enable rembg sends them looking for an option that isn't there.
export function alphaBadge(hasAlpha, requiresAlpha) {
  if (hasAlpha) return { tone: 'alpha-good', text: 'transparent foreground ✓' };
  if (requiresAlpha) return { tone: 'alpha-bad', text: 'no alpha — enable rembg to continue' };
  return { tone: 'alpha-good', text: 'no alpha — background will be removed automatically' };
}
