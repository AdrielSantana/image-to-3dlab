// Whether a Setup card still needs its "this model is gated" warning. Once the Hugging Face
// sign-in check says the account can download the gated repo, the warning is old news and
// only contradicts the green "✓ access" a few lines above it. Any other caveat (Hunyuan's
// territorial limits, say) has nothing to do with sign-in and always shows.
export function showsCaveat(backend, access) {
  if (!backend.caveat) return false;
  if (!backend.gated_repo) return true;
  return access[backend.gated_repo] !== 'yes';
}
