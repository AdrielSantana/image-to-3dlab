"""Static wiring checks for the Finish mode and the Pixal3D backend option.

`finish.js` touches the DOM at import time, so it cannot be executed under Node the way
`job-progress.js` is. The failure mode worth catching is cheaper than that anyway: a
`getElementById` in the module that no longer matches an id in `index.html` silently yields
null, and the panel breaks at the moment someone tries to use it. These tests read both
files and compare.
"""

from __future__ import annotations

import re
from pathlib import Path

VIEWER = Path(__file__).resolve().parents[1] / "viewer"
INDEX = (VIEWER / "index.html").read_text()
APP = (VIEWER / "app.js").read_text()
FINISH = (VIEWER / "modes" / "finish.js").read_text()
GENERATE = (VIEWER / "modes" / "generate.js").read_text()


def _element_ids(markup: str) -> set[str]:
    return set(re.findall(r'id="([^"]+)"', markup))


def test_every_id_the_finish_mode_looks_up_exists_in_the_page():
    referenced = set(re.findall(r"f\('([^']+)'\)", FINISH))
    assert referenced, "the module should look up some elements"
    missing = sorted(referenced - _element_ids(INDEX))
    assert not missing, f"finish.js references ids absent from index.html: {missing}"


def test_the_finish_mode_is_imported_and_registered():
    assert "import './modes/finish.js';" in APP
    assert "finish: byId('finish-view')" in APP
    assert "modes.finish.hidden = activeMode !== 'finish';" in APP


def test_the_page_has_a_finish_button_and_section():
    ids = _element_ids(INDEX)
    assert "mode-finish" in ids
    assert "finish-view" in ids


def test_the_finish_mode_posts_to_the_documented_endpoints():
    assert "'/api/finish'" in FINISH
    assert "/api/finish/${jobId}/status" in FINISH
    assert "/api/finish/${payload.job_id}/cancel" in FINISH


def test_pixal3d_is_offered_as_a_backend_with_its_settings():
    assert '<option value="pixal3d">' in INDEX
    assert 'data-backend="pixal3d"' in INDEX
    ids = _element_ids(INDEX)
    for field in ("pixal3d-res", "pixal3d-seed", "pixal3d-fov"):
        assert field in ids


def test_the_generate_mode_reads_the_pixal3d_fields_that_exist():
    block = GENERATE[GENERATE.index("pixal3d: () => ({"):]
    block = block[: block.index("}),")]
    referenced = set(re.findall(r"g\('([^']+)'\)", block))
    assert referenced == {"pixal3d-res", "pixal3d-seed", "pixal3d-fov"}
    assert not referenced - _element_ids(INDEX)


def test_the_mlx_attention_choice_is_still_offered():
    """The three routes this viewer exposes: MLX attention, finishing, and Pixal3D."""
    assert 'id="generate-attention"' in INDEX
    assert 'value="mlx"' in INDEX and 'value="mlx-fp16"' in INDEX


def test_skipping_the_repaint_hides_only_the_paint_fields():
    assert "finish-paint-fields" in _element_ids(INDEX)
    assert "f('finish-paint-fields').hidden = f('finish-skip-paint').checked;" in FINISH


def test_every_progress_track_has_a_styled_fill():
    """The bar inside a .progress-track must be reachable by a rule that paints it.

    `#generate-overall-bar` was the only selector, so the Finish panel's identically
    structured bar rendered as an empty groove for the whole run (found 2026-09-21).
    """
    css = (VIEWER / "styles" / "generate.css").read_text()
    assert ".progress-track > div" in css, "the fill is styled per-id, so a new track shows nothing"

    tracks = re.findall(r'<div class="progress-track"><div id="([^"]+)"></div></div>', INDEX)
    assert len(tracks) >= 2, "expected the Generate and Finish tracks to share this markup"


def test_the_finish_mode_can_list_and_resume_runs_on_disk():
    # A job registry lives in the server's memory; the run directories outlive it, and
    # this list is the only way back to a run whose browser tab was closed.
    assert "/api/finish/runs" in FINISH
    assert "/resume" in FINISH
