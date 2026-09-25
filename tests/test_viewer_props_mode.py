"""Static wiring checks for the Props mode.

`props.js` touches the DOM at import time, so like `finish.js` it is checked by reading
it beside `index.html`: a `getElementById` that matches no id yields null, and the tab
breaks the moment someone uses it. The endpoints are checked against the server's own
routes for the same reason.
"""

from __future__ import annotations

import re
from pathlib import Path

VIEWER = Path(__file__).resolve().parents[1] / "viewer"
INDEX = (VIEWER / "index.html").read_text()
APP = (VIEWER / "app.js").read_text()
PROPS = (VIEWER / "modes" / "props.js").read_text()
SERVER = (VIEWER / "generate_api.py").read_text()


def _element_ids(markup: str) -> set[str]:
    return set(re.findall(r'id="([^"]+)"', markup))


def test_every_id_the_props_mode_looks_up_exists_in_the_page():
    referenced = set(re.findall(r"f\('([^']+)'\)", PROPS))
    assert referenced, "the module should look up some elements"
    missing = sorted(referenced - _element_ids(INDEX))
    assert not missing, f"props.js references ids absent from index.html: {missing}"


def test_the_props_mode_is_imported_registered_and_has_a_button():
    assert "import './modes/props.js';" in APP
    assert "props: byId('props-view')" in APP
    assert "modes.props.hidden = activeMode !== 'props';" in APP
    assert {"mode-props", "props-view"} <= _element_ids(INDEX)


def test_its_stylesheet_is_linked():
    assert (VIEWER / "styles" / "props.css").is_file()
    assert '<link rel="stylesheet" href="./styles/props.css">' in INDEX


def test_the_props_mode_uses_the_routes_the_server_has():
    for url in ("'/api/props'", "'/api/props/runs'", "/api/props/${jobId}/status",
                "/api/props/${payload.job_id}/cancel", "/turn`"):
        assert url in PROPS, url
    assert 'parts == ["api", "props"]' in SERVER
    assert 'parts == ["api", "props", "runs"]' in SERVER
    assert 'parts[:3] == ["api", "props", "runs"] and parts[4] == "turn"' in SERVER
    assert 'parts[:2] == ["api", "props"] and parts[3] == "cancel"' in SERVER


def test_the_progress_track_shares_the_styled_markup():
    assert '<div class="progress-track"><div id="props-overall-bar"></div></div>' in INDEX


def test_a_missing_gltfpack_is_said_rather_than_silently_skipped():
    assert "gltfpack not found" in PROPS
    assert "vendor/gltfpack/gltfpack" in PROPS


def test_a_near_45_degree_turn_offers_the_quarter_turn_fix():
    assert "prop.yaw_tie" in PROPS
    assert "Turn 90°" in PROPS
    assert "degrees: 90" in PROPS


def test_the_preview_is_the_restricted_compare_page():
    assert "/viewer/index.html?a=${encodeURIComponent(src)}" in PROPS
    assert "restricted=1" in PROPS


def test_a_generation_and_a_prop_bake_refuse_to_share_the_machine():
    """Both hold gigabytes in the same unified memory, so neither starts over the other."""
    assert "def _props_busy(self)" in SERVER
    assert "a generation is running; wait for it to finish" in SERVER
    assert '"a prop sheet is baking; wait for it to finish"' in SERVER


def test_the_tab_sits_under_the_menu_bar_like_generate():
    """It was a plain section, so its heading scrolled under the fixed 42px menu."""
    css = (VIEWER / "styles" / "props.css").read_text()
    assert "#props-view { position: fixed; inset: 42px 0 0 0; overflow: auto;" in css


def test_every_prop_gets_a_chip_and_one_detail_panel():
    assert "chip.onclick = () => showProp(run, prop);" in PROPS
    assert {"props-list", "props-detail", "props-frame", "props-turn", "props-lods-body"} <= _element_ids(INDEX)
