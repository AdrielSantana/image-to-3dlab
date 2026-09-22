"""No test may hand-load a module that production code imports by name.

Python guarantees one object per module *name*, and everything that reads a module's
state depends on it. `importlib.util.spec_from_file_location` plus a write to
`sys.modules` breaks the guarantee: it creates a second, independent copy. If production
code imported the first one and a test then patches the second, the patch is applied to an
object nobody is looking at -- and the test passes while testing nothing.

That is not a theory. `tests/test_download_api.py` patched `backend_catalog` while
`download_api` held a different copy, so "an unsupported machine is refused before
anything downloads" passed without refusing anything. It only failed in a file order the
alphabetical suite never takes, so it stayed green for weeks.

Hand-loading is fine where **nothing else imports the module** -- a private copy that no
one shares cannot diverge from anyone, and most of this suite is in that position. The
rule this pins is narrower: the moment production code imports a name, tests of that name
must use a plain `import` (which `tests/conftest.py` makes possible) rather than a copy.

**Known blind spot:** the check reads source text, so it only sees the literal spelling
`sys.modules["name"] =`. Aliasing the import (`import sys as s; s.modules[...]`) slips
past it. That is the spelling every file in this suite uses, and a convention check that
covers the convention is worth more than no check; it is not a sandbox.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TESTS = REPO / "tests"
PRODUCTION = ("viewer", "scripts", "image_to_3dlab")

# `sys.modules["name"] = ...`, the line that publishes a hand-loaded copy under a name
# other code may already be holding.
REGISTERS = re.compile(r'sys\.modules\[\s*["\'](?P<name>[\w.]+)["\']\s*\]\s*=')
# `import name` / `from name import ...`, at the start of a line so a mention inside a
# string or comment does not count.
def _imports(module: str) -> re.Pattern[str]:
    return re.compile(rf'^\s*(?:from\s+{re.escape(module)}\s+import|import\s+{re.escape(module)})\b',
                      re.MULTILINE)


def _production_files() -> list[Path]:
    return [p for d in PRODUCTION for p in (REPO / d).rglob("*.py") if p.is_file()]


def test_no_test_hand_loads_a_module_that_production_code_imports():
    production = [(p, p.read_text(encoding="utf-8", errors="ignore")) for p in _production_files()]
    offences = []
    for test_file in sorted(TESTS.glob("test_*.py")):
        # This file quotes the offending line as an example, so scanning it would report
        # its own fixture. Everything else is fair game.
        if test_file.name == Path(__file__).name:
            continue
        for name in set(REGISTERS.findall(test_file.read_text(encoding="utf-8", errors="ignore"))):
            pattern = _imports(name)
            importers = [p.relative_to(REPO).as_posix() for p, text in production if pattern.search(text)]
            if importers:
                offences.append(
                    f"{test_file.name} hand-loads {name!r}, which is imported by "
                    + ", ".join(sorted(importers))
                    + " -- use a plain `import` so both hold the same object"
                )
    assert not offences, "\n".join(offences)


def test_the_check_can_actually_see_an_offence():
    """A guard that never fires is indistinguishable from a guard that cannot."""
    assert REGISTERS.search('sys.modules["download_api"] = module').group("name") == "download_api"
    assert _imports("download_api").search("from download_api import DOWNLOADS\n")
    assert _imports("download_api").search("import download_api as dl\n")
    # A mention that is not an import must not count, or the rule would fire on prose.
    assert not _imports("download_api").search("# download_api is loaded elsewhere\n")
