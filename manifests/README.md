# Run manifests

A manifest records everything about one generation — input, backend, parameters, and the
licensing intent the run was gated on — so a result can be reproduced and audited later.
It is the preferred way to run the pipeline:

    python pipeline.py --run-manifest manifests/my-run.json

## Start here

Copy [`example-trellis2.json`](example-trellis2.json), point `input.path` at your own
image, and run it. **Paths inside a manifest are resolved relative to the manifest file**,
not the working directory — so `"../cat.png"` means a `cat.png` sitting in the repo root.

You can also skip manifests entirely and pass an image directly:

    python pipeline.py --trellis your-image.png

## Why there is only one manifest here

Manifests we write for our own runs point at source art in `assets_to_test/`, which is
not part of the repository — so they would fail for anyone else with:

    error: input image does not exist: .../assets_to_test/<something>.png

Rather than ship files that cannot run, `manifests/*.json` is git-ignored with the
template as the single exception. Your own manifests land here and stay local unless you
choose to track them.

## Required fields

`schema_version` must be `1`. `use_case` is `showcase` or `game`; `distribution` is
`private`, `public` or `worldwide`; `commercial_intent` is a boolean. Together with
`license_policy`, these gate generation *before* any model loads — see
`image_to_3dlab/provenance.py`. Outputs land in license-class folders under
`output.directory` with a `.provenance.json` sidecar.
