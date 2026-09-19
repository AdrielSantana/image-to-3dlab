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

## The other files here are our run records, not examples

Every other manifest in this folder is a real run we made, kept for traceability. Their
`input.path` points into `assets_to_test/`, which holds our own source art and is **not
part of the repository** — so running one as-is fails with:

    error: input image does not exist: .../assets_to_test/<something>.png

That is expected. Read them for the parameter combinations that worked; do not expect
them to run unmodified.

## Required fields

`schema_version` must be `1`. `use_case` is `showcase` or `game`; `distribution` is
`private`, `public` or `worldwide`; `commercial_intent` is a boolean. Together with
`license_policy`, these gate generation *before* any model loads — see
`image_to_3dlab/provenance.py`. Outputs land in license-class folders under
`output.directory` with a `.provenance.json` sidecar.
