# Images in docs

Screenshots and renders that a doc refers to. Kept here rather than in a scratch folder so
a reader months later sees what the text is talking about.

**This repository is deliberately slim** (it was cut from 244 MB to 8.5 MB in September
2026), so images earn their place the same way scripts do:

- Only add an image that makes a claim checkable. A comparison, a defect, a before and
  after. Not decoration.
- **JPEG, at most ~1600px on the long edge, quality around 65.** That lands a three-panel
  viewer comparison near 200-300 KB. Never commit a raw retina PNG screenshot; they run to
  several megabytes each.
- Name by what is being compared, not by the asset or the date:
  `attention-backends-same-seed.jpg`, not `fox-screenshot-2.png`. The asset is an example;
  the comparison is the point.
- macOS screenshot temp folders are cleaned aggressively. Save into this directory
  directly, or the file will be gone within the hour.

Convert with `sips`:

```
sips -Z 1600 -s format jpeg -s formatOptions 65 input.png --out docs/images/name.jpg
```
