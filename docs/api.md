# omniem Python API guide

The CLI is a thin wrapper around the Python API. Use Python when you want to keep
models in memory, run multiple images, share encoders, or postprocess logits
yourself.

Start with [the quick start](../README.md) for installation, model files, and
model config YAMLs. Use [the CLI guide](cli.md) when you want command-line
examples.

## Contents

- [Run A Model](#run-a-model)
- [Split The Input Stage](#split-the-input-stage)
- [Controlling Output Size (Super-Resolution & Quick Inference)](#controlling-output-size-super-resolution--quick-inference)
- [Save Weights](#save-weights)
- [Share One Encoder Across Heads](#share-one-encoder-across-heads)
- [Extract Encoder Features](#extract-encoder-features)
- [Catalogs And Errors](#catalogs-and-errors)
- [Build API Docs](#build-api-docs)

## Run A Model

```python
import numpy as np
import tifffile
import torch
from omniem import OmniEM

model = OmniEM.load(
    "model_mito-seg-ViT-L-2D.yaml",
    backbone="weights/backbone_emdino_v1.pt",
    head="weights/head_mito-seg-ViT-L-2D.pt",
)

img = tifffile.imread("examples/2d_MitoEM_H_0_0_0.tif")
x = torch.from_numpy(img.astype(np.float32) / 255.0)  # uint8 -> [0,1]

logits = model.predict(x, axes="yx")
labels = model.apply_output(logits, axes="yx", dtype="uint8")
```

Main points:

- `OmniEM.load(...)` builds an inference-ready model from config plus weights.
- `predict(...)` returns pure logits and preserves the caller's original shape.
- `apply_output(...)` applies the config's `task_type` transform and collapses the
  channel axis.
- `norm=None` uses config mean/std; `norm="per-image"` z-scores per sample;
  `norm="prenormalized"` skips normalization;
  `norm={"mean": m, "std": s}` overrides it.
- `conform="strict"`, `"pad"`, or `"resize"` controls XY shape handling.

## Split The Input Stage

Use `apply_input` when you want to inspect or cache the prepared tensor before
model compute.

```python
prepared = model.apply_input(x, axes="yx", conform="pad")
logits = model.predict(prepared)
labels = model.apply_output(logits, axes="yx")
```

## Controlling Output Size (Super-Resolution & Quick Inference)

omniem models are **shape-preserving**: `predict` returns its output at the XY
size of the input it was given. There is no API parameter for output size — **the
resized shape is your choice**, applied by resizing the input yourself before
`predict`. Arbitrary shapes can give poor results (the model was trained at a
fixed ROI), so prefer the model's native size or a clean integer multiple of it.

The CLI `omniem infer --output-scale F` mirrors this; in Python you do the resize
explicitly.

**Super-resolution — upscale the input, then enhance.** Resize the input up
(bicubic), run the model, and the output lands at the larger size. Use
`conform="resize"` (the default `predict` `conform="strict"` rejects a
non-square / non-stride-multiple XY), or pick a size that is already a multiple of
the model stride:

```python
import torch.nn.functional as F

x = torch.from_numpy(img.astype(np.float32) / 255.0)   # [Y, X] in [0,1]
# Pass an explicit size=(round(F*Y), round(F*X)) — this matches the CLI's
# `--output-scale F` exactly. (scale_factor= floors and would diverge for odd
# sizes / fractional factors.)
factor = 1.5
new_y, new_x = round(factor * x.shape[-2]), round(factor * x.shape[-1])
up = F.interpolate(x[None, None], size=(new_y, new_x), mode="bicubic",
                   align_corners=False)[0, 0]          # [new_y, new_x]

logits = model.predict(up, axes="yx", conform="resize")
out = model.apply_output(logits, axes="yx", dtype="uint8")  # at the upscaled size
```

**Quick inference — downscale, infer, then resize the output back.** For a large
2D image, run the model on a smaller input and upsample the result (a
speed/quality trade-off). Use `mode="nearest"` for `image2label` outputs and
`mode="bicubic"` for `image2image`:

```python
orig_y, orig_x = x.shape[-2], x.shape[-1]
factor = 0.5
new_y, new_x = round(factor * orig_y), round(factor * orig_x)
down = F.interpolate(x[None, None], size=(new_y, new_x), mode="bicubic",
                     align_corners=False)[0, 0]

logits = model.predict(down, axes="yx", conform="resize")
small = model.apply_output(logits, axes="yx", dtype="uint8")

# image2label -> nearest; image2image -> bicubic
out = F.interpolate(small[None, None].float(), size=(orig_y, orig_x),
                    mode="nearest")[0, 0]
```

**3D caveat.** Do **not** apply this to 3D (`zyx`) volumes: resizing XY only
leaves Z untouched, which changes in-plane spatial information relative to Z
(anisotropy) with no Z alignment / resampling, and tends to give worse results.

## Save Weights

```python
# merged
model.save_weights("model.pt")

# split
model.save_weights(
    backbone="weights/backbone.pt",
    head="weights/head.pt",
)
```

## Share One Encoder Across Heads

```python
from omniem import EMEncoder, OmniEM

enc = EMEncoder.load(
    "emdinov1",
    "weights/backbone_emdino_v1.pt",
    device="cuda",
)

seg = OmniEM.load(
    "model_mito-seg-ViT-L-2D.yaml",
    head="weights/head_mito-seg-ViT-L-2D.pt",
    encoder=enc,
)

denoise = OmniEM.load(
    "model_denoise-emdiffuse-l.yaml",
    head="weights/head_denoise-emdiffuse-l.pt",
    encoder=enc,
)
```

Borrowing rules:

- `encoder=` injects `encoder.vit` by reference; no backbone copy is made.
- With `encoder=`, pass only head weights (`head=` for `load`, `head_weights=`
  for `from_config`).
- Do not combine `encoder=` with `weights=`, `backbone=`, or `encoder_weights=`.
- The borrowed encoder must already be eval, frozen, and on the target
  device/dtype. `EMEncoder.load(...)` returns it that way.
- Borrowed `OmniEM` models reject whole-model mutators that would touch the
  shared encoder. Relocate the `EMEncoder`, then rebuild borrowed models.

## Extract Encoder Features

```python
import numpy as np
import tifffile
import torch
from omniem import EMEncoder

enc = EMEncoder.load("emdinov1", "weights/backbone_emdino_v1.pt")

img = tifffile.imread("examples/2d_MitoEM_H_0_0_0.tif")
x = torch.from_numpy(img.astype(np.float32) / 255.0)

features = enc(
    x,
    axes="yx",
    return_cls=True,
    return_patch=True,
)

cls = features["cls"]
patch = features["patch"]
```

For inner taps on a 3D example volume:

```python
vol = tifffile.imread("examples/3d_AxonEM-H-0-0-0_0_0_0.tif")
x3d = torch.from_numpy(vol.astype(np.float32) / 255.0)

features = enc(
    x3d,
    axes="zyx",
    return_cls=True,
    return_blocks=[5, 11, 17, 23],
)
inner_11 = features["inner"][11]
```

## Catalogs And Errors

```python
from omniem import list_encoders, list_models, arch_info, model_arch_info
from omniem.errors import OmniEMError

print(list_encoders())
print(list_models())

try:
    ...
except OmniEMError as exc:
    print(exc)
```

All package-specific errors inherit from `omniem.errors.OmniEMError`.

| Error | Typical cause |
|---|---|
| `ConfigError` | bad YAML, missing config file, unknown architecture, schema problem |
| `WeightFormatError` | unreadable checkpoint, wrong keys, non-tensor values, shape mismatch |
| `InputContractError` | bad axes, non-conforming XY in strict mode, `--unit-range` + `--scale` |
| `MissingExtraError` | optional dependency missing for a later-stage feature |
| `OOMError` | out-of-memory surface for later tiling/streaming paths |

There is also `omniem.OmniEMWarning` (a `UserWarning`, importable from `omniem` or
`omniem.errors`) — a **warn-only** advisory emitted when a scaled input or a config /
override `mean`/`std` falls outside `[0, 1]`. It never stops a run; promote it with
`warnings.simplefilter("error", OmniEMWarning)` if you want it to raise.

```python
from omniem.errors import OmniEMError

try:
    ...
except OmniEMError:
    ...
```

## Build API Docs

The API reference is generated from the **installed** package's docstrings with
`pdoc`. Because `pdoc` imports `omniem` by name, this works from a plain
`pip install` — no repository checkout is needed:

```bash
python -m pip install "pdoc>=14"
python -m pdoc omniem -d google --no-show-source -o omniem-api
```

Open `omniem-api/omniem.html`. Run it in the same environment where `omniem`
(and its dependencies: torch, numpy, monai, …) is importable, since `pdoc`
imports the package to read its docstrings.

If you cloned the repository, `scripts/build_docs.sh [OUTPUT_DIR]` wraps the same
command (default output `docs/api/`, which is gitignored).
