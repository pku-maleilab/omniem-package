# omniem Python API guide

New here? Start with [the quick start](../README.md) for installation and the
example files, then come back. The [CLI guide](cli.md) covers the command-line
equivalents. Use Python API when you want to keep models in memory, run many
images, share one encoder across heads, or postprocess logits yourself.

There are **two main objects**. The practical difference is the task head.

| Object | What it is | Output |
|---|---|---|
| **`EMEncoder`** | **just the backbone** (the EM-DINO encoder, no head) | features: CLS / patch / inner-block vectors |
| **`OmniEM`** | a backbone encoder **+ a trained task head** | a prediction: segmentation, restoration, or logits |

Start with `EMEncoder`: it is the reusable backbone. `OmniEM` builds on that
encoder by adding a trained task head for segmentation, restoration, or logits.
Both objects use the **same two-tier interface**:

- **`run(image, *, axes, …)`** is the everyday path: a **raw image** in, the
  result out, in one call (prepare → compute → recover).
- **`forward`** (`EMEncoder`) / **`predict`** (`OmniEM`) are the advanced path: pure
  compute on a pre-built **canonical** `[b, z, y, x]` tensor (training loops, tiling).

Inputs are grayscale, so `axes` may only contain `{b, z, y, x}` (no `c`).

## Contents

- [Inputs](#inputs) (read first; applies to both objects)
- [EMEncoder](#emencoder) (backbone only; returns features)
  - [Extract features](#extract-features) · `EMEncoder.run`
  - [Canonical compute (forward)](#canonical-compute-forward) · `EMEncoder.forward`
  - [Persisting an encoder](#persisting-an-encoder)
- [OmniEM](#omniem) (backbone plus task head; returns predictions)
  - [Run a model](#run-a-model) · `OmniEM.run`
  - [Canonical compute (predict)](#canonical-compute-predict) · `OmniEM.predict`
  - [Output layout, channels, and squeeze](#output-layout-channels-and-squeeze)
  - [Controlling output size](#controlling-output-size)
  - [Save weights](#save-weights)
- [Common to both](#common-to-both)
  - [Share one encoder across heads](#share-one-encoder-across-heads)
  - [Catalogs and errors](#catalogs-and-errors)
  - [Build API docs](#build-api-docs)

## Inputs

Both objects take **grayscale** input; the Python API has no channel axis. `axes`
is drawn from `{b, z, y, x}` (batch, depth, row, col); declaring a `c` (channel) axis
raises `InputContractError`. EM is a single intensity per pixel, so the package
synthesises the backbone's channel count internally and you never pass one.

- **Pass floats, and choose the scaling yourself.** `image` must be a floating-point
  `torch.Tensor` / `numpy.ndarray`. Integer inputs are rejected: the package never
  guesses `int → float` (a `uint8` is not always `÷255`). Scale to your working
  range yourself, e.g. `img.astype(np.float32) / 255`.
- **`[0, 1]` is the assumed range.** The default `mean`/`std` (from the config, or
  the arch for the encoder) were fit on `[0, 1]` data, so feed `[0, 1]`-scaled
  pixels, or override `norm=`. Nothing is enforced, but a `[0, 65535]` input with
  the default norm degrades quality.
- **RGB-stored grayscale?** Reduce it to a single plane before calling the Python API
  (e.g. take one channel of an identical-channel RGB array). Only the CLI reduces
  channels for you (`--axes cyx` + `--color-to-gray`).

The **canonical layout** `[b, z, y, x]` mentioned throughout is the channel-less,
z-first (ZYX) tensor the power-path methods (`forward` / `predict`) consume directly;
`z = 1` for a 2D tile.

---

## EMEncoder

`EMEncoder` is the backbone by itself, with no task head. Give it an image and it
returns features (CLS, patch, or inner-block vectors) for downstream code.
The encoder is two-tier: `enc.run` takes a raw image, and `enc.forward` takes a
canonical tensor.

### Extract features

`EMEncoder.run` is the usual path: give it a raw image and get back a feature dict.

```python
import numpy as np
import tifffile
import torch
from omniem import EMEncoder

enc = EMEncoder.load("emdinov1", "weights/backbone_emdino_v1.pt")

img = tifffile.imread("examples/2d_MitoEM_H_0_0_0.tif")
x = torch.from_numpy(img.astype(np.float32) / 255.0)

features = enc.run(x, axes="yx", return_cls=True, return_patch=True)

cls = features["cls"]      # [B, Z, D]  -> here [1, 1, D]
patch = features["patch"]  # [B, Z, N, D]
```

Signature:

```python
enc.run(image, *, axes, norm=None, conform="strict", squeeze="",
        return_cls=True, return_patch=False, return_blocks=None,
        block_callback=None) -> dict
```

**Feature shapes** (before `squeeze`, unfolded from the internal `B*Z` batch):
`cls [B, Z, D]`, `patch [B, Z, N, D]`, `inner[i] [B, Z, T, D]`. Encoder features have
**no spatial XY to mirror**, so this shape follows `[B, Z, …]` **regardless of
`axes`**, unlike `OmniEM`, whose output layout mirrors `axes`. `squeeze` drops a
singleton `b` / `z`: on a 2D tile (`B = Z = 1`), `squeeze="bz"` gives `cls [D]`,
`patch [N, D]`.

- `conform` is `"strict"` or `"resize"` only; `"pad"` is rejected (padded tokens
  would have no clean restore).
- `norm` follows the same scalar-only directive used by OmniEM (`None` →
  arch mean/std; `"per-image"`; `"prenormalized"`; `{"mean": m, "std": s}`).

Inner taps on a 3D example volume:

```python
vol = tifffile.imread("examples/3d_AxonEM-H-0-0-0_0_0_0.tif")
x3d = torch.from_numpy(vol.astype(np.float32) / 255.0)

features = enc.run(x3d, axes="zyx", return_cls=True, return_blocks=[5, 11, 17, 23])
inner_11 = features["inner"][11]   # [B, Z, T, D]
```

### Canonical compute (forward)

`EMEncoder.forward` is the advanced path: pure compute on a pre-built **canonical**
`[b, z, y, x]` tensor (the same layout `OmniEM.predict` uses).

```python
enc.forward(
    tensor,                 # canonical [b, z, y, x], float (z=1 for a 2D tile)
    return_cls=True,
    return_patch=False,
    return_blocks=None,
    block_callback=None,
    squeeze="",
) -> dict
```

`forward` strictly validates the canonical shape and rejects integer tensors. The
optional `block_callback(i, x) -> x` still sees the folded `[B*Z, tokens, dim]` and
must be shape-preserving. Output shapes match `run` (`cls [B, Z, D]`, etc.).

### Persisting an encoder

There is **no `EMEncoder.save_weights`**; `save_weights` (merged / split) is an
`OmniEM`-only convenience (see [Save weights](#save-weights)). An encoder is loaded
from a fixed pretrained backbone checkpoint and is not modified at inference, so you
rarely need to write it back. If you do, `EMEncoder` is a plain `nn.Module`, so use
PyTorch directly. `enc.state_dict()` produces exactly the keys that
`EMEncoder.load(arch, ...)` reads back (the key prefix is the arch's backbone name,
`vit.*` for `emdinov1`):

```python
import torch

torch.save(enc.state_dict(), "weights/backbone_emdino_v1.pt")
enc = EMEncoder.load("emdinov1", "weights/backbone_emdino_v1.pt")
```

---

## OmniEM

`OmniEM` is an encoder plus a trained task head. Give it an image and it returns a
segmentation map, a restored image, or raw logits.

### Run a model

`OmniEM.run` is the usual path: give it a raw image and get back the task output
(or logits).

```python
import numpy as np
import tifffile
import torch
from omniem import OmniEM

omniem = OmniEM.load(
    "model_mito-seg-ViT-L-2D.yaml",
    backbone="weights/backbone_emdino_v1.pt",
    head="weights/head_mito-seg-ViT-L-2D.pt",
)

img = tifffile.imread("examples/2d_MitoEM_H_0_0_0.tif")
x = torch.from_numpy(img.astype(np.float32) / 255.0)  # uint8 -> [0,1]

# Task output (channel collapsed; needs config.task_type). uint8 by default.
labels = omniem.run(x, axes="yx")

# Caller-layout FLOAT logits (channels intact, no task transform).
logits = omniem.run(x, axes="yx", return_logits=True)
```

Signature:

```python
omniem.run(image, *, axes, norm=None, conform="strict", squeeze="",
           dtype=None, return_logits=False) -> torch.Tensor
```

- `OmniEM.load(...)` builds an inference-ready model from a config plus weights.
- `run(...)` takes a **raw float image** and returns output at the caller's
  **original XY** size. `image` must be float (integer arrays/tensors are rejected).
- `axes` describes `image` and drives the output layout (see
  [Output layout, channels, and squeeze](#output-layout-channels-and-squeeze)).
- `return_logits=False` (default) gives the **task output**: the config's
  `task_type` transform (`image2image` → sigmoid+quantize; `image2label` → argmax),
  with the channel axis collapsed. Requires `config.task_type`; without it,
  `run(return_logits=False)` raises (use `return_logits=True` and postprocess
  yourself).
- `return_logits=True` gives the **restored caller-layout FLOAT logits** (the
  predicted `c_out` channel intact, no task transform). Passing a non-`None` `dtype`
  here raises (logits are not quantized).
- `dtype` (`"uint8"` default, or `"uint16"`) picks the task output integer dtype.
- `norm=None` uses config mean/std; `norm="per-image"` z-scores per sample;
  `norm="prenormalized"` skips normalization; `norm={"mean": m, "std": s}` overrides
  it. **`mean`/`std` must be scalars**; per-channel sequences are rejected.
- `conform="strict"` (default), `"pad"`, or `"resize"` controls XY shape handling;
  the output is un-conformed back to the caller's original XY.

### Canonical compute (predict)

`OmniEM.predict` is the advanced path: pure compute on a pre-built **canonical**
`[b, z, y, x]` tensor (channel-less, ZYX, `z = 1` for a 2D model). The canonical
layout is part of the public contract, so training code or advanced callers can
build it directly.

```python
# canonical [b, z, y, x]: float, z == config.img_z (1 for 2D), square +
# multiple-of-stride XY (omniemv1 stride == 112).
canonical = torch.randn(1, 1, 112, 112)
logits = omniem.predict(canonical)   # -> canonical logits [b, c_out, z, y, x]
```

`predict` strictly validates the **shape** (rank 4; `z == config.img_z`; square +
multiple of the arch stride) and **rejects integer tensors**; it never infers
normalization and never scales int→float. It does **no recovery**: it returns
canonical logits `[b, c_out, z, y, x]` on the conformed grid (no un-conform, no
caller-layout reshape). Use `run` when you want caller-shape output.

### Output layout, channels, and squeeze

`axes` (channel-less, from `{b, z, y, x}`) controls the **output layout** of `run`. The
predicted output channel `c_out` is inserted **right after `b` if `axes` has a `b`,
else at the front**, independent of the spatial order:

| `axes` | `run(return_logits=True)` shape |
|---|---|
| `"yx"`   | `[C, Y, X]` |
| `"zyx"`  | `[C, Z, Y, X]` |
| `"byx"`  | `[B, C, Y, X]` |
| `"bzyx"` | `[B, C, Z, Y, X]` |
| `"byxz"` | `[B, C, Y, X, Z]` |

The task output (`return_logits=False`) collapses `C`, so the same layouts drop the
channel: `"yx"` → `[Y, X]`, `"bzyx"` → `[B, Z, Y, X]`, etc.

A `b` / `z` axis that the caller did **not** name is dropped only when **singleton**; if
`axes` omits `b` but the batch is `> 1` (or omits `z` but depth is `> 1`), `run`
raises rather than silently dropping data.

`squeeze` is a subset of `{b, z}`: each named axis must exist in the restored layout
and be singleton (else raise). `c` / `x` / `y` and duplicates are rejected. Default
`squeeze=""` mirrors `axes`. *(`squeeze` works the same way on encoder features; see
[Extract features](#extract-features).)*

### Controlling output size

`OmniEM` models are **shape-preserving**: the output lands at the XY size of the
input. There is no separate API parameter for output size; **you choose the resized
shape** by resizing the input yourself before `run`. Arbitrary shapes can
give poor results (the model was trained at a fixed ROI), so prefer its native size
or a clean integer multiple of it.

The CLI `omniem infer --output-scale F` mirrors this; in Python you do the resize
explicitly.

**Super-resolution: upscale the input, then enhance.** Resize the input up
(bicubic), run the model, and the output lands at the larger size. Use
`conform="resize"` (the default `conform="strict"` rejects a non-square /
non-stride-multiple XY), or pick a size that is already a multiple of the stride:

```python
import torch.nn.functional as F

x = torch.from_numpy(img.astype(np.float32) / 255.0)   # [Y, X] in [0,1]
# Pass an explicit size=(round(F*Y), round(F*X)); this matches the CLI's
# `--output-scale F` exactly. (scale_factor= floors and would diverge for odd
# sizes / fractional factors.)
factor = 1.5
new_y, new_x = round(factor * x.shape[-2]), round(factor * x.shape[-1])
up = F.interpolate(x[None, None], size=(new_y, new_x), mode="bicubic",
                   align_corners=False)[0, 0]          # [new_y, new_x]

out = omniem.run(up, axes="yx", conform="resize")        # task output at upscaled size
```

**Quick inference: downscale, infer, then resize the output back.** For a large 2D
image, run the model on a smaller input and upsample the result (a speed/quality
trade-off). Use `mode="nearest"` for `image2label` outputs and `mode="bicubic"` for
`image2image`:

```python
orig_y, orig_x = x.shape[-2], x.shape[-1]
factor = 0.5
new_y, new_x = round(factor * orig_y), round(factor * orig_x)
down = F.interpolate(x[None, None], size=(new_y, new_x), mode="bicubic",
                     align_corners=False)[0, 0]

small = omniem.run(down, axes="yx", conform="resize")    # task output at smaller size

# image2label -> nearest; image2image -> bicubic
out = F.interpolate(small[None, None].float(), size=(orig_y, orig_x),
                    mode="nearest")[0, 0]
```

**3D caveat.** Avoid this for 3D (`zyx`) volumes: resizing XY only leaves Z
untouched, which changes in-plane spatial information relative to Z (anisotropy) with
no Z alignment / resampling, and tends to give worse results.

### Save weights

```python
# merged
omniem.save_weights("model.pt")

# split
omniem.save_weights(
    backbone="weights/backbone.pt",
    head="weights/head.pt",
)
```

## Common to both

### Share one encoder across heads

Load one `EMEncoder` and borrow it (by reference) across several `OmniEM` heads.
This avoids copying the backbone, so the ViT-L weights live in memory once.

```python
from omniem import EMEncoder, OmniEM

enc = EMEncoder.load("emdinov1", "weights/backbone_emdino_v1.pt", device="cuda")

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

- `encoder=` injects the encoder's backbone by reference; no backbone copy is made.
- With `encoder=`, pass only head weights (`head=` for `load`, `head_weights=` for
  `from_config`).
- Do not combine `encoder=` with `weights=`, `backbone=`, or `encoder_weights=`.
- The borrowed encoder must already be eval, frozen, and on the target device/dtype.
  `EMEncoder.load(...)` returns it that way.
- Borrowed `OmniEM` instances reject whole-model mutators that would touch the shared
  encoder. Relocate the `EMEncoder`, then rebuild the borrowed model.

### Catalogs and errors

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

All package-specific errors inherit from `omniem.errors.OmniEMError`, so you can
catch one base class when you want simple error handling.

| Error | Typical cause |
|---|---|
| `ConfigError` | bad YAML, missing config file, unknown architecture, schema problem |
| `WeightFormatError` | unreadable checkpoint, wrong keys, non-tensor values, shape mismatch |
| `InputContractError` | a `c` axis, integer input, non-canonical `predict`/`forward` tensor, non-conforming XY in strict mode, `--unit-range` + `--scale` |
| `MissingExtraError` | optional dependency missing for a later-stage feature |
| `OOMError` | out-of-memory surface for later tiling/streaming paths |

There is also `omniem.OmniEMWarning` (a `UserWarning`, importable from `omniem` or
`omniem.errors`): a **warn-only** advisory emitted when a scaled input or a config /
override `mean`/`std` falls outside `[0, 1]`. It never stops a run; promote it with
`warnings.simplefilter("error", OmniEMWarning)` if you want it to raise.

### Build API docs

The API reference is generated from the **installed** package's docstrings with
`pdoc`. Because `pdoc` imports `omniem` by name, this works from a plain
`pip install`, with no repository checkout needed:

```bash
python -m pip install "pdoc>=14"
python -m pdoc omniem -d google --no-show-source -o omniem-api
```

Open `omniem-api/omniem.html`. Run it in the same environment where `omniem` (and its
dependencies: torch, numpy, monai, …) is importable, since `pdoc` imports the package
to read its docstrings.

If you cloned the repository, `scripts/build_docs.sh [OUTPUT_DIR]` wraps the same
command (default output `docs/api/`, which is gitignored).
