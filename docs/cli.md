# omniem CLI guide

This page covers the command-line interface.

New here? Start with [the quick start](../README.md) for installation and the
example files, then come back. The [Python API guide](api.md) covers the
programmatic equivalents. Use CLI when you want to extract encoder features, run
OmniEM from a shell script, process files directly, inspect available models, or
split and merge weight files without writing Python code.

## Contents

- [CLI Overview](#cli-overview)
- [EMEncoder Features](#emencoder-features)
- [OmniEM Inference](#omniem-inference)
- [Split / Merge Weights](#split--merge-weights)

## CLI Overview

```bash
omniem --help
omniem --version
omniem list-encoders
omniem list-models
omniem features ...
omniem infer ...
omniem split ...
omniem merge ...
```

Which command should you use?

| Command | Object | What it does |
|---|---|---|
| `omniem features` | **`EMEncoder`** (encoder) | extract EM-DINO backbone features (CLS / patch / inner) |
| `omniem infer` | **`OmniEM`** (model) | run a model on an image/volume → segmentation, restoration, or logits |
| `omniem split` / `merge` | model weight files | convert between a merged `.pt` and a `backbone` + `head` pair |
| `omniem list-models` / `list-encoders` | catalogs | list the built-in model / encoder architectures |

CLI conventions:

- successful commands print the output path to stdout;
- user-facing errors print to stderr and exit with code `2`;
- output files and JSON sidecars are not overwritten unless you pass `--force`.

## EMEncoder Features

Use `omniem features` when you want encoder outputs without an OmniEM head.

```bash
omniem features \
  -i IMG \
  --arch emdinov1 \
  --weights BACKBONE.pt \
  --want cls,patch \
  -o OUT.npz
```

### Important Arguments

| Argument | Meaning |
|---|---|
| `-i`, `--input` | input image, `.tif`, `.tiff`, or `.npy` |
| `--arch NAME` | encoder architecture; see `omniem list-encoders` |
| `--weights PATH` | raw encoder backbone checkpoint (for `emdinov1`, a `vit.*` file) |
| `--axes AXES` | explicit layout, such as `yx`, `zyx`, `cyx`, `czyx`, `bcyx` |
| `--color-to-gray` | average a declared channel axis into grayscale |
| `--want FIELDS` | comma-separated subset of `cls,patch,inner`; default `cls` |
| `--blocks i,j` | inner block ids; requires `--want inner` |
| `--scale dtype\|max\|S` | scale axis -> `[0,1]` (default `dtype`); or `--unit-range` to skip |
| `--norm model\|per-image\|prenormalized` | norm source (default `model` = arch mean/std); or `--mean M --std S` |
| `-o`, `--output` | `.npz` for multiple fields or `.npy` for one field |
| `--force` | overwrite existing output and sidecar |

### Example: CLS And Patch Features

```bash
omniem features \
  -i "$EXAMPLES/2d_MitoEM_H_0_0_0.tif" \
  --arch emdinov1 \
  --weights "$WEIGHTS/backbone_emdino_v1.pt" \
  --want cls,patch \
  -o out/features.npz
```

This writes:

- `out/features.npz`, with arrays named `cls` and `patch`;
- `out/features.npz.json`, with arch, weights, axes, scaling, normalization, and
  selector metadata.

### Example: Inner Block Features

```bash
omniem features \
  -i examples/2d_MitoEM_H_0_0_0.tif \
  --arch emdinov1 \
  --weights backbone.pt \
  --want cls,inner \
  --blocks 5,11,17,23 \
  -o out/inner_features.npz
```

`--blocks` and `--want inner` must be used together. Inner outputs are written as
keys such as `inner_5`, `inner_11`, and so on.

### Example: 3D Volume Features

```bash
omniem features \
  -i examples/3d_AxonEM-H-0-0-0_0_0_0.tif \
  --axes zyx \
  --arch emdinov1 \
  --weights backbone.pt \
  --want cls \
  -o out/axon_cls.npy
```

For 3D inputs, each XY slice is encoded by the same backbone, and the resulting
features keep their z ordering.

## OmniEM Inference

Use `omniem infer` when you want to run an OmniEM model on one image or volume.

```bash
omniem infer \
  -i IMG \
  -m CONFIG.yaml \
  (--weights MERGED.pt | --backbone BACKBONE.pt --head HEAD.pt) \
  [--axes AXES] [--color-to-gray] \
  [--scale dtype|max|S | --unit-range] \
  [--norm model|per-image|prenormalized | --mean M --std S] \
  [--conform strict|pad|resize] [--output-scale F] \
  -o OUT.tif
```

### Input Handling

The inference CLI reads an image, applies input scaling, applies the selected
normalization, then runs OmniEM. OmniEM configs use `[0, 1]`-domain
`mean`/`std`, so the default path scales integer EM images into `[0, 1]` before
the fixed OmniEM normalization runs.

#### Input Scaling (`--scale`)

`--scale` brings the input into `[0, 1]`. It defaults to `dtype`, so common
integer images work without you choosing a divisor.

| `--scale` | meaning |
|---|---|
| `dtype` (default) | divide by the dtype max (`uint8` -> 255, `uint16` -> 65535); `bool`/float pass through |
| `max` | divide by the input's own maximum (data-dependent) |
| `<float>` | divide by that positive number |
| `--unit-range` | the input is already `[0, 1]`; skip scaling (a `[0, 1]` check still warns) |

Signed-integer inputs divide into `[-1, 1]` and trigger a warning (warn-only). If
the scaled input falls outside `[0, 1]` under model/argument norm, `omniem`
warns (`OmniEMWarning`) but still runs.

Python API callers pass float tensors directly:

```python
x = image.astype("float32") / 255.0   # uint8 -> [0,1]
```

See [the Python API guide](api.md) for the full Python surface.

#### Normalization (`--norm`)

After scaling, `omniem` normalizes the image. The norm source is independent of the scale
axis:

| `--norm` | meaning |
|---|---|
| `model` (default) | OmniEM config `mean`/`std` |
| `per-image` | per-sample z-score `(x - mean(x)) / std(x)`; scale-invariant, so the scale axis is skipped |
| `prenormalized` | the input is already normalized; skip the affine and scaling (cast only) |
| `--mean M --std S` | override mean/std for this run (mutually exclusive with `--norm`) |

In the Python API this is the `norm=` argument: `None` (model) ·
`"per-image"` · `"prenormalized"` · `{"mean": m, "std": s}`.

#### Axes

OmniEM is **channel-less** (grayscale). `omniem infer` accepts `--axes` to
declare the input layout (chars from `{b, c, z, y, x}`, `y` and `x` required),
matching the `features` command:

- omit `--axes` → 2D input is `yx`, 3D input is `zyx`;
- give `--axes` → any permutation, e.g. `yx`, `zyx`, `cyx`, `czyx`, `bcyx`.

A declared `c` (color) axis is reduced to grayscale **before** the channel-less
OmniEM model sees the image: a size-1 `c` is squeezed; a size-3 `c` with identical
channels is collapsed automatically; a size > 1 `c` with differing channels
requires `--color-to-gray`, which averages the channels. `--color-to-gray` is the
only place RGB→gray reduction happens — the Python API never takes a `c` axis.

#### Non-Conforming XY Shapes

`omniemv1` expects square XY input whose side is a multiple of the OmniEM stride
(`112` for `omniemv1`). The `--conform` option controls what happens when input
does not match:

| Mode | Behavior |
|---|---|
| `strict` | reject non-square / non-stride-multiple XY |
| `pad` | pad bottom/right, run OmniEM, crop back to original XY |
| `resize` | resize XY before inference, then resize output back |

`pad` keeps the original image geometry and crops the output back afterward.
`resize` is useful when interpolation is acceptable.

### Important Arguments

| Argument | Meaning |
|---|---|
| `-i`, `--input` | input image, `.tif`, `.tiff`, or `.npy`; 2D=`yx`, 3D=`zyx` |
| `-m`, `--model` | OmniEM config YAML |
| `--weights` | merged whole state dict |
| `--backbone` | split encoder/backbone state dict |
| `--head` | split head/adapters state dict |
| `--axes AXES` | declare the input layout (`{b,c,z,y,x}`, `y`/`x` required); default infers 2D=`yx` / 3D=`zyx` |
| `--color-to-gray` | average a size>1 channel (`c`) axis to grayscale before the channel-less OmniEM model |
| `--scale dtype\|max\|S` | scale axis -> `[0,1]` (default `dtype`); or `--unit-range` to skip |
| `--norm model\|per-image\|prenormalized` | norm source (default `model`); or `--mean M --std S` |
| `--out-dtype uint8\|uint16` | dtype for transformed `task_type` output; default `uint8` |
| `--save-logits` | also write raw float logits to `<OUT>.logits.npy` |
| `--conform strict\|pad\|resize` | shape handling for non-conforming XY; default `strict` |
| `--output-scale F` | resize input XY by factor `F` (>0, finite) before inference; super-res when `F>1`; XY only (warns for 3D); orthogonal to `--conform` |
| `-o`, `--output-path` | output path, `.tif`, `.npy`, or `.npz` |
| `--force` | overwrite existing output and sidecar |

### Example: Segmentation With Split Weights

```bash
EXAMPLES=examples
WEIGHTS=weights

omniem infer \
  -i "$EXAMPLES/2d_MitoEM_H_0_0_0.tif" \
  -m model_mito-seg-ViT-L-2D.yaml \
  --backbone "$WEIGHTS/backbone_emdino_v1.pt" \
  --head "$WEIGHTS/head_mito-seg-ViT-L-2D.pt" \
  -o out/mito_labels.tif
```

This writes:

- `out/mito_labels.tif`: the label map;
- `out/mito_labels.tif.json`: a sidecar with config, weights, scaling, norm,
  conform mode, and output metadata. The axes are recorded as `source_axes` (the
  declared/inferred input layout), `forward_axes` (the channel-less layout passed
  to OmniEM), `channel_reduction`, and `color_to_gray`.

Internally the CLI uses the same channel-less stages as `OmniEM.run`, in a single
forward. With `--save-logits`, it reuses that same forward to write both the task
output and the float-logits sibling.

No `--scale` is needed here: a `uint8` tile uses the default `--scale dtype` (divide
by 255 into `[0,1]`), which matches these `[0,1]`-domain OmniEM configs.

### Example: Restoration / Denoise

```bash
omniem infer \
  -i "$EXAMPLES/2d_MitoEM_H_0_0_0.tif" \
  -m model_denoise-emdiffuse-l.yaml \
  --backbone "$WEIGHTS/backbone_emdino_v1.pt" \
  --head "$WEIGHTS/head_denoise-emdiffuse-l.pt" \
  --save-logits \
  -o out/mito_denoised.tif
```

For `task_type: image2image`, the main output is a restored image. With
`--save-logits`, raw logits are also written beside it:

```text
out/mito_denoised.tif
out/mito_denoised.logits.npy
out/mito_denoised.tif.json
```

### Example: Merged Weights

```bash
omniem infer \
  -i "$EXAMPLES/2d_MitoEM_H_0_0_0.tif" \
  -m model_mito-seg-ViT-L-2D.yaml \
  --weights merged_mito-seg.pt \
  -o out/mito_labels.tif
```

Do not pass `--weights` together with `--backbone` or `--head`.

### Example: Non-Conforming Input

```bash
omniem infer \
  -i "$EXAMPLES/gly-z=0.tif" \
  -m model_denoise-emdiffuse-l.yaml \
  --backbone "$WEIGHTS/backbone_emdino_v1.pt" \
  --head "$WEIGHTS/head_denoise-emdiffuse-l.pt" \
  --conform resize \
  -o out/gly_denoised.tif
```

OmniEM sees a conformed square input, but the file written to disk is returned to
the original XY shape. The sidecar records both the original and conformed XY.

### Example: Output-Size Control (`--output-scale`)

OmniEM models are **shape-preserving** (output XY == input XY). To get a larger
output — for example for super-resolution — resize the input up first with
`--output-scale F`; OmniEM then returns its output at the scaled size:

```bash
omniem infer \
  -i "$EXAMPLES/2d_MitoEM_H_0_0_0.tif" \
  -m model_superreso-emdiffuse-l.yaml \
  --backbone "$WEIGHTS/backbone_emdino_v1.pt" \
  --head "$WEIGHTS/head_superreso-emdiffuse-l.pt" \
  --output-scale 1.5 \
  -o out/mito_1.5x.tif
```

Notes:

- The flag resizes the **input** XY by `F` (bicubic), then runs the
  shape-preserving OmniEM. The example input `2d_MitoEM_H_0_0_0.tif` is `224×224`, so
  `--output-scale 1.5` yields a `336×336` output (`= 3 × 112`, a multiple of stride
  `112`, so it runs under the default `--conform strict`). `F < 1` downsizes (a
  quick-inference speed trade-off).
- **Orthogonal to `--conform`.** `--output-scale` only resizes to the target; a
  scaled size that is **not** a multiple of stride `112` (e.g. `--output-scale 1.25`
  → `280`) is rejected under the default `--conform strict`. Pass
  `--conform pad|resize` to make such a size runnable.
- **Not recommended for 3D** (`zyx`): the resize touches XY only and leaves Z
  untouched, which changes in-plane spatial information relative to Z (anisotropy)
  with no Z alignment / resampling — it warns and still runs.
- The bicubic resize can push values slightly outside `[0, 1]`; the `[0, 1]` range
  check is warn-only (`OmniEMWarning`), so the run continues unchanged.
- The sidecar records `output_scale: {factor, input_yx, scaled_yx}`.

### Example: Raw Logits Only

If the config has `task_type: null` or omits `task_type`, the CLI writes float
logits only. In that mode, `--out-dtype` and `--save-logits` are rejected because
there is no derived output transform.

```bash
omniem infer \
  -i examples/2d_MitoEM_H_0_0_0.tif \
  -m model_without_task_type.yaml \
  --weights model.pt \
  -o out/logits.npy
```

## Split / Merge Weights

Use `omniem split` / `omniem merge` to convert between the two weight layouts: a
single **merged** whole `.pt` and the **split** backbone + head pair. Both
wrap the config-based `OmniEM.load` + `OmniEM.save_weights` round-trip.

```bash
# split: merged -> backbone + head
omniem split \
  -m CONFIG.yaml \
  -i merged.pt \
  --backbone out/backbone.pt \
  --head out/head.pt

# merge: backbone + head -> merged
omniem merge \
  -m CONFIG.yaml \
  --backbone backbone.pt \
  --head head.pt \
  -o out/merged.pt
```

`split` prints the two output paths (backbone first, then head); `merge` prints the
one merged output path.

### Important Arguments

| Argument | Meaning |
|---|---|
| `-m`, `--model` | OmniEM config YAML (**required** — see note below) |
| `-i`, `--input` | (`split`) the merged `.pt` to split |
| `-o`, `--output` | (`merge`) the merged `.pt` to write |
| `--backbone` | the backbone (encoder) state dict — output for `split`, input for `merge` |
| `--head` | the head + adapters state dict — output for `split`, input for `merge` |
| `--force` | overwrite existing output file(s) |

`-m CONFIG` is required because the backbone/head boundary is OmniEM's **derived
encoder prefix** — it is read from the net the config builds, not assumed to be
`vit.*`. The same config that loads an OmniEM is the one that splits or merges its
weights.

The three file paths of a command must be distinct (e.g. `--backbone` and `--head`
cannot be the same file, and an output cannot overwrite an input); a collision is
rejected before anything is written.
