# omniem

`omniem` is a GUI-free Python package for electron microscopy (EM) image
workflows, introduced from [EM-SSL project](https://github.com/pku-maleilab/EM-SSL-project). It provides two main capabilities:

- **Run OmniEM models** for single-shot segmentation or restoration.
- **Run EM-DINO encoders** to extract CLS, patch, or inner-block features from
  EM images.

Downstream tools build on the same public API: the
[`omniem-train`](https://github.com/pku-maleilab/omniem-train) training pipeline
and the [napari-omniem](https://github.com/pku-maleilab/napari-omniem) GUI plugin.

## Contents

- [Install](#install)
- [Main Features](#main-features)
- [Model Config YAML](#model-config-yaml)
- [First Commands](#first-commands)
- [Full Guides](#full-guides)
- [Related Projects](#related-projects)
- [Future Features](#future-features)
- [License](#license)

## Install

`omniem` requires **Python >= 3.10**.

For inference and feature extraction, CUDA is recommended when you have a
supported NVIDIA GPU. Install the PyTorch build that matches your CUDA driver /
runtime first; use the selector in the
[PyTorch install guide](https://pytorch.org/get-started/locally/) for the exact
command for your machine.

Then install `omniem` from PyPI:

```bash
pip install omniem
```

Or clone the package repository and install it locally:

```bash
git clone https://github.com/pku-maleilab/omniem-package.git
cd omniem-package
pip install .
```

Core runtime dependencies include
[PyTorch](https://pytorch.org/), [NumPy](https://numpy.org/),
[tifffile](https://github.com/cgohlke/tifffile),
[Pydantic](https://docs.pydantic.dev/), [PyYAML](https://pyyaml.org/), and
[MONAI](https://monai.io/).

## Main Features

| Feature | Use it when | Main CLI | Main Python API |
|---|---|---|---|
| Model inference | you have a model config plus model weights and want segmentation, restoration, or raw logits | `omniem infer` | `OmniEM.load(...)`, `model.predict(...)`, `model.apply_output(...)` |
| Encoder features | you only need EM-DINO backbone features, without a model head | `omniem features` | `EMEncoder.load(...)`, `enc(...)` |

### Common Concepts

#### Model = Config + Weights

An OmniEM model is fully specified by a model config YAML plus model weights.
The config describes how to build the head and interpret its output: model
architecture, encoder architecture, 2D/3D shape, output channels, `task_type`,
and the fixed training `mean`/`std` in `[0, 1]` image space.

Weights are plain PyTorch `state_dict` files. They may be split into a shared
EM-DINO backbone file plus a head file, or stored as one merged whole-model file.
Split weights are useful when several heads share one encoder backbone. Merged
weights are convenient when you want one standalone model file.

### Available Models

Model files are distributed outside the Python wheel. Download config YAML files
from [here](https://drive.google.com/drive/folders/1cFPBmozY5VAh8ZgSe16U7ydX9RMmvbzu?usp=drive_link). Download backbone and head weight files from [here](https://drive.google.com/drive/folders/1vpzVk6vDui8Aj34FdTMfJpXbt5wlMsx_?usp=drive_link).

#### Encoder

Use an encoder when you only need the EM-DINO backbone output, without an
OmniEM head or model config. The encoder converts an EM image into feature
tensors that downstream code can reuse:

- `cls`: one global feature vector for the image;
- `patch`: a grid of local patch features;
- `inner`: optional intermediate block features.

For a 2D image, the encoder extracts features from that single XY tile. For a
3D volume, each XY slice is encoded with the same backbone, and the resulting
features are kept alongside the z-axis so downstream code can relate features
back to their original slices.

Available encoder models:

| Encoder arch | Description | Default norm | Input stride | Weights |
|---|---|---|---|---|
| `emdinov1` | EM-DINOv2 ViT-L/14, EM-domain pretrained encoder | mean `0.595446`, std `0.211906` in `[0, 1]` image space | 14 | `backbone_emdino_v1.pt` (bare `vit.*` checkpoint) |

#### OmniEM

Use an OmniEM model when you have a config YAML, model weights, and a 2D or 3D
EM image. The model returns raw logits internally; the config controls whether
`omniem` also applies a canonical output transform.

Available OmniEM models:

| Model | Purpose | Training on | Input | Weights | Config YAML |
|---|---|---|---|---|---|
| `mito-seg-ViT-L-2D` | mitochondria segmentation (2D) | MitoLab dataset | 2D EM tile | `backbone_emdino_v1.pt` + `head_mito-seg-ViT-L-2D.pt` | `model_mito-seg-ViT-L-2D.yaml` |
| `mito-seg-ViT-L-3D` | mitochondria segmentation (3D) | MitoEM-R | 3D subvolume (z >= 16) | `backbone_emdino_v1.pt` + `head_mito-seg-ViT-L-3D.pt` | `model_mito-seg-ViT-L-3D.yaml` |
| `denoise-emdiffuse-l` | image denoise | Low-level denoise EMDiffuse | 2D EM tile | `backbone_emdino_v1.pt` + `head_denoise-emdiffuse-l.pt` | `model_denoise-emdiffuse-l.yaml` |
| `superreso-emdiffuse-l` | image super-resolution | Low-level superresolution EMDiffuse | 2D EM tile | `backbone_emdino_v1.pt` + `head_superreso-emdiffuse-l.pt` | `model_superreso-emdiffuse-l.yaml` |

## Model Config YAML

A model config tells `OmniEM` how to build the model head and how to interpret
outputs.

```yaml
arch: omniemv1
encoder: emdinov1
img_z: 1
out_channels: 2
kernel3d_z: null
task_type: image2label
resize4emdino: false
mean: 0.5333333333333333
std: 0.23137254901960785
```

Field guide:

| Field | Meaning |
|---|---|
| `arch` | model architecture; see `omniem list-models` |
| `encoder` | encoder architecture; see `omniem list-encoders` |
| `img_z` | `1` for 2D heads; `>1` for 3D heads |
| `out_channels` | model output channels |
| `kernel3d_z` | z-kernel for 3D heads; usually `null` for 2D |
| `task_type` | `image2label`, `image2image`, or `null` |
| `resize4emdino` | whether the model uses resize-to-encoder-grid behavior |
| `mean`, `std` | fixed training normalization for this head |

`task_type` controls the canonical output transform:

| `task_type` | Meaning | Output transform |
|---|---|---|
| `image2label` | segmentation / labels | `argmax` over channels |
| `image2image` | restoration / denoise | `sigmoid`, clamp to `[0, 1]`, scale to uint |
| omitted / `null` | model has no output opinion | raw float logits only |

For a denoise/restoration head, `out_channels` is usually `1` and
`task_type: image2image`. For segmentation, `out_channels` is the number of
classes and `task_type: image2label`.

## First Commands

### Get the example inputs, configs, and weights

The commands below read from three local folders. None of them ship inside the
pip wheel, so gather them once before running anything:

| Folder | What it holds | How to get it |
|---|---|---|
| `examples/` | small example EM images (`.tif`) | tracked in the repo (see below) |
| `configs/` | model config YAMLs | Google Drive (see [Available Models](#available-models)) |
| `weights/` | backbone + head weight files | Google Drive (see [Available Models](#available-models)) |

**`examples/`** — if you installed by `git clone`, the example images are already
in `examples/`. If you installed with `pip`, download them into a local
`examples/` folder:

```bash
mkdir -p examples
BASE=https://raw.githubusercontent.com/pku-maleilab/omniem-package/main/examples
curl -L -o examples/2d_MitoEM_H_0_0_0.tif       "$BASE/2d_MitoEM_H_0_0_0.tif"
curl -L -o examples/3d_AxonEM-H-0-0-0_0_0_0.tif "$BASE/3d_AxonEM-H-0-0-0_0_0_0.tif"
curl -L -o "examples/gly-z=0.tif"               "$BASE/gly-z=0.tif"
```

**`configs/` and `weights/`** — these are distributed outside the wheel. Download
the model config YAMLs and the backbone/head weight files from the Google Drive
links in [Available Models](#available-models), then place them in local
`configs/` and `weights/` folders so the paths below resolve:

```text
configs/   model_*.yaml         (config YAMLs)
weights/   backbone_emdino_v1.pt, head_*.pt   (weight files)
```

Run the commands from the directory that contains these `examples/`, `configs/`,
and `weights/` folders.

### Run a model

Run model inference from the CLI:

```bash
omniem infer \
  -i examples/2d_MitoEM_H_0_0_0.tif \
  -m configs/model_mito-seg-ViT-L-2D.yaml \
  --backbone weights/backbone_emdino_v1.pt \
  --head weights/head_mito-seg-ViT-L-2D.pt \
  -o out/mito_labels.tif
```

Run the same model from Python:

```python
import numpy as np
import tifffile
import torch
from omniem import OmniEM

model = OmniEM.load(
    "configs/model_mito-seg-ViT-L-2D.yaml",
    backbone="weights/backbone_emdino_v1.pt",
    head="weights/head_mito-seg-ViT-L-2D.pt",
)

img = tifffile.imread("examples/2d_MitoEM_H_0_0_0.tif")
x = torch.from_numpy(img.astype(np.float32) / 255.0)
logits = model.predict(x, axes="yx")
labels = model.apply_output(logits, axes="yx", dtype="uint8")
```

### Output-size control (super-resolution)

OmniEM models are shape-preserving (output XY == input XY). To get a larger
output, for example super-resolution, resize the input up first with
`--output-scale F`; the model then returns its output at the scaled size
(`F > 1` upscales, `F < 1` is a quick-inference speed trade-off). It is XY-only
(Z is never resized; 3D volumes warn) and orthogonal to `--conform`:

```bash
omniem infer \
  -i examples/2d_MitoEM_H_0_0_0.tif \
  -m configs/model_superreso-emdiffuse-l.yaml \
  --backbone weights/backbone_emdino_v1.pt \
  --head weights/head_superreso-emdiffuse-l.pt \
  --output-scale 1.5 \
  -o out/mito_1.5x.tif
```

### Split or merge weight files

Convert between a merged whole-model `.pt` and a `backbone` + `head` pair. The
boundary is the net's derived encoder prefix, so it is correct for any encoder.

```bash
# merged -> split pair
omniem split -m configs/model_mito-seg-ViT-L-2D.yaml \
  -i weights/merged_mito-seg.pt \
  --backbone weights/backbone_emdino_v1.pt --head weights/head_mito-seg-ViT-L-2D.pt

# split pair -> merged
omniem merge -m configs/model_mito-seg-ViT-L-2D.yaml \
  --backbone weights/backbone_emdino_v1.pt --head weights/head_mito-seg-ViT-L-2D.pt \
  -o weights/merged_mito-seg.pt
```

## Full Guides

- [CLI guide](docs/cli.md): all `omniem infer`, `omniem features`, `omniem split`,
  and `omniem merge` options, with command examples.
- [Python API guide](docs/api.md): `OmniEM`, `EMEncoder`, shared encoders,
  lower-level calls, weight saving, errors, and API-doc generation.

## Related Projects

- [omniem-train](https://github.com/pku-maleilab/omniem-train): the recommended
  training pipeline for OmniEM heads; it builds on this package's public API.
- [napari-omniem](https://github.com/pku-maleilab/napari-omniem): a napari GUI
  plugin for interactive OmniEM inference.

## Future Features

The current package focuses on the core model/encoder surface. These features are
planned for later releases:

- large-image tiling and blending (`Inferer`);
- volume streaming and hdf5/zarr/n5 IO;
- feature-export orchestration (`Exporter`);
- install extras such as `[infer]`, `[volume]`, and `[full]`.

## License

[MIT](LICENSE).
