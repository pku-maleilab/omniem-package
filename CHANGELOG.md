# Changelog

## [0.1.1]

A unified, **channel-less, two-tier** input/output pipeline. This is a **breaking**
change to the public surface; the numerics are unchanged. Inputs are grayscale, so
`axes` is drawn from `{b, z, y, x}` (no `c`). The canonical compute tensor is
`[b, z, y, x]` (ZYX), and model logits are `[b, c_out, z, y, x]`.

### Added

- **`OmniEM.run(image, *, axes, norm=None, conform="strict", squeeze="", dtype=None,
  return_logits=False)`**, the everyday path. It prepares, computes, and recovers from
  a raw image in one call. `return_logits=False` gives the `task_type` output, and
  `return_logits=True` gives caller-layout float logits.
- **`EMEncoder.run(image, *, axes, norm=None, conform="strict", squeeze="", ...)`**, the
  encoder equivalent, returning the CLS / patch / inner feature dict.
- **Channel-less `OmniEM.predict(tensor)` / `EMEncoder.forward(tensor)`**, the power
  path: pure compute on a pre-built canonical `[b, z, y, x]` tensor. Encoder features
  come back as `cls [B, Z, D]` / `patch [B, Z, N, D]` / `inner[i] [B, Z, T, D]`.
- **`squeeze`** (a subset of `{b, z}`) drops a singleton batch or depth axis from the
  output.
- **CLI.** `omniem infer` gained `--axes` and `--color-to-gray` (matching `features`),
  and records `source_axes` / `forward_axes` / `channel_reduction` / `color_to_gray`
  in the sidecar.

### Changed

- **Normalization is scalar-only.** `norm={'mean': m, 'std': s}` requires scalars; a
  per-channel sequence is rejected (EM is grayscale).

### Removed (breaking)

- The `Prepared` carrier and the `omniem.prepared` module.
- `OmniEM.apply_input` / `OmniEM.apply_output` and `EMEncoder.apply_input`, now folded
  into `run`. The raw one-shots `predict(x, axes=…)`, `enc.forward(x, axes=…)`, and
  `enc(x, axes=…)` are gone; each removed call raises `InputContractError` with a
  migration message.

### Migration

- Use `run(image, axes=…)` for a raw image, or build a canonical `[b, z, y, x]` tensor
  and call `predict(canonical)` / `enc.forward(canonical)`. Consumers (`omniem-train`,
  `napari-omniem`) migrate in their own repos; the numerics are unchanged, so any saved
  parity goldens stay valid.

## [0.1.0]

First tagged release: the GUI-free encoder / model / head surface that `omniem-train`
and other downstream tools build on. A model is fully specified by **(config,
weights)**, with no bundle, no metadata, no tag.

### Added

- **Encoder.** `EMEncoder`: `load(arch, weights)` (raw `vit.*` checkpoint, loaded
  directly), single-shot `forward` → CLS / patch / inner-block / register features,
  `apply_input` (split-out input transform), and `name_parameter_group`. The
  owner-frozen encoder arch catalog: `list_encoders` / `arch_info` (`emdinov1`).
- **Model.** `OmniEM` (EM-DINO encoder + STAdapter z-fusion + UNETR decoder):
  `load` / `from_config` (optional, separable weight loading: merged, or
  encoder-/head-only, or none → random init), `predict` (single-shot forward →
  **pure logits** at the caller's shape), `apply_input`, the `task_type`-gated output
  stage `apply_output` (`image2image` → sigmoid+uint image; `image2label` → argmax
  label map), `save_weights` (merged or backbone+head split), and `prepare_train`
  (training handoff: unfreeze, optionally fix the encoder backbone). The owner-frozen
  model arch catalog: `list_models` / `model_arch_info` (`omniemv1`).
- **Shared-encoder borrow.** `OmniEM.load` / `from_config` accept `encoder=` (a
  pre-built `EMEncoder`) to share one ViT backbone by reference across many heads
  (memory-efficient). Head-only load; borrowed models are read-only (whole-model
  mutators rejected) so the shared encoder is never mutated.
- **Input conform round-trip.** `predict` / `apply_input` accept
  `conform={'strict','pad','resize'}` so non-square / non-stride-multiple XY is handled
  gracefully and the output is round-tripped to the original shape.
- **Output-size control (CLI super-resolution).** `omniem infer --output-scale F`
  bicubic-resizes the input XY by `F` before inference; since the model is
  shape-preserving, the output lands at the scaled size (`F>1` super-resolution,
  `F<1` quick-inference). XY only; Z is never resized; 3D (`zyx`) inputs warn
  (anisotropy / no Z alignment) and still run. Orthogonal to `--conform`; CLI-only
  (the Python API resizes the input directly; see `docs/api.md`). The resize is
  recorded in the infer sidecar (`output_scale: {factor, input_yx, scaled_yx}`).
- **CLI** (`omniem`): `list-encoders`, `list-models`, `features` (single-shot encoder
  feature extraction), `infer` (single-shot model inference with `--weights` merged or
  `--backbone`/`--head` split, `--conform`, `--output-scale`,
  `--scale`/`--unit-range`/`--norm`/`--mean`/`--std`, `--out-dtype`, `--save-logits`),
  and `split` / `merge` (weight-file utilities: split a merged `.pt` into a
  `--backbone` + `--head` pair, or merge a pair back; the boundary is the net's
  derived encoder prefix, not a hardcoded `vit.`). Each `features`/`infer` run writes a
  store + a JSON reproducibility sidecar.
- **Typed error taxonomy** under `OmniEMError`: `ConfigError`, `WeightFormatError`,
  `MissingExtraError`, `InputContractError`, `OOMError`.
- **Config.** `omniem.config.ModelConfig` (+ `BaseConfig`) with YAML I/O and a
  schema-version policy.


### Deferred to Stage 2 (v0.2+)

- Application-level tiling (`Inferer`), volume streaming, feature-export (`Exporter`),
  and the `[infer]` / `[volume]` / `[full]` extras (which debut with their code).
- Encoder-swap proof (a second encoder family) is deferred to a later release; v0.1
  ships and tests a single encoder (`emdinov1`).
