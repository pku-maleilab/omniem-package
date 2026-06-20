# Changelog


## [0.1.0]


### Added

- **Encoder.** `EMEncoder` — `load(arch, weights)` (raw `vit.*` checkpoint, loaded
  directly), single-shot `forward` → CLS / patch / inner-block features,
  `apply_input` (split-out input transform), and `name_parameter_group`. The
  owner-frozen encoder arch catalog: `list_encoders` / `arch_info` (`emdinov1`).
- **Model.** `OmniEM` (EM-DINO encoder + STAdapter z-fusion + UNETR decoder) —
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
  `F<1` quick-inference). XY only — Z is never resized; 3D (`zyx`) inputs warn
  (anisotropy / no Z alignment) and still run. Orthogonal to `--conform`; CLI-only
  (the Python API resizes the input directly — see `docs/api.md`). The resize is
  recorded in the infer sidecar (`output_scale: {factor, input_yx, scaled_yx}`).
- **CLI** (`omniem`): `list-encoders`, `list-models`, `features` (single-shot encoder
  feature extraction), `infer` (single-shot model inference with `--weights` merged or
  `--backbone`/`--head` split, `--conform`, `--output-scale`,
  `--scale`/`--unit-range`/`--norm`/`--mean`/`--std`, `--out-dtype`, `--save-logits`),
  and `split` / `merge` (weight-file utilities: split a merged `.pt` into a
  `--backbone` + `--head` pair, or merge a pair back — the boundary is the net's
  derived encoder prefix, not a hardcoded `vit.`). Each `features`/`infer` run writes a
  store + a JSON reproducibility sidecar.
- **Typed error taxonomy** under `OmniEMError`: `ConfigError`, `WeightFormatError`,
  `MissingExtraError`, `InputContractError`, `OOMError`.
- **Config.** `omniem.config.ModelConfig` (+ `BaseConfig`) with YAML I/O and a
  schema-version policy.


