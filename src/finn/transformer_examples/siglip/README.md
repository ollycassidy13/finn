# SigLIP image-tower example

This example compiles the static image path of
`google/siglip2-base-patch16-224` with FINN. It accepts a QONNX model exported
by the QV-LSQ quantization flow, detects the 12 repeated vision-transformer
blocks, rolls them into a `FINNLoop`, and builds estimates or VCK190 hardware.
The quantizer and its checkpoints are deliberately not bundled with FINN.

The build uses FINN's phase API. It replaces the preparation and optimization
phases with SigLIP-specific variants, then uses the standard FINN hardware
phases:

1. `phase_prepare_siglip`
2. `phase_optimize_siglip`
3. `phase_convert_to_hardware`
4. `phase_optimize_hardware`
5. `phase_build_hardware` for RTL/OOC builds
6. `phase_generate_outputs` for RTL/OOC builds

The SigLIP optimization phase extracts LayerNorm scale/bias and exposes
quantized attention projections to FINN's existing hardware conversions. A
model-specific step before loop rolling then locates the repeated vision
blocks, rejects topology mismatches, and writes `siglip_mlo_ranges.json` into
the build directory.

## Input contract

The default profile expects:

- model: `google/siglip2-base-patch16-224`
- static image input: `[1, 3, 224, 224]`
- 12 vision-transformer blocks
- QV-LSQ quantization: W6A7, with 8-bit edge layers
- QONNX `Quant` nodes and inferred static tensor shapes
- an `image_embeds` graph output (the optional `class_scores` branch is removed)

Use the canonical, shape-inferred QONNX output from the quantization flow. Pass
its `qat_report.json` with `--quantization-report` to check model identity,
precision, depth, image size, and quantizer provenance before FINN starts.

Validate that handoff without starting a build:

```bash
./run-docker.sh python -m finn.transformer_examples.siglip.build \
  --model build/siglip/qat_static_imagenet_qonnx.onnx \
  --quantization-report build/siglip/qat_report.json \
  --output-dir build/siglip/validation \
  --validate-only
```

## Quick start

Run commands from the FINN repository root. Place the QONNX model and report in
a path visible inside the FINN container, such as the checkout or
`$FINN_HOST_BUILD_DIR`.

Generate deterministic reference vectors:

```bash
./run-docker.sh python -m finn.transformer_examples.siglip.make_test_vectors \
  --model build/siglip/qat_static_imagenet_qonnx.onnx \
  --output-dir build/siglip/vectors
```

Run the phase-based estimate flow with Python checks:

```bash
./run-docker.sh python -m finn.transformer_examples.siglip.build \
  --model build/siglip/qat_static_imagenet_qonnx.onnx \
  --quantization-report build/siglip/qat_report.json \
  --output-dir build/siglip/estimate \
  --mode estimate \
  --verify python \
  --input-npy build/siglip/vectors/input.npy \
  --expected-output-npy build/siglip/vectors/expected_output.npy
```

Generate stitched IP and run functional RTL simulation:

```bash
./run-docker.sh python -m finn.transformer_examples.siglip.build \
  --model build/siglip/qat_static_imagenet_qonnx.onnx \
  --quantization-report build/siglip/qat_report.json \
  --output-dir build/siglip/stitched_ip \
  --mode stitched_ip \
  --verify rtlsim \
  --input-npy build/siglip/vectors/input.npy \
  --expected-output-npy build/siglip/vectors/expected_output.npy
```

Generate a routed out-of-context VCK190 checkpoint and record ideal-memory RTL
performance:

```bash
./run-docker.sh python -m finn.transformer_examples.siglip.build \
  --model build/siglip/qat_static_imagenet_qonnx.onnx \
  --quantization-report build/siglip/qat_report.json \
  --output-dir build/siglip/ooc \
  --mode ooc_synth \
  --verify rtlsim \
  --measure-rtlsim-performance \
  --input-npy build/siglip/vectors/input.npy \
  --expected-output-npy build/siglip/vectors/expected_output.npy
```

Vivado 2024.2 or newer is required for MLO. OOC synthesis can take several
hours; run it in a persistent session. The RTL performance mode uses an ideal
AXI-MM memory model and is an upper bound, not a board measurement.

## Configuration

The `configs` directory contains:

- `siglip2_base_patch16_224_w6a7_qv_lsq.json`: model, quantization, build, and
  evidence contract;
- `siglip2_base_patch16_224_w6a7_vck190_specialization.json`: RTL preferences
  with the three verified HLS MVAU exceptions;
- `siglip2_base_patch16_224_w6a7_vck190_folding.json`: initial folding and
  memory settings for the matching QV-LSQ graph.

The folding file is node-name-specific. If a new exporter changes graph names,
start from the generated `auto_folding_config.json`, review the differences,
and create a new profile instead of silently applying this one to another
graph. The build applies matching settings before shuffle decomposition, then
requires every profile entry to match the decomposed graph before continuing.

The default Python verification tolerance is `0.25`. On the deterministic
seed-0 vector, the measured maximum raw-QONNX difference is `0.186622` after
preparation and `0.182954` after SigLIP optimization. This tolerance is only a
conversion smoke-check bound; it is not an accuracy measurement.

## Reference results

These values describe the exact W6A7 QV-LSQ profile above. They are reference
evidence, not values inferred by the scripts and not promises for a modified
model.

| Measurement | Result | Scope |
| --- | ---: | --- |
| ImageNet-1K top-1 | 72.472% | 50,000 images, static zero-shot comparison head |
| ImageNet-1K top-5 | 92.228% | same evaluation |
| FINN modeled latency | 29.972477007 ms | 7,494,993 cycles at 3.999 ns, overlapped MLO schedule |
| VCK190 OOC timing | met at 250.062 MHz | routed, zero route errors |
| Board-runtime throughput | not measured | no board run is claimed |
| Ideal-memory RTL throughput | not measured | enable `--measure-rtlsim-performance` to produce it |

For a new run, use the generated accuracy report, `report` directory,
`rtlsim_performance.json`, and OOC timing/route reports as the authoritative
measurements. Do not derive throughput from the target FPS setting.
