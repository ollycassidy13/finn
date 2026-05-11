# BERT Safety FINN Flow

This directory is the bring-up path for a BERT-shaped MCP safety accelerator on
V80. The current hardware graph uses initialized INT8 weights so the FINN flow
can be verified now; the GPU step later fine-tunes a student with a safety
teacher and exports the trained checkpoint artifacts.

## Components

- `train_safety_student.py`: exports an initialized student today and supports
  later teacher distillation with hard labels plus KL loss.
- `make_core_model.py`: emits the quantized FINN core with `AddCLSToken`,
  rolled repeated encoder-style blocks, `SelectToken`, and a classifier head.
- `weight_import.py`: maps shape-compatible trained BERT dense weights into
  the FINN core and quantizes them to INT8 initializers.
- `build.py`: generates deterministic reference I/O and runs FINN estimate,
  stitched-IP RTLSIM, and DCP flows for V80.
- `system.md`: records how this accelerator fits into the wider MCP safety
  system.

All Hugging Face, Torch, dataset, and build caches are kept under `bert/cache`
or `bert/build`. Do not point model downloads at `/home`.

## Run

Run commands from the FINN repository root through the project Docker wrapper:

```bash
source ./vivado-env-setup.sh
FINN_DOCKER_PREBUILT=1 FINN_SKIP_DEP_REPOS=1 ./run-docker.sh \
  python -m bert.train_safety_student \
  --init-only --random-init --output-dir bert/build/training_init_check
FINN_DOCKER_PREBUILT=1 FINN_SKIP_DEP_REPOS=1 ./run-docker.sh \
  python -m bert.build \
  --preset smoke --mode dcp --output-dir bert/build/v80_smoke_dcp
FINN_DOCKER_PREBUILT=1 FINN_SKIP_DEP_REPOS=1 ./run-docker.sh \
  python -m bert.build \
  --preset v80 --mode dcp --output-dir bert/build/v80_dcp
```

The full-size paper geometry uses the same entry point. This is the current
stress target, not the routine regression target:

```bash
FINN_DOCKER_PREBUILT=1 FINN_SKIP_DEP_REPOS=1 ./run-docker.sh \
  python -m bert.build \
  --preset max-util --mode dcp --target-fps 2000 \
  --output-dir bert/build/max_util_mlo_dcp \
  --liveness-threshold-cycles 20000000
```

The full max-util RTLSIM is very slow under XSIM. Use it only when a long
run is acceptable; use `--no-verify` and optionally `--fixed-mlo-fifos` for
DCP/timing exploration, then rely on the smoke/V80 targets for routine
stitched-IP RTLSIM regression checks.

## Bring-Up Results

These artifacts were generated from initialized weights, not trained safety
weights:

| Build directory | Verification | Timing |
| --- | --- | --- |
| `bert/build/training_init_check_v4` | `student_init.onnx` and `student_export_manifest.json` exported in Docker. | N/A |
| `bert/build/v80_smoke_mlo_dcp_v1` | Stitched-IP RTLSIM passed and `stitched_ip/finn_design.dcp` exists. | 306.28 MHz, WNS 0.735 ns |
| `bert/build/v80_mlo_dcp_v1` | Stitched-IP RTLSIM passed and `stitched_ip/finn_design.dcp` exists. | 302.48 MHz, WNS 0.694 ns |
| `bert/build/max_util_p16s32_dcp_v1` | Stitched DCP generated, but full RTLSIM was stopped at about 90k cycles because XSIM throughput made the 6.636M-cycle first-output estimate impractical here. | 4.0 ns report fails WNS -1.445 ns; 5.5 ns report passes WNS 0.055 ns |
| `bert/build/max_util_p16s32_fixedfifo_dcp_v1` | Stitched IP DCP generated with `--fixed-mlo-fifos --no-verify`, but out-of-context implementation failed placement. | No closed OOC timing; placement failed on SLR/SLL overuse and long DSP cascade chains. |

## Presets

| Preset | Shape | Purpose |
| --- | --- | --- |
| `smoke` | 2 layers, hidden 8, sequence 4 | Fast DCP and RTLSIM sanity target. |
| `v80` | 4 layers, hidden 64, sequence 16 | Regular V80 DCP validation target. |
| `paper` | 12 layers, hidden 768, sequence 128 | BERT-base geometry without aggressive unfolding. |
| `max-util` | 12 layers, hidden 768, sequence 128 | Paper geometry with PE=16/SIMD=32 V80 unfolding. |

An earlier PE=6/SIMD=32 max-util attempt generated a DCP but timed out in
stitched-IP RTLSIM at 20M cycles with no output. The current max-util preset
uses a 128-bit loop output stream and PE=16/SIMD=32. It generates the intended
paper-sized stress design, but it is not yet a closed target.

The max-util stitched DCP report from `stitched_dcp_utilization.rpt` uses
about 425k CLB LUTs, 625k registers, 2387 DSP58s, 3456.5 BRAM tiles, and 1
URAM. The non-fixed FIFO attempt fails a 4.0 ns report and passes a 5.5 ns
synthesized checkpoint report at 181.8 MHz. The fixed-FIFO attempt avoids the
long FIFO-sizing simulation but fails `place_design`: Vivado reports 88,711
SLLs required across SLR0/SLR1 where 18,870 are available, plus DSP cascade
chains that do not fit inside a clock region or SLR DSP column. Full RTLSIM and
closed OOC implementation are still open for this target.

The next max-util implementation work is physical, not training-related:
reduce or floorplan SLR crossings, break the long DSP cascade chains, or reduce
PE/SIMD until the full-size design can place. The verified handoff targets for
now are `v80_smoke_mlo_dcp_v1` and `v80_mlo_dcp_v1`.

## GPU Training

Use JSONL rows with `text` and integer `label` fields:

```json
{"text": "example prompt", "label": 0}
{"text": "unsafe prompt", "label": 1}
```

Then run on a GPU node:

```bash
python -m bert.train_safety_student \
  --dataset-jsonl bert/data/safety_train.jsonl \
  --student bert-base-uncased \
  --teacher unitary/toxic-bert \
  --teacher-tokenizer unitary/toxic-bert \
  --output-dir bert/build/trained
```

The training export writes `student_init.onnx` and
`student_export_manifest.json`; training runs also write `dataset_split.json` so
the held-out validation split is reproducible.

To import trained weights into the FINN core after fine-tuning, pass the saved
student checkpoint directory to the build:

```bash
FINN_SKIP_DEP_REPOS=1 ./run-docker.sh python -m bert.build \
  --preset paper --mode dcp --output-dir bert/build/paper_trained_dcp \
  --weight-state bert/build/trained --strict-weights
```

The importer currently maps dense attention-output, FFN expand/project, and
classifier weights when their shapes match the selected preset. It writes
`quantized_weight_manifest.json` with imported, missing, and mismatched tensors.
Use `paper` or `max-util` for BERT-base-shaped checkpoints; the smaller `smoke`
and `v80` presets are bring-up targets unless you train a matching smaller
student.
