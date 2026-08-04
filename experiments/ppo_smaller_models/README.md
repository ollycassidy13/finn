# TFC and CNV PPO Folding Experiments

This directory preserves the two smaller resource-aware PPO experiments used
before the TinyDeiT run. The shared runner starts from the quantized ONNX model,
runs the FINN dataflow preparation flow, selects folding factors for the target
FPS and board budget, and writes an `experiment_result.json` summary.

## Models

- **TFC-W1A1** uses FINN's existing
  `src/finn/qnn-data/build_dataflow/model.onnx`. It is byte-for-byte identical
  to the retained `tfc-w1a1.onnx` experiment input.
- **CNV-W2A2** uses `models/cnv-w2a2.onnx`, originally published in
  `Xilinx/finn-examples` v0.0.1 under `onnx-models-bnn-pynq`.

The default settings reproduce the experiment inputs and targets:

| Model | Board | Clock | Target | Current estimate |
| --- | --- | ---: | ---: | --- |
| TFC-W1A1 | Pynq-Z1 | 100 MHz | 100,000 FPS | 111,607 estimated FPS |
| CNV-W2A2 | ZCU102 | 100 MHz | 500 FPS | 514.403 estimated FPS |

The retained implemented CNV folding measured 578.704 estimated FPS and
578.684 interval-based RTLSIM FPS, a 0.0035% difference. The current search
selects a lower-resource target-meeting folding as the analytical models and
deterministic refinement have evolved.

## Estimate-only runs

Run inside the FINN Docker environment from the repository root:

```bash
./run-docker.sh python \
  experiments/ppo_smaller_models/run_ppo_experiment.py tfc \
  --output-dir /tmp/finn-ppo-tfc

./run-docker.sh python \
  experiments/ppo_smaller_models/run_ppo_experiment.py cnv \
  --output-dir /tmp/finn-ppo-cnv
```

An estimate-only run exits unsuccessfully if the selected folding does not meet
the requested target.

## RTLSIM and synthesis

Pass `--implement` to continue through IP generation, automatic FIFO sizing,
stitched-IP RTLSIM, and out-of-context synthesis. Vivado/Vitis must be configured
for these runs. Implemented-mode validation requires:

- a complete multi-frame RTLSIM run with no timeout or unfinished streams;
- estimated FPS within 5% of steady-state RTLSIM FPS;
- both RTLSIM and timing-adjusted OOC throughput meeting the target; and
- non-negative OOC synthesis timing slack.

```bash
./run-docker.sh python \
  experiments/ppo_smaller_models/run_ppo_experiment.py cnv \
  --output-dir /tmp/finn-ppo-cnv-implemented \
  --implement
```

RTLSIM throughput is calculated from the interval between completed frames. The
OOC synthesis does not directly simulate frame throughput, so it validates
timing and resources; the report's achieved-Fmax throughput is included as
additional headroom evidence. Target error and estimate-versus-RTLSIM error are
written separately because legal folding factors can intentionally overshoot a
target. The runner accepts `--model` and the other command-line overrides for
rerunning PPO on a replacement TFC or CNV ONNX model.
