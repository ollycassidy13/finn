# BERT Safety System Scaffold

## Boundary

The host owns tokenization, embedding lookup, attention-mask handling, policy
thresholds, logging, and dataset/teacher iteration. The FPGA owns the quantized
repeated safety core and returns compact logits to host software.

This split keeps unsupported or frequently changing model pieces outside the
hardware shell while still accelerating the repeated BERT-shaped dense core.

## Flow

1. Train or initialize a student with `train_safety_student.py`.
2. Export the student ONNX, checkpoint, and `student_export_manifest.json`.
3. Generate the FINN core with `make_core_model.py` or through `build.py`.
4. Optionally pass `--weight-state` to quantize matching trained BERT dense
   weights into the FINN core initializers.
5. Build V80 stitched IP, run stitched-IP RTLSIM, and emit the DCP.
6. Integrate the DCP into the wider MCP board shell or benchmark harness.

## Verification Contract

Each build writes deterministic `input.npy`, `expected_output.npy`, and
`expected_context.npz`. DCP and RTL modes request stitched-IP RTLSIM unless
`--no-verify` is explicitly supplied.

The verified artifact checklist for a build directory is:

- `verification_output/verify_stitched_ip_rtlsim_0_SUCCESS.npy`
- `stitched_ip/finn_design.dcp`
- `report/ooc_synth_and_timing.json`

Current checked artifacts:

- `bert/build/training_init_check_v4/student_init.onnx`
- `bert/build/training_init_check_v4/student_export_manifest.json`
- `bert/build/v80_smoke_mlo_dcp_v1/verification_output/verify_stitched_ip_rtlsim_0_SUCCESS.npy`
- `bert/build/v80_smoke_mlo_dcp_v1/stitched_ip/finn_design.dcp`
- `bert/build/v80_mlo_dcp_v1/verification_output/verify_stitched_ip_rtlsim_0_SUCCESS.npy`
- `bert/build/v80_mlo_dcp_v1/stitched_ip/finn_design.dcp`
- `bert/build/max_util_p16s32_dcp_v1/stitched_ip/finn_design.dcp`
- `bert/build/max_util_p16s32_dcp_v1/report/stitched_dcp_utilization.rpt`
- `bert/build/max_util_p16s32_dcp_v1/report/stitched_dcp_timing_4ns_summary.rpt`
- `bert/build/max_util_p16s32_dcp_v1/report/stitched_dcp_timing_5p5ns_summary.rpt`
- `bert/build/max_util_p16s32_fixedfifo_dcp_v1/stitched_ip/finn_design.dcp`

The max-util artifacts are not yet verified handoff targets. The non-fixed FIFO
attempt reached a stitched DCP, but full stitched-IP RTLSIM was manually
stopped at about 90k cycles because the XSIM rate made the estimated
6.636M-cycle first output impractical in this environment. Its synthesized
stitched DCP fails a 4.0 ns clock report with WNS -1.445 ns and passes a
5.5 ns clock report with WNS 0.055 ns.

The fixed-FIFO max-util attempt generated `stitched_ip/finn_design.dcp`, then
failed out-of-context implementation during `place_design`. Vivado reported
88,711 SLLs required across SLR0/SLR1 where 18,870 are available, 33,171 SLLs
required across SLR1/SLR2 where 18,870 are available, and multiple DSP cascade
chains that could not fit inside a clock region or SLR DSP column.

## Open Integration Work

- Train the safety student on the real held-out dataset.
- Run the DCP flow with `--weight-state bert/build/trained --strict-weights`
  once the GPU-trained checkpoint exists. Use a checkpoint whose hidden size,
  FFN size, and layer count match the selected preset.
- Wire the generated DCP into the final V80 shell and host policy service.

## Max-Util Implementation Plan

The current max-util failure is driven by automatic target-FPS folding. The
failing reports show loop-body MVAUs folded to SIMD 768 and SIMD 3072, which
creates long DSP cascade chains and heavy SLR crossings.

Next implementation experiments:

- Run `max-util` with `--no-target-fps` to preserve generated PE/SIMD values.
- Sweep `--target-fps 250/500/1000` with `--mvau-wwidth-max 128/256/512`.
- Keep `--fixed-mlo-fifos --no-verify` for placement/timing exploration only.
- Promote only builds that later pass stitched-IP RTLSIM.
- If capped folding still fails, add SLR-aware floorplanning around the loop
  body and reduce cross-SLR stream widths.

## Verification Plan

Use a ladder rather than full max-util XSIM first:

- `bert.verify_build` checks RTLSIM success, DCP presence, and OOC timing.
- `smoke` remains the fast end-to-end RTLSIM/DCP regression.
- `v80` remains the practical verified V80 DCP regression.
- Full-size builds first prove DCP/timing with `--no-verify`.
- A full-size build becomes a handoff artifact only after stitched-IP RTLSIM or
  board-level validation compares against `expected_output.npy`.

## Host Contract

`host_runtime.py` defines the current service boundary:

- Host produces one UINT8 activation tensor shaped `[1, seq_len - 1, hidden]`.
- FPGA inserts CLS, runs the quantized safety core, selects CLS, and returns
  logits shaped `[1, num_classes]`.
- Host applies softmax, unsafe thresholding, logging, and policy actions.
- Tokenization, embedding lookup, masks, and unsupported BERT pieces remain on
  host until they have supported FINN operators and a verified hardware path.
