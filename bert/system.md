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
- Fix or relax the max-util target: add physical partitioning around the loop
  body, reduce cross-SLR stream width/fanout, break long DSP cascade chains,
  or reduce PE/SIMD until the full-size design places.
- Find a tractable full-graph verification strategy for max-util, such as a
  faster simulator, a staged loop-body verification, or a smaller repeated
  proof target plus board-level validation.
- Wire the generated DCP into the final V80 shell and host policy service.
