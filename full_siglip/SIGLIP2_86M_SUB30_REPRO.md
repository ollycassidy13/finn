# SigLIP2 86M Sub-30 ms Reproduction

This records the source path for regenerating the proven W6A7 SigLIP2-86M
VCK190 build from a QONNX model. Generated build outputs are intentionally not
tracked.

## Inputs

The build expects a SigLIP2 base patch16 224 static ImageNet QONNX model. The
checked-in config defaults to this path:

```text
full_siglip/build/static_qat_siglip2_86m_patch16_224_w6a7_qvlsq_featurehead_50k_50k_export/qat_static_imagenet_qonnx.onnx
```

Use that path for the original run, or override `QONNX`, `INPUT_MODEL`,
`EST_DIR`, and `DCP_DIR` after sourcing the config if the model lives
elsewhere.

## Commands

Run from the repository root:

```bash
source full_siglip/configs/siglip2_86m_w6a7_qvlsq_sub30_unpump3top234loop4.env

# Optional when the QONNX is not at the default checked-in recipe path:
# export QONNX=/scratch/path/to/qat_static_imagenet_qonnx.onnx

bash full_siglip/run_siglip2_86m_finn_estimate_from_qonnx.sh
bash full_siglip/run_siglip2_86m_dcp_from_estimate.sh
```

For long local runs, launch the same commands in a named `tmux` session and
write logs under `full_siglip/logs`.

## Proven Recipe

The committed recipe uses:

- `CLOCK_NS=3.999`
- `TARGET_FPS=1200`
- `LOOP_TARGET_FPS=120`
- `MVAU_WWIDTH_MAX=72`
- `MVAU_HLS_NODES=MVAU_0,MVAU_97,MVAU_98`
- `SIGLIP2_86M_STREAM_FEEDBACK=1`
- `MVAU_PUMPED_EXCLUDE_NODES=MVAU_rtl_2,MVAU_rtl_3,MVAU_rtl_4,FINNLoop_0_body_FINNLoop_0_MVAU_rtl_3,FINNLoop_0_body_FINNLoop_0_MVAU_rtl_4`
- `FOLDING_CONFIG_FILE=full_siglip/configs/siglip2_86m_w6a7_qvlsq_sub30_unpump3top234loop4_folding.json`

The folding JSON and pumped-compute exclude list must stay together because the
pumped-compute pass runs after folding-config application.

## Verification

After the estimate and DCP complete, audit the regenerated artifacts:

```bash
python3 full_siglip/audit_siglip2_86m_sub30_goal.py \
  --estimate-dir "${EST_DIR}" \
  --dcp-dir "${DCP_DIR}" \
  --json-out full_siglip/build/siglip2_86m_sub30_goal_status.current.json \
  --md-out full_siglip/build/siglip2_86m_sub30_goal_status.current.md
```

The original passing build reported `29.972477007 ms` at `3.999 ns`, ImageNet
top-1 `72.472%`, a route-clean VCK190 DCP, and timing met on `ap_clk`.
