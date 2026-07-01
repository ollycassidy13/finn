#!/bin/bash
export NUM_DEFAULT_WORKERS=4
script_dir=$(realpath $(dirname "${BASH_SOURCE[0]}") )
export FINN_HOST_BUILD_DIR=${script_dir}/scripts_finn/finn/finn_temp_file
export FINN_XILINX_PATH="/proj/xbuilds/SWIP/2024.2_1113_1001/installs/lin64/"
export FINN_XILINX_VERSION="2024.2"
export FINN_DOCKER_EXTRA=" -v /proj/xbuilds/licenses:/proj/xbuilds/licenses -e XILINXD_LICENSE_FILE=/proj/xbuilds/licenses "
export XILINX_VIVADO="${FINN_XILINX_PATH}/Vivado/${FINN_XILINX_VERSION}"
export VIVADO_PATH="${XILINX_VIVADO}"
export VITIS_PATH="${FINN_XILINX_PATH}/Vitis/${FINN_XILINX_VERSION}"
export VITIS_HLS="${FINN_XILINX_PATH}/Vitis_HLS/${FINN_XILINX_VERSION}"
export HLS_PATH="${VITIS_HLS}"
export XILINXD_LICENSE_FILE="${XILINXD_LICENSE_FILE:-/proj/xbuilds/licenses}"

if [ -f "${XILINX_VIVADO}/settings64.sh" ]; then
  # shellcheck disable=SC1090
  source "${XILINX_VIVADO}/settings64.sh"
else
  export PATH="${XILINX_VIVADO}/bin:${PATH}"
fi
