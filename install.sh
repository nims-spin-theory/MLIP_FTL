#!/usr/bin/env bash
# One-step installer for MLIP_FTL.
#
# Tested configuration:
#   Python 3.9 / PyTorch 2.4.1 / CUDA 12.4 (cu124) / Linux x86_64
#
# Usage (inside an activated conda env, e.g. `conda activate MLIP_FTL`):
#   bash install.sh          # GPU install (CUDA 12.4 wheels)
#   bash install.sh --cpu    # CPU-only install
#
# For other CUDA/PyTorch combinations see "Advanced installation" in README.md.

set -euo pipefail

TORCH_VERSION="2.4.1"
MODE="gpu"

for arg in "$@"; do
    case "$arg" in
        --cpu) MODE="cpu" ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown option: $arg (use --cpu or --help)" >&2
            exit 1
            ;;
    esac
done

cd "$(dirname "$0")"

echo "==> Checking Python environment"
if ! command -v python >/dev/null 2>&1; then
    echo "ERROR: no 'python' found. Create and activate the conda env first:" >&2
    echo "    conda create -n MLIP_FTL python=3.9 && conda activate MLIP_FTL" >&2
    exit 1
fi
python - <<'EOF'
import sys
if not ((3, 9) <= sys.version_info[:2] <= (3, 12)):
    sys.exit(f"ERROR: Python {sys.version.split()[0]} is unsupported; use 3.9-3.12 (3.9 recommended).")
print(f"Using Python {sys.version.split()[0]} at {sys.executable}")
EOF

if [[ "$MODE" == "gpu" ]]; then
    CUDA_TAG="cu124"
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "==> NVIDIA driver detected:"
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
    else
        echo "WARNING: nvidia-smi not found. If this machine has no NVIDIA GPU," >&2
        echo "         re-run with:  bash install.sh --cpu" >&2
    fi
else
    CUDA_TAG="cpu"
fi

TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"
PYG_INDEX="https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html"

echo "==> Installing PyTorch ${TORCH_VERSION} (${CUDA_TAG})"
pip install "torch==${TORCH_VERSION}" torchvision torchaudio --index-url "$TORCH_INDEX"

echo "==> Installing modified FairChem core"
pip install -e packages/fairchem-core

echo "==> Installing PyTorch Geometric extensions (${CUDA_TAG} wheels)"
pip install torch-scatter torch-sparse torch-spline-conv -f "$PYG_INDEX"
pip install torch-cluster torch_geometric -f "$PYG_INDEX"

echo "==> Installing MLIP_FTL dependencies"
pip install ase_db_backends seaborn scikit-learn

echo "==> Running installation check"
if [[ "$MODE" == "cpu" ]]; then
    python scripts/check_install.py --cpu
else
    python scripts/check_install.py
fi

echo
echo "Installation complete. Try the examples in examples_scripts/."
