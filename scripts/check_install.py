#!/usr/bin/env python
"""Sanity check for the MLIP_FTL installation.

Run after `install.sh` (or any manual install):

    python scripts/check_install.py [--cpu]

Use --cpu to skip the CUDA availability check for CPU-only installs.
"""

from __future__ import annotations

import argparse
import sys

OK = "\u2713"
FAIL = "\u2717"

failures = []


def check(label, fn, warn_only=False):
    try:
        detail = fn()
    except Exception as exc:  # noqa: BLE001
        marker = "!" if warn_only else FAIL
        print(f"{marker} {label}: {exc}")
        if not warn_only:
            failures.append(label)
        return False
    print(f"{OK} {label}" + (f" ({detail})" if detail else ""))
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true", help="skip the CUDA check")
    args = parser.parse_args()

    def torch_check():
        import torch

        return f"version {torch.__version__}"

    def cuda_check():
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "torch.cuda.is_available() is False. "
                "For CPU-only installs re-run with --cpu."
            )
        return torch.cuda.get_device_name(0)

    def pyg_check():
        import torch_cluster  # noqa: F401
        import torch_geometric
        import torch_scatter  # noqa: F401
        import torch_sparse  # noqa: F401

        return f"torch_geometric {torch_geometric.__version__}"

    def fairchem_check():
        import warnings

        # fairchem model modules emit torch.load FutureWarnings on import
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            import fairchem.core  # noqa: F401

            from fairchem.core.common.registry import registry
            from fairchem.core.common.utils import setup_imports

            setup_imports()
        if registry.get_model_class("esen_backbone") is None:
            raise RuntimeError("esen_backbone model is not registered")
        return None

    def extras_check():
        import ase  # noqa: F401
        import pymatgen.core  # noqa: F401
        import sklearn  # noqa: F401

        return None

    check("PyTorch imported", torch_check)
    if args.cpu:
        print(f"{OK} CUDA check skipped (--cpu)")
    else:
        check("CUDA available", cuda_check)
    check("PyG imported", pyg_check)
    check("FairChem imported", fairchem_check)
    check("Data tools imported (ase, pymatgen, scikit-learn)", extras_check)

    if failures:
        print(f"\n{FAIL} Installation check FAILED: {', '.join(failures)}")
        return 1
    print(f"\n{OK} MLIP_FTL ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
