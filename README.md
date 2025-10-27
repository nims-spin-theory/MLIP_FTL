# FairChem Frozen Transfer Learning Implementation

[![arXiv](https://img.shields.io/badge/arXiv-2508.20556-b31b1b.svg)](https://arxiv.org/abs/2508.20556)
[![Python](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.md)

## Overview

This repository implements frozen transfer learning for the FairChem framework, specifically for the eSEN (equivariant Smooth Energy Network) model. The implementation enables efficient transfer learning from pre-trained models to new tasks, significantly reducing training time and computational requirements.

### Key Features

- **Frozen Transfer Learning**: Transfer knowledge from pre-trained models while keeping first several layers frozen.
- **MLIP Integration**: Support for using machine learning interatomic potential (MLIP) as base models.
- **FairChem Compatibility**: Built on top of the robust FairChem v1 framework.

### Related Work

This implementation and application are described in detail in our paper:
[arXiv:2508.20556](https://arxiv.org/abs/2508.20556). 

In this paper, we also performed structure optimization and calculated formation energies and distances to the convex hull using machine learning interatomic potentials (MLIPs). For code implementing these computational tasks, please refer to our companion repository [link to be added].

## Installation

### Prerequisites

- Python 3.9
- CUDA 12.4 (for GPU support)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 

### Quick Start

1. **Clone the repository**:
```bash
git clone [this_repo]
cd fairchem
git checkout dev/enda  # Switch to the transfer learning branch
```

2. **Set up the environment**:
```bash
# Create and activate conda environment
conda create -n fairchem python=3.9
conda activate fairchem
```

### GPU Installation (Recommended)

**Note**: If your CUDA version is different from 12.4, please check your CUDA version and modify the installation URLs accordingly. Visit [PyTorch Geometric installation page](https://data.pyg.org/whl/) for available combinations.

```bash
# Load CUDA module if managed by your system
module load cuda/12.4  # Optional: only if CUDA is managed by modules

# Install PyTorch with CUDA support
pip install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install FairChem in development mode
pip install -e packages/fairchem-core[dev]

# Install additional PyTorch dependencies
pip install torch-scatter torch-sparse torch-spline-conv -f https://data.pyg.org/whl/torch-2.4.1+cu124.html
pip install torch-cluster torch_geometric -f https://data.pyg.org/whl/torch-2.4.1+cu124.html

# Install additional dependencies
pip install ase_db_backends
```

### CPU-Only Installation

For systems without GPU support:

```bash
# Install PyTorch CPU version
pip install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install FairChem in development mode
pip install -e packages/fairchem-core[dev]

# Install additional PyTorch dependencies (CPU versions)
pip install torch-scatter torch-sparse torch-spline-conv -f https://data.pyg.org/whl/torch-2.4.1+cpu.html
pip install torch-cluster torch_geometric -f https://data.pyg.org/whl/torch-2.4.1+cpu.html

# Install additional dependencies
pip install ase_db_backends
```


## Usage: Use interface scripts to train models efficiently
This section provides examples of common workflows using interface scripts we created. These examples can provide some hand-on experience and inputs can be modified easily to fit in your research.

First, move to the example folder:
``` bash
cd examples_scripts
```

### 1. Dataset Preparation

The `prepare_data.py` is used to generated train/valid/test datasets for training. If you copy the example folder to other place, please update the path to scirpt accordingly. 

```bash
python ../scripts/prepare_data.py --csv_file my_data.csv --target_property Formation_Energy \
                                  --split_ratios 0.8 0.1 0.1 --output_dir set_formation_train
```

For more flags available, please do `python prepare_data.py -h` for help information.

### 2. Training from Scratch

```bash
python train.py \
   --target_property Formation_Energy \
   --transfer_learning \
   --fl_layer 5 \
```

### 3. Transfer Learning: Formation Energy → Critical Temperature

### 4. Transfer Learning: MLIP → Critical Temperature

For more flags available, please do `python train.py -h` for help information.


## Usage: step-by-step instruction

This section provides examples of common workflows with more details and explains what is done in the scripts provided. All examples include detailed Jupyter notebooks with step-by-step instructions. Please find these notebooks within `examples_notebook`.

### 1. Dataset Preparation

Convert your train dataset to the database format required by FairChem for training.

📁 **Example**: See detailed instructions in `examples_notebook/1_prepare_dataset/`

### 2. Training from Scratch

Train a formation energy model from scratch using the prepared dataset.

📁 **Example**: See detailed instructions in `examples_notebook/2_train_scratch_formE/`

### 3. Transfer Learning: Formation Energy → Critical Temperature

Train a critical temperature (Tc) model using Transfer Learning technique.

The trained formation energy model is used as the base model for transfer learning. 

📁 **Example**: See detailed instructions in `examples_notebook/3_train_TL_Tc/`

### 4. Transfer Learning: MLIP → Critical Temperature
Leverage a pre-trained Machine Learning Interatomic Potential (MLIP) as the base model for critical temperature prediction. 

**Prerequisites**: This example requires the OMAT24 `eSEN-30M-OAM` MLIP model. Please download `esen_30m_oam.pt` from [here](https://huggingface.co/facebook/OMAT24/blob/main/esen_30m_oam.pt) and place it in the example folder before proceeding.

📁 **Example**: See detailed instructions in `examples_notebook/4_train_TL_Tc_MLIP/`

## Troubleshooting

### Common Issues with Frozen Transfer Learning

#### DistributedDataParallel Error

When using frozen transfer learning, you may encounter this error:

```bash
[rank0]: RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. This error indicates that your module has parameters that were not used in producing loss. You can enable unused parameter detection by passing the keyword argument `find_unused_parameters=True` to `torch.nn.parallel.DistributedDataParallel`, and by
[rank0]: making sure all `forward` function outputs participate in calculating loss.
[rank0]: If you already have done the above, then the distributed data parallel module wasn't able to locate the output tensors in the return value of your module's `forward` function. Please include the loss function and the structure of the return value of `forward` of your module when reporting this issue (e.g. list, dict, iterable).
[rank0]: Parameter indices which did not receive grad for rank 0: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40
[rank0]: In addition, you can set the environment variable TORCH_DISTRIBUTED_DEBUG to either INFO or DETAIL to print out information about which particular parameters did not receive gradient on this rank as part of this error
```

**Solution**: This error occurs because frozen layers don't participate in gradient computation. To fix it:

1. **Locate the PyTorch distributed.py file within your environment**:
   `distributed.py` is located at:
   ```bash
    ~/miniconda3/envs/{conda_env_name}/lib/python3.9/site-packages/torch/nn/parallel/distributed.py
   ```
   For example, if your conda environment is named `fairchem` (as shown in the installation instructions above), the file path would be:
   ```bash
   ~/miniconda3/envs/fairchem/lib/python3.9/site-packages/torch/nn/parallel/distributed.py
   ```

2. **Edit the DistributedDataParallel class** (around line 637):

   ```python
   def __init__(
       self,
       module,
       device_ids=None,
       output_device=None,
       dim=0,
       broadcast_buffers=True,
       process_group=None,
       bucket_cap_mb=None,
       find_unused_parameters=True,  # Change from False to True
       check_reduction=False,
       gradient_as_bucket_view=False,
       static_graph=False,
       delay_all_reduce_named_params=None,
       param_to_hook_all_reduce=None,
       mixed_precision: Optional[_MixedPrecision] = None,
       device_mesh=None,
   ):
   ```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{your_paper_2024,
  title={Your Paper Title},
  author={Your Name and Co-authors},
  journal={arXiv preprint arXiv:2508.20556},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

## Acknowledgments

- Built on top of the [FairChem](https://github.com/FAIR-Chem/fairchem) framework
- Thanks to the FairChem team for providing the foundational codebase




