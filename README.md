# FairChem Frozen Transfer Learning Implementation

[![arXiv](https://img.shields.io/badge/arXiv-2508.20556-b31b1b.svg)](https://arxiv.org/abs/2508.20556)
[![Python](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.md)

## Overview

This repository provides a framework for frozen transfer learning in materials science, specifically designed for the eSEN (equivariant Smooth Energy Network) model. The implementation allows you to adopt pre-trained models, includining pretrained machine learning interatomic potentials (MLIPs), for other properties, reducing  dataset size requirements while maintaining good performance. 

### Key Features

- **🧊 Frozen Transfer Learning**: Transfer knowledge from pre-trained models while keeping the first several layers frozen, preserving learned representations
- **🔗 MLIP Integration**: Seamlessly integrate machine learning interatomic potentials (MLIPs) eSEN-30M-OAM as base models for enhanced performance
- **⚡ FairChem Compatibility**: Built on the robust and well-tested FairChem v1 framework, ensuring reliability and extensibility

### Related Work

This implementation and its applications are described in our research paper: [arXiv:2508.20556](https://arxiv.org/abs/2508.20556).

This paper also includes structure optimization and formation energy calculations, as well as distances to the convex hull using machine learning interatomic potentials (MLIPs). For the code for these tasks, please visit our companion repository [link to be added].

## Installation

### Prerequisites

Before getting started, please ensure you have the following installed:

- **[Miniconda](https://docs.conda.io/en/latest/miniconda.html)** - For environment management

### Clone Repo

1. **Clone the repository**:

```bash
git clone [this_repo]
cd fairchem
git checkout dev/TL  # Switch to the transfer learning branch
```

2. **Set up the environment**:

```bash
# Create and activate conda environment
conda create -n fairchem python=3.9
conda activate fairchem
```

### GPU Installation (Recommended)

**Important Note**: This guide assumes CUDA 12.4. If you're using a different CUDA version, please check your version with `nvcc --version` and modify the installation URLs accordingly. Visit the [PyTorch Geometric installation page](https://data.pyg.org/whl/) for compatible combinations.

```bash
# Load CUDA module if managed by your system (optional)
module load cuda/12.4  # Only needed if CUDA is managed by environment modules

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

For systems without GPU support or for testing purposes:

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


## Usage Example: Quick Start with Interface Scripts

This section shows common workflows using our interface scripts in `scripts` folder. These examples provide hands-on experience and can be adapted for your research needs.

**Workflow Overview:**

1. **Train from scratch**: Build a formation energy model and use it for predictions
2. **Transfer learning (Model → Model)**: Train a critical temperature model using the formation energy model as base
3. **Transfer learning (MLIP → Model)**: Train a critical temperature model using the pre-trained eSEN-30M-OAM MLIP as base

First, navigate to the examples folder containing input csv files:

```bash
cd examples_scripts
```

### 1. Dataset Preparation

The `prepare_data.py` script converts your dataset into the format required for training. If you copy the examples folder elsewhere, please update the path to the `prepare_data.py` script accordingly.

```bash
python ../scripts/prepare_data.py  --csv_file database_example_train.csv \
                        --material_id  UUID \
                        --target_property "formation energy (eV/atom)" \
                        --split_ratios 0.8 0.1 0.1
```

The `--csv_file` flag specifies the input CSV file containing the training dataset. The file must include the following columns:

1. **Structure definition columns** (fixed names):
   - `cell`: Unit cell parameters
   - `positions`: Atomic positions
   - `numbers`: Atomic numbers

2. **Property and identifier columns** (customizable names):
   - Target property column: Contains the values you want to predict
   - Material ID column: Contains unique identifiers (formulas, UUIDs, or labels)

The target property and material ID column names are arbitrary and should match the values passed to the `--target_property` and `--material_id` flags respectively.

Please refer to the example CSV files for the format requirements.

For complete information on available parameters, run `python prepare_data.py -h`.

### 2. Training Formation Energy Model from Scratch

**Single GPU training**
```bash
python ../scripts/train.py --data_dir "set_formation_energy_(eV_atom)_train" \
                --material_id  UUID \
                --target_property "formation energy (eV/atom)" \
                --num_layers 5 --max_epochs 50
```

**Key Parameters:**
- `--data_dir`: Specifies the data directory created by `prepare_data.py`
- `--num_layers`: Specifies the number of message-passing layers within the model
- `--max_epochs`: Specifies the number of training epochs

The training process outputs detailed information explaining each procedure step. Model performance is evaluated using the test set (data not seen during training) and visualized in a generated PNG figure. Upon completion, the script provides the trained model checkpoint path and exports test results to a CSV file.

**Multi-GPU training**

```bash
python ../scripts/train.py --data_dir "set_formation_energy_(eV_atom)_train" \
                --material_id  UUID \
                --target_property "formation energy (eV/atom)" \
                --num_layers 5 --max_epochs 50 \
                --num_gpu 2
```

**CPU-only training**
```bash
python ../scripts/train.py --data_dir "set_formation_energy_(eV_atom)_train" \
                --material_id  UUID \
                --target_property "formation energy (eV/atom)" \
                --num_layers 5 --max_epochs 50 \
                --cpu_only
```

**Apply the trained model for predictions**:

First, prepare the LMDB file containing the structures for prediction. The `--apply` flag specifies prediction application mode:

```bash
python ../scripts/prepare_data.py  --csv_file database_example_apply.csv \
                  --material_id  UUID \
                  --apply
```
Then apply the trained model to make predictions:

```bash
python ../scripts/train.py --apply \
                --model_path "result_formation_energy_(eV_atom)/checkpoints/2025-10-28-19-42-08-formation_energy_(eV_atom)_MPL5/checkpoint.pt" \
                --lmdb_path "set_apply/apply.lmdb" \
                --material_id UUID
```

- `--apply`: Specifies the prediction application mode
- `--model_path`: Specifies the path to the trained model checkpoint file
- `--lmdb_path`: Specifies the path to the LMDB file containing compounds for prediction

The script provides the path to output at the end.

#### Tips

**Customizing Output Settings**

You can customize the output directory and job name using the `--output_dir` and `--job_name` flags. If not specified, these will be automatically generated based on the target property name, as demonstrated in the examples above.

**Selecting GPU Device**

If multiple GPUs are available, you can specify which GPU to use with the `--gpu-id` flag. The script displays available GPU information at startup.

**Other Available Parameters**

For complete information on available parameters, run `python python train.py -h`.


### 3. Transfer Learning: Formation Energy → Critical Temperature

First, prepare the dataset for critical temperature training:

```bash
python ../scripts/prepare_data.py  --csv_file database_example_trainTL.csv \
                        --material_id  UUID \
                        --target_property "Tc (K)(KKR-FULL)" \
                        --split_ratios 0.8 0.1 0.1
```

Now train the critical temperature model using transfer learning. Make sure to update the path to your formation energy model checkpoint:

```bash
python ../scripts/train.py --data_dir "set_Tc_(K)(KKR-FULL)_train" \
                --material_id  UUID \
                --target_property "Tc (K)(KKR-FULL)" \
                --num_layers 5 --max_epochs 100 \
                --transfer_learning \
                --fl_layer 2 \
                --base_model "result_formation_energy_(eV_atom)/checkpoints/2025-10-28-19-42-08-formation_energy_(eV_atom)_MPL5/checkpoint.pt"
```

**Key Parameters:**
- `--transfer_learning`: Enables transfer learning mode
- `--base_model`: Specifies the path to the base model checkpoint file
- `--fl_layer`: Specifies the number of frozen layers (layers that remain unchanged during training)


### 4. Transfer Learning: MLIP → Critical Temperature

**Prerequisites**: This example requires the OMAT24 `eSEN-30M-OAM` MLIP model. Please download `esen_30m_oam.pt` from the [OMAT24 Hugging Face repository](https://huggingface.co/facebook/OMAT24/blob/main/esen_30m_oam.pt) and place it in the examples folder before proceeding.

```bash
python ../scripts/train.py --data_dir "set_Tc_(K)(KKR-FULL)_train" \
                --material_id  UUID \
                --target_property "Tc (K)(KKR-FULL)" \
                --num_layers 10 --max_epochs 100 \
                --transfer_learning \
                --fl_layer 7 \
                --base_model "./esen_30m_oam.pt"
```

For complete information on available parameters, run `python train.py -h`.


## Usage: Detailed Step-by-Step Instructions

This section provides examples with explanations of the underlying processes. Each workflow includes Jupyter notebooks with step-by-step instructions for learning and customization.

### 1. Dataset Preparation

Learn how to convert your dataset into the FairChem-compatible format required for training.

📁 **Tutorial**: Detailed instructions available in `examples_notebook/1_prepare_dataset/`

### 2. Training from Scratch

Build and train a formation energy model from the ground up using your prepared dataset.

📁 **Tutorial**: Detailed instructions available in `examples_notebook/2_train_scratch_formE/`

### 3. Transfer Learning: Formation Energy → Critical Temperature

Train a critical temperature (Tc) model using a pre-trained formation energy model as the base.

📁 **Tutorial**: Instructions available in `examples_notebook/3_train_TL_Tc/`

### 4. Transfer Learning: MLIP → Critical Temperature

Use pre-trained Machine Learning Interatomic Potentials (MLIPs) for critical temperature prediction.

**Prerequisites**: Download the OMAT24 `eSEN-30M-OAM` MLIP model (`esen_30m_oam.pt`) from the [OMAT24 Hugging Face repository](https://huggingface.co/facebook/OMAT24/blob/main/esen_30m_oam.pt) and place it in the examples folder.

📁 **Tutorial**: Instructions available in `examples_notebook/4_train_TL_Tc_MLIP/`

## Troubleshooting

### Common Issues and Solutions

#### DistributedDataParallel Error with Frozen Transfer Learning

When using frozen transfer learning, you may encounter the following error:

```bash
[rank0]: RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. This error indicates that your module has parameters that were not used in producing loss. You can enable unused parameter detection by passing the keyword argument `find_unused_parameters=True` to `torch.nn.parallel.DistributedDataParallel`, and by
[rank0]: making sure all `forward` function outputs participate in calculating loss.
[rank0]: If you already have done the above, then the distributed data parallel module wasn't able to locate the output tensors in the return value of your module's `forward` function. Please include the loss function and the structure of the return value of `forward` of your module when reporting this issue (e.g. list, dict, iterable).
[rank0]: Parameter indices which did not receive grad for rank 0: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40
[rank0]: In addition, you can set the environment variable TORCH_DISTRIBUTED_DEBUG to either INFO or DETAIL to print out information about which particular parameters did not receive gradient on this rank as part of this error
```

**Solution**: This error occurs because frozen layers don't participate in gradient computation, causing PyTorch's distributed training to fail. Here's how to fix it:

1. **Locate the PyTorch distributed.py file**:

   The file is located at:

   ```bash
   ~/miniconda3/envs/{conda_env_name}/lib/python3.9/site-packages/torch/nn/parallel/distributed.py
   ```

   For example, if using the `fairchem` environment from our installation guide:

   ```bash
   ~/miniconda3/envs/fairchem/lib/python3.9/site-packages/torch/nn/parallel/distributed.py
   ```

2. **Edit the DistributedDataParallel class** (around line 637):

   Change `find_unused_parameters=False` to `find_unused_parameters=True`:

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

And please cite the FairChem paper specified at
`https://pypi.org/project/fairchem-core/1.10.0/`:
```bibtex
@article{ocp_dataset,
    author = {Chanussot*, Lowik and Das*, Abhishek and Goyal*, Siddharth and Lavril*, Thibaut and Shuaibi*, Muhammed and Riviere, Morgane and Tran, Kevin and Heras-Domingo, Javier and Ho, Caleb and Hu, Weihua and Palizhati, Aini and Sriram, Anuroop and Wood, Brandon and Yoon, Junwoong and Parikh, Devi and Zitnick, C. Lawrence and Ulissi, Zachary},
    title = {Open Catalyst 2020 (OC20) Dataset and Community Challenges},
    journal = {ACS Catalysis},
    year = {2021},
    doi = {10.1021/acscatal.0c04525},
}
```
and eSEN model paper:
```bibtex
@misc{fu2025learningsmoothexpressiveinteratomic,
      title={Learning Smooth and Expressive Interatomic Potentials for Physical Property Prediction}, 
      author={Xiang Fu and Brandon M. Wood and Luis Barroso-Luque and Daniel S. Levine and Meng Gao and Misko Dzamba and C. Lawrence Zitnick},
      year={2025},
      eprint={2502.12147},
      archivePrefix={arXiv},
      primaryClass={physics.comp-ph},
      url={https://arxiv.org/abs/2502.12147}, 
}
```



## Acknowledgments

- Built on top of the [FairChem](https://github.com/FAIR-Chem/fairchem) framework v-1.10.0. Thanks to the FairChem team for providing this powerful framework.




