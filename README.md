# Implementation of Frozen transfer Learning to FairChem (eSEN model)

## 1. Introduction

This repo introduces the code in this work `https://arxiv.org/abs/2508.20556`.
The code for ML interatomic potential optimization and calculation of formation
energy and distance to hull convex can be found in another repo of ours.

## 2. Installation
We used `git` and `conda` for version control. Since these are common tools for
coding and scientific computation, the details are not provided here. `git` is
commonly installed already in the system. For conda, please install it following
the instruction on official website. `miniconda` is recommended.  

First, please get the source file using `git clone`.
```bash
git clone    XXXXX
git checkout dev/enda # switch to branch of transfer learning feature.
```

Please note that the implementation is based on FairChem. The `main` branch is
the standard `FairChem v1` package. The **modifications for transfer learning**
is contained in the branch `dev/enda`. Thus, you can easily see how modification
is done and do further development. 

Below is a full script used to install the `FairChem` in `dev` mode with GPU on
Linux and Mac. To install the package on a computer without GPU, just change
`cuXXX` to `cpu`.

If your `cuda` version is not 12.4, please check the `cuda` version and do
 modification to the number after `cu`. For a bit more details, you can open
`https://data.pyg.org/whl/` to find combinations of `torch` and `cuda`
available.  

```bash
# if your cuda is managed by module, load the module first
module load cuda/12.4 

# create conda enviroment
conda create -n fairchem python=3.9
# activate conda environment
conda activate  fairchem

# install torch
pip install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# install fairchem using the sorce file in git reop 
cd fairchem
pip install -e packages/fairchem-core[dev]

# install other dependence
pip install torch-scatter torch-sparse torch-spline-conv -f https://data.pyg.org/whl/torch-2.4.1+cu124.html
pip install torch-cluster torch_geometric                -f https://data.pyg.org/whl/torch-2.4.1+cu124.html
pip install ase_db_backends

# the tests provided within FairChem
pytest tests/core
```

## 3. Usage: Convert train dataset to database for training
This is mainly shown using the example in `example/prepare_database` .

In these examples, we used 1000 data points of formation energy as the train
dataset. This is just for illustration. Although a good performance is obtained
for formation energy with this relatively small dataset, 1000 might be not
enough for other properties. The dataset is from DXMag Computational HeuslerDB
`https://www.nims.go.jp/group/spintheory/database/`.


## 4. Usage: Train a model: from scratch or transfer learning
This is mainly shown using the example in `example/train` .




## 5. Possible problem and fix
### 1. Error message when TL is used.
When frozen transfer learning feature is used, you might meet this error message.

``` python
[rank0]: RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. This error indicates that your module has parameters that were not used in producing loss. You can enable unused parameter detection by passing the keyword argument `find_unused_parameters=True` to `torch.nn.parallel.DistributedDataParallel`, and by
[rank0]: making sure all `forward` function outputs participate in calculating loss.
[rank0]: If you already have done the above, then the distributed data parallel module wasn't able to locate the output tensors in the return value of your module's `forward` function. Please include the loss function and the structure of the return value of `forward` of your module when reporting this issue (e.g. list, dict, iterable).
[rank0]: Parameter indices which did not receive grad for rank 0: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40
[rank0]:  In addition, you can set the environment variable TORCH_DISTRIBUTED_DEBUG to either INFO or DETAIL to print out information about which particular parameters did not receive gradient on this rank as part of this error
```

To fix it,  please change `find_unused_parameters=False` to `find_unused_parameters=True` in 
`init`  of `class DistributedDataParallel(Module, Joinable)`  in the `torch` module file `distributed.py`.

This file is located at
`~/miniconda3/envs/{conda_env_name}/lib/python3.9/site-packages/torch/nn/parallel/distributed.py`
(line 637).  

If the conda environment is created using the same name as in this README file,
the `conda_env_name` is `fairchem`.

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
        find_unused_parameters=False, ## <- change this to True
        check_reduction=False,
        gradient_as_bucket_view=False,
        static_graph=False,
        delay_all_reduce_named_params=None,
        param_to_hook_all_reduce=None,
        mixed_precision: Optional[_MixedPrecision] = None,
        device_mesh=None,
    ):

```


## 5. Usage: Application of model for prediction
This is mainly shown using the example in `example/application` .

Here, we applied the obtained ML model to predict for formation energy for 500 compounds not included in our train dataset. The ML prediction is compared to DFT results.  


## Acknowledgement 



