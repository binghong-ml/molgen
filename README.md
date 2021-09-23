# tree_based_molecule_generation

## 1. Setting up the environment
You can set up the environment by following commands. You need to specify cudatoolkit version and torch geometric versions accordinly to your local computing device.

```
conda create -n mol python=3.7
source ~/.bashrc
conda activate mol
conda install -y pytorch cudatoolkit=10.1 -c pytorch
conda install -y tqdm
conda install -y -c conda-forge neptune-client
conda install -y -c conda-forge rdkit

pip install pytorch-lightning
pip install neptune-client[pytorch-lightning]

pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html
pip install torch-geometric

pip install cython
pip install molsets

```

## 2. Executing the scripts
You can execute the scripts in the following order.

```
cd molgen/src/
CUDA_VISIBLE_DEVICES=${GPU} bash ../script/generator_moses_hparam0.sh
CUDA_VISIBLE_DEVICES=${GPU} bash ../script/generator_moses_hparam1.sh
CUDA_VISIBLE_DEVICES=${GPU} bash ../script/generator_moses_hparam2.sh
CUDA_VISIBLE_DEVICES=${GPU} bash ../script/generator_moses_hparam3.sh
CUDA_VISIBLE_DEVICES=${GPU} bash ../script/generator_moses_hparam4.sh
CUDA_VISIBLE_DEVICES=${GPU} bash ../script/generator_moses_hparam5.sh
CUDA_VISIBLE_DEVICES=${GPU} bash ../script/generator_zinc_hparam0.sh
CUDA_VISIBLE_DEVICES=${GPU} bash generator_zinc_hparam1.sh
CUDA_VISIBLE_DEVICES=${GPU} bash generator_qm9_hparam0.sh
CUDA_VISIBLE_DEVICES=${GPU} bash generator_qm9_hparam1.sh
CUDA_VISIBLE_DEVICES=${GPU} bash smiles_generator_moses.sh
CUDA_VISIBLE_DEVICES=${GPU} bash smiles_generator_zinc.sh
CUDA_VISIBLE_DEVICES=${GPU} bash smiles_generator_qm9.sh
```
