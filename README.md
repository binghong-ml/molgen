# tree_based_molecule_generation

## 1. Setting up the environment
You can set up the environment by following commands. You need to specify cudatoolkit version and torch geometric versions accordinly to your local computing device.

```
conda create -n mol python=3.7
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